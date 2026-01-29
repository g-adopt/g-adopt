import warnings
from pathlib import Path
from typing import Callable, Optional, Union, List

import firedrake as fd
import h5py
import numpy as np
from firedrake.ufl_expr import extract_unique_domain
from pyadjoint.tape import annotate_tape, stop_annotating
from scipy.spatial import cKDTree
from ..utility import log, DEBUG, INFO, log_level, InteriorBC, is_continuous
from ..solver_options_manager import SolverConfigurationMixin

import pygplates

__all__ = [
    "GplatesVelocityFunction",
    "GplatesScalarFunction",
    "LithosphereConnector",
    "pyGplatesConnector"
]


class GPlatesFunctionalityMixin:
    def update_plate_reconstruction(self, ndtime):
        """A placeholder method to update the Function with data from GPlates.
        Updates the function based on plate tectonics data for a given model time.

        This method fetches the plate velocities from the GPlates connector based on
        the provided model time and applies them to the function's data at the top
        boundary nodes.

        Args:
            ndtime (float): The model time for which to update the plate
                velocities. This should be non-dimensionalised time.
        """

        # Print ndtime translated to geological age
        log(f"PyGPlates: Updating surface velocities for {self.gplates_connector.ndtime2age(ndtime):.2f} Ma.")
        # Check if we need to update plate velocities at all
        if self.gplates_connector.reconstruction_age is not None:
            if abs(self.gplates_connector.ndtime2age(ndtime) - self.gplates_connector.reconstruction_age) < self.gplates_connector.delta_t:
                return

        # Assuming `self` is a Firedrake Function instance,
        self.dat.data_with_halos[self.dbc.nodes, :] = (
            self.gplates_connector.get_plate_velocities(
                self.boundary_coords, ndtime)
        )

        # Remove radial component of surface velocities. If annotation is on, do not
        # put this on tape, as we will manually create a block variable for this (see below)
        with stop_annotating():
            self.remove_radial_component()

        # Create block variable only if annotating
        if annotate_tape():
            self.create_block_variable()


class GplatesVelocityFunction(GPlatesFunctionalityMixin, SolverConfigurationMixin, fd.Function):
    """Extends `firedrake.Function` to incorporate velocities calculated by
    Gplates, coming from plate tectonics reconstruction.

    `GplatesVelocityFunction` is designed to associate a Firedrake function with a GPlates
    connector, allowing the integration of plate tectonics reconstructions. This is particularly
    useful when setting "top" boundary condition for the Stokes systems when performing
    data assimilation (sequential or adjoint). Note that we subtract the radial component of the velocity
    field to ensure that the velocity field is tangential to the surface in a FEM sense.

    Attributes:
        dbc (firedrake.DirichletBC): A Dirichlet boundary condition that applies the function
            only to the top boundary.
        boundary_coords (numpy.ndarray): The coordinates of the function located at the
            "top" boundary, normalised, so that it is meaningful for PyGPlates.
        gplates_connector: The GPlates connector instance used for fetching plate
            tectonics data.

    Args:
        function_space: The function space on which the GplatesVelocityFunction is defined.
        gplates_connector: An instance of a pyGplatesConnector, used to integrate
            GPlates functionality or data. See Documentation for pyGplatesConnector.
        top_boundary_marker (defaults to "top"): marker for the top boundary.
        val (optional): Initial values for the function. Defaults to None.
        name (str, optional): Name for the function. Defaults to None.
        dtype (data type, optional): Data type for the function. Defaults to ScalarType.
        quad_degree (int, optional): Quadrature degree for the surface measure. If None,
            defaults to 2 * element_degree + 1.
        solver_parameters (dict, optional): Solver parameters for the tangential projection solver.
            If provided, these will be merged with the default parameters. If None, uses default parameters.
        kwargs (dict, optional): Additional keyword arguments passed to the Firedrake Function.

    Methods:
        update_plate_reconstruction(ndtime):
            Updates the function values based on plate velocities from GPlates
            for a given model time.

            **Note** model time is non-dimensionalised

    Examples:
        >>> gplates_function = GplatesVelocityFunction(V,
        ...                                    gplates_connector=pl_rec_model,
        ...                                    name="GplateVelocity")
        >>> gplates_function.update_plate_reconstruction(ndtime=0.0)
    """
    # We use the default BaseProjector solver parameters.
    tangential_project_solver_parameters = {
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "sub_pc_type": "icc",
        "ksp_rtol": 1e-9,
        "ksp_atol": 1e-12,
        "ksp_max_it": 20,
    }

    # Deciding on the right level of verbosity
    # for the tangential projection solver
    if DEBUG >= log_level:
        # GADOPT_LOGLEVEL=DEBUG: Show convergence reason and monitor
        tangential_project_solver_parameters |= {
            "ksp_converged_reason": None,
            "ksp_monitor": None
        }
    elif INFO >= log_level:
        # GADOPT_LOGLEVEL=INFO: Show only convergence reason
        tangential_project_solver_parameters |= {
            "ksp_converged_reason": None
        }

    def __init__(self, function_space, *args, gplates_connector=None, top_boundary_marker="top", quad_degree=None, solver_parameters=None, **kwargs):
        # Initialise as a Firedrake Function
        super().__init__(function_space, *args, **kwargs)

        # Set the class name required by SolverConfigurationMixin
        self._class_name = self.__class__.__name__

        # Cache all the necessary information that will be used to assign surface velocities
        # the marker for surface boundary. This is typically "top" in extruded mesh.
        self.top_boundary_marker = top_boundary_marker

        # establishing the DirichletBC that will be used to find surface nodes
        # Note: If one day we are moving towards adaptive mesh, we need to be dynamic with this.
        self.dbc = fd.DirichletBC(
            self.function_space(),
            self,
            sub_domain=self.top_boundary_marker)

        # coordinates of surface points
        self.boundary_coords = fd.Function(
            self.function_space(),
            name="coordinates").interpolate(
                fd.SpatialCoordinate(
                    extract_unique_domain(self))).dat.data_ro_with_halos[self.dbc.nodes]
        self.boundary_coords /= np.linalg.norm(self.boundary_coords, axis=1)[:, np.newaxis]

        # Store the GPlates connector
        self.gplates_connector = gplates_connector

        # Store the kwargs and the quad_degree for surface measure
        self.kwargs = kwargs
        self.quad_degree = quad_degree

        # Initialise solver configuration
        self.reset_solver_config(
            default_config=self.tangential_project_solver_parameters,
            extra_config=solver_parameters
        )
        self.register_update_callback(self.setup_solver)

        # Set up the solver for tangential projection
        self.setup_solver()

    def setup_solver(self):
        """Set up the linear variational solver for removing radial component."""
        # Define the solution in the velocity function space
        # If velocity is discontinuous, we need to use a continuous equivalent
        if not is_continuous(self):
            raise ValueError(
                "GplatesVelocityFunction: Velocity field is discontinuous."
                "Removing the radial component of discontinuous velocity fields is not supported."
            )
        else:
            V = self.function_space()

        self.tangential_velocity = fd.Function(V, name="tangential_velocity")

        # Normal vector (radial direction for spherical geometry)
        r = fd.FacetNormal(V.mesh())

        # Define the test and trial functions for a projection solve
        phi = fd.TestFunction(V)
        v = fd.TrialFunction(V)

        # Define the tangential projection: v_tangential = v - (v·r)r
        tangential_expr = self - fd.dot(self, r) * r

        # Getting ds measure: this can be set by the user through quad_degree parameter
        # We are only looking at the surface measure at the top boundary
        if self.quad_degree is not None:
            degree = self.quad_degree
        else:
            degree = 2 * V.ufl_element().degree()[0] + 1

        domain = self.function_space().mesh()

        if domain.extruded:
            self.ds = fd.ds_t(domain=domain, degree=degree)
        else:
            self.ds = fd.ds(domain=domain, degree=degree, subdomain_id=self.top_boundary_marker)

        # Setting up a manual projection
        # Project onto the tangential space: solve for tangential component
        a = fd.inner(phi, v) * self.ds
        L = fd.inner(phi, tangential_expr) * self.ds

        # Setting up boundary condition, problem and solver
        # The field is only meaningful on the boundary, so set zero everywhere else
        self.interior_null_bc = InteriorBC(V, 0., [self.top_boundary_marker])

        self.problem = fd.LinearVariationalProblem(a, L, self.tangential_velocity,
                                                   bcs=self.interior_null_bc,
                                                   constant_jacobian=True)
        self.solver = fd.LinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            options_prefix="gplates_projection"
        )
        self._solver_is_set_up = True

    def remove_radial_component(self):
        """
        Project velocity to tangent plane by removing radial component using linear variational solver.

        This method uses a pre-built linear variational solver for efficiency, avoiding the overhead
        of rebuilding the projection operator each time.
        """
        # Use the a linear variational solver for removing the radial component
        if not getattr(self, "_solver_is_set_up", False):
            self.setup_solver()

        # Project the velocity to the tangential plane
        self.solver.solve()

        # Copy the tangential velocity back to self
        self.dat.data_wo_with_halos[self.dbc.nodes, :] = (
            self.tangential_velocity.dat.data_ro_with_halos[self.dbc.nodes, :]
        )


class pyGplatesConnector(object):
    # Non-dimensionalisation constants
    # characteristic length scale: d
    L = 2890e3  # [m]
    # t [year] X yrs2sec = t [sec]
    yrs2sec = 365 * 24 * 60 * 60
    #   t [Myrs] X myrs2sec = t[sec]
    myrs2sec = 1e6*yrs2sec
    #   L [cm] X cm2m = L [m]
    cm2m = 1e-2
    # minimum distance, bellow which we do not interpolate
    #   this is just to avoid division by zero when weighted averaging
    epsilon_distance = 1e-8
    # minimum magnitude, below which we do not scale the velocities
    # This is to avoid division by zero when scaling the velocities
    eps_rel = 1e-10

    def __init__(self,
                 rotation_filenames,
                 topology_filenames,
                 oldest_age,
                 delta_t=1.,
                 scaling_factor=1.0,
                 nseeds=1e5,
                 nneighbours=4,
                 kappa=1e-6,
                 continental_polygons=None,
                 static_polygons=None):
        """An interface to PyGPlates, used for updating top Dirichlet boundary conditions
        using plate tectonic reconstructions.

        This class provides functionality to assign plate velocities at different geological
        ages to the boundary conditions specified with dbc. Due to potential challenges in
        identifying the plate id for a given point with PyGPlates, especially in high-resolution
        simulations, this interface employs a method of calculating velocities at a number
        of equidistant points on a sphere. It then interpolates these velocities for points
        assigned a plate id. A warning is raised for any point not assigned a plate id.

        Arguments:
            rotation_filenames (Union[str, List[str]]): Collection of rotation file names for PyGPlates.
            topology_filenames (Union[str, List[str]]): Collection of topology file names for PyGPlates.
            oldest_age (float): The oldest age present in the plate reconstruction model.
            delta_t (Optional[float]): The t window range outside which plate velocities are updated.
            scaling_factor (Optional[float]): Scaling factor for surface velocities.
            nseeds (Optional[int]): Number of seed points used in the Fibonacci sphere generation. Higher
                    seed point numbers result in finer representation of boundaries in pygpaltes. Notice that
                    the finer velocity variations will then result in more challenging Stokes solves.
            nneighbours (Optional[int]): Number of neighbouring points when interpolating velocity for each grid point. Default is 4.
            kappa: (Optional[float]): Diffusion constant used for don-dimensionalising time. Default is 1e-6.
            continental_polygons (Optional[Union[str, List[str]]]): Path(s) to continental polygon files.
                    Required for LithosphereConnector to filter continental regions.
            static_polygons (Optional[Union[str, List[str]]]): Path(s) to static polygon files for plate ID assignment.
                    Required for LithosphereConnector to assign plate IDs for rotation.

        Examples:
            >>> connector = pyGplatesConnector(rotation_filenames, topology_filenames, oldest_age)
            >>> connector.get_plate_velocities(ndtime=100)

            >>> # With polygon files for lithosphere tracking
            >>> connector = pyGplatesConnector(
            ...     rotation_filenames, topology_filenames, oldest_age,
            ...     continental_polygons='continental_polygons.gpml',
            ...     static_polygons='static_polygons.gpml'
            ... )
        """

        # Store original filenames for downstream use (e.g., LithosphereConnector)
        self.rotation_filenames = rotation_filenames
        self.topology_filenames = topology_filenames
        self.continental_polygons = continental_polygons
        self.static_polygons = static_polygons

        # Rotation model(s)
        self.rotation_model = pygplates.RotationModel(rotation_filenames)

        # Topological plate polygon feature(s).
        self.topology_features = []
        for fname in topology_filenames:
            for f in pygplates.FeatureCollection(fname):
                self.topology_features.append(f)

        # oldest_age coincides with non-dimensionalised time ndtime=0
        self.oldest_age = oldest_age

        # time window for recalculating velocities
        self.delta_t = delta_t

        # seeds are equidistantial points generate on a sphere
        self.seeds = self._fibonacci_sphere(samples=nseeds)

        # number of neighbouring points that will be used for interpolation
        self.nneighbours = nneighbours

        # PyGPlates velocity features
        self.velocity_domain_features = (
            self._make_GPML_velocity_feature(
                self.seeds
            )
        )

        # last reconstruction time
        self.reconstruction_age = None

        # Factor to scale plate velocities to RMS velocity of model,
        # the definition: RMS_Earth / RMS_Model is used
        # This is specially used in low-Rayleigh-number simulations
        self.scaling_factor = scaling_factor

        # Factor to non-dimensionalise time
        self.kappa = kappa

    # setting the time that we are interested in
    def get_plate_velocities(self, target_coords, ndtime):
        """Returns plate velocities for the specified target coordinates at the top boundary of a sphere,
        for a given non-dimensional time, by integrating plate tectonic reconstructions from PyGPlates.

        This method calculates new plate velocities.
        It utilizes the cKDTree data structure for efficient nearest neighbor
        searches to interpolate velocities onto the mesh nodes.

        Args:
            target_coords (array-like): Coordinates of the points at the top of the sphere.
            ndtime (float): The non-dimensional time for which plate velocities are to be calculated and assigned.
                This time is converted to geological age and used to extract relevant plate motions
                from the PyGPlates model.

        Raises:
            Exception: If the requested ndt ime is a negative age (in the future), indicating an issue with
                the time conversion.

        Notes:
            - The method uses conversions between non-dimensionalised time and geologic age.
            - Velocities are non-dimensionalised and scaled for the simulation before being applied.
        """

        # Raising an error if the user is asking for invalid time
        if self.ndtime2age(ndtime=ndtime) < 0:
            max_time = self.oldest_age / (self.time_dimDmyrs2sec / self.scaling_factor)
            raise ValueError(
                "Input non-dimensionalised time corresponds to negative age (it is in the future)!"
                f" Maximum non-dimensionalised time: {max_time}"
            )

        # cache the reconstruction age
        self.reconstruction_age = self.ndtime2age(ndtime)
        # interpolate velicities onto our grid
        self.interpolated_u = self._interpolate_seeds_u(target_coords)
        return self.interpolated_u

    @property
    def velocity_non_dim_factor(self):
        """
        u [metre/sec] * velocity_non_dim_factor = u []
        """
        return pyGplatesConnector.L / self.kappa

    @property
    def time_dim_factor(self):
        """
        Factor to dimensionalise model time: d^2/kappa
           t [] X time_dim_factor = t [sec]
        """
        return pyGplatesConnector.L**2 / self.kappa

    @property
    def time_dimDmyrs2sec(self):
        """
        Converts the time dimension factor from Myrs (million years) to seconds.

        This method calculates the conversion factor to transform time values
        from Myrs to seconds using the predefined `time_dim_factor` and the
        constant `myrs2sec` from the `pyGplatesConnector` module.

        Returns:
            float: The time dimension factor in seconds.
        """
        return self.time_dim_factor / pyGplatesConnector.myrs2sec

    @property
    def velocity_dimDcmyr(self):
        """
        Converts velocity from cm/year to non-dimensionalised units.

        Returns:
            float: The conversion factor for velocity.
        """
        return self.velocity_non_dim_factor * pyGplatesConnector.cm2m / pyGplatesConnector.yrs2sec

    def ndtime2age(self, ndtime):
        """Converts non-dimensionalised time to age (Myrs before present day).

        Args:
            ndtime (float): The non-dimensionalised time to be converted.

        Returns:
            float: The converted geologic age in millions of years before present day(Ma).
        """

        return self.oldest_age - float(ndtime) * self.time_dimDmyrs2sec / self.scaling_factor

    def age2ndtime(self, age):
        """Converts geologic age (years before present day in Myrs (Ma) to non-dimensionalised time.

        Args:
            age (float): geologic age (before present day in Myrs)

        Returns:
-            float: non-dimensionalised time
        """
        return (self.oldest_age - age) * (self.scaling_factor / self.time_dimDmyrs2sec)

    # convert seeds to Gplate features
    def _make_GPML_velocity_feature(self, coords):
        """Creates a pygplates.GPML velocity feature from specified coordinates.

        This function takes a set of coordinates and converts them
        into a GPML velocity feature, at those points.

        Args:
            coords (numpy.ndarray): An array of coordinates (shape [# of points, 3]) representing
                                    points on a sphere where velocities are to be calculated.

        Returns:
            pygplates.FeatureCollection: A feature collection containing the velocity mesh nodes.
        """

        # Add points to a multipoint geometry
        multi_point = pygplates.MultiPointOnSphere(
            [pygplates.PointOnSphere(x=coords[i, 0], y=coords[i, 1], z=coords[i, 2], normalise=True)
             for i in range(np.shape(coords)[0])]
        )

        # Create a feature containing the multipoint feature, and defined as MeshNode type
        meshnode_feature = pygplates.Feature(pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
        meshnode_feature.set_geometry(multi_point)
        meshnode_feature.set_name('Velocity Mesh Nodes from PyGPlates')

        output_feature_collection = pygplates.FeatureCollection(meshnode_feature)

        return output_feature_collection

    def _interpolate_seeds_u(self, target_coords):
        """Interpolates seed velocities onto target coordinates.

        This method generates a KD-tree of seed points with numerical values, then uses
        this tree to find the nearest neighbors of the target coordinates. It interpolates
        the velocities of these neighbors to the target points, applying a weighted average
        based on distance. If any target point is exceptionally close to a seed point,
        it directly assigns the seed's velocity to avoid division by zero errors.

        Args:
            target_coords (numpy.ndarray): Array of target coordinates for velocity interpolation.

        Returns:
            numpy.ndarray: Interpolated velocities at the target coordinates.
        """
        # calculate velocities here
        seeds_u = self._calc_velocities(
            velocity_domain_features=self.velocity_domain_features,
            topology_features=self.topology_features,
            rotation_model=self.rotation_model,
            age=self.reconstruction_age,
            delta_t=self.delta_t)

        seeds_u = np.array([i.to_xyz() for i in seeds_u])
        seeds_u *= self.velocity_dimDcmyr / self.scaling_factor

        # if PyGPlates does not find a plate id for a point, assings NaN to the velocity
        # here we make sure we only use velocities that are not NaN.
        non_nan_values = ~np.isnan(seeds_u[:, 0])

        # generate a KD-tree of the seeds points that have a numerical value
        tree = cKDTree(data=self.seeds[non_nan_values, :], leafsize=16)

        # find the neighboring points
        dists, idx = tree.query(x=target_coords, k=self.nneighbours)

        # Use np.where to avoid division by very small values
        # Replace tiny distances with pyGplatesConnector.epsilon_distance to avoid division
        # by tiny values while keeping original distances intact for later use
        safe_dists = np.where(dists < pyGplatesConnector.epsilon_distance, pyGplatesConnector.epsilon_distance, dists)

        # Then, calculate the weighted average using safe_dists
        res_u = np.einsum(
            'i, ij->ij',
            1/np.sum(1/safe_dists, axis=1),
            np.einsum('ij, ijk ->ik', 1/safe_dists, seeds_u[non_nan_values][idx])
        )
        close_points_mask = dists[:, 0] < pyGplatesConnector.epsilon_distance
        # Now handle the case where points are too close to each other:
        res_u[close_points_mask, :] = seeds_u[non_nan_values][idx[close_points_mask, 0]]

        # Calculate radial component for res_u: res_u · e_r where e_r is the normalised radial unit vector
        radial_velocity_res = np.sum(res_u * target_coords, axis=1)  # Dot product for each point

        # Store original magnitudes before projection
        original_magnitudes = np.linalg.norm(res_u, axis=1)

        # Project res_u onto tangent plane to remove radial component
        # res_u_tangential = res_u - (res_u · e_r) * e_r
        res_u_tangential = res_u - radial_velocity_res[:, np.newaxis] * target_coords

        # Calculate new magnitudes after projection
        tangential_magnitudes = np.linalg.norm(res_u_tangential, axis=1)

        # Rescale to preserve original magnitude (avoid division by zero)
        scale_factor = np.divide(original_magnitudes, tangential_magnitudes,
                                 where=tangential_magnitudes > pyGplatesConnector.eps_rel)

        res_u = res_u_tangential * scale_factor[:, np.newaxis]

        return res_u

    def _calc_velocities(self, velocity_domain_features, topology_features, rotation_model, age, delta_t):
        """Calculates velocity vectors for domain points at a specific geological age (Myrs before present day).

        This method calculates the velocities for all points in the velocity domain
        at the specified geological age, using PyGPlates to account for tectonic plate
        motions. It utilizes a plate partitioner to determine the tectonic plate each
        point belongs to and computes the velocity based on the rotation model and
        the change over the specified time interval.

        Args:
            velocity_domain_features: Domain features representing the points at which velocities are to be calculated.
            topology_features: Topological features of tectonic plates used in the partitioning.
            rotation_model: The rotation model defining plate movements over ages.
            age (float): The geological age at which velocities are to be calculated.
            delta_t (float): The time interval over which velocity is calculated.

        Returns:
            list of pygplates.Vector3D: Velocity vectors for all domain points.

        Warns:
            RuntimeWarning: If no plate ID is found for a given point, indicating an issue with the reconstruction model.
        """
        # All domain points and associated (magnitude, azimuth, inclination) velocities for the current age.
        all_domain_points = []
        all_velocities = []

        # Gplates works particularly with rounded ages in Myrs and does not work well with ages between
        age_rounded = round(age, 0)

        # Partition our velocity domain features into our topological plate polygons at the current 'age'.
        # Note: PyGPlates can only deal with rounded ages in million years
        plate_partitioner = pygplates.PlatePartitioner(topology_features, rotation_model, age_rounded)

        for velocity_domain_feature in velocity_domain_features:

            # A velocity domain feature usually has a single geometry but we'll assume it can be any number.
            # Iterate over them all.
            for velocity_domain_geometry in velocity_domain_feature.get_geometries():

                for velocity_domain_point in velocity_domain_geometry.get_points():

                    all_domain_points.append(velocity_domain_point)

                    partitioning_plate = plate_partitioner.partition_point(velocity_domain_point)
                    if partitioning_plate:

                        # We need the newly assigned plate ID to get the equivalent stage rotation of that tectonic plate.
                        partitioning_plate_id = partitioning_plate.get_feature().get_reconstruction_plate_id()

                        # Get the stage rotation of partitioning plate from 'age + delta_t' to 'age'.
                        # Note: PyGPlates can only deal with rounded ages in million years
                        equivalent_stage_rotation = rotation_model.get_rotation(age_rounded, partitioning_plate_id, age_rounded + delta_t)

                        # Calculate velocity at the velocity domain point.
                        # This is from 'age + delta_t' to 'age' on the partitioning plate.
                        # NB: velocity unit is fixed to cm/yr, but we convert it to m/yr and further on non-dimensionalise it later.
                        velocity_vectors = pygplates.calculate_velocities(
                            [velocity_domain_point],
                            equivalent_stage_rotation,
                            delta_t, velocity_units=pygplates.VelocityUnits.cms_per_yr)

                        # add it to the list
                        all_velocities.extend(velocity_vectors)
                    else:
                        warnings.warn("PyGPlates couldn't assign plate IDs to some seeds due to irregularities in the reconstruction model. G-ADOPT will interpolate the nearest values.", category=RuntimeWarning)
                        all_velocities.extend([pygplates.Vector3D(np.nan, np.nan, np.nan)])

        return all_velocities

    def _fibonacci_sphere(self, samples):
        """Generates points on a sphere using the Fibonacci sphere algorithm, which
        distributes points approximately evenly over the surface of a sphere.

        This method calculates coordinates for each point using the golden angle,
        ensuring that each point is equidistant from its neighbors. The algorithm
        is particularly useful for creating evenly spaced points on a sphere's
        surface without clustering at the poles, a common issue in other spherical
        point distribution methods.

        Args:
            samples (int): The number of points to generate on the sphere's surface.

        Returns:
            numpy.ndarray: A 2D array of shape (samples, 3), where each row
                           contains the [x, y, z] coordinates of a point on the
                           sphere.

        Example:
            >>> sphere = _fibonacci_sphere(100)
            >>> print(sphere.shape)
            (100, 3)

        Note:
            We use this method for generating seed points that can be interpolated onto our mesh

        References:
            The algorithm is based on the concept of the golden angle, derived from
            the Fibonacci sequence and the golden ratio.
        """

        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        y = 1 - (np.arange(samples)/(samples-1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * np.arange(samples)
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.array([[x[i], y[i], z[i]] for i in range(len(x))])


class LithosphereConnector:
    """Connector for lithosphere indicator field through geological time.

    Combines oceanic lithosphere (age-tracked via gtrack's SeafloorAgeTracker,
    converted to thickness) with continental lithosphere (back-rotated
    present-day thickness data via gtrack's PointRotator) to produce a smooth
    3D indicator field: 1 inside lithosphere, 0 in mantle.

    The indicator uses a tanh transition for numerical stability:
        indicator = 0.5 * (1 + tanh((r - lith_base) / transition_width))

    where lith_base = r_outer - thickness (in mesh units).

    Uses pyGplatesConnector for plate model access and time conversion.

    Attributes:
        gplates_connector: The pyGplatesConnector instance for time conversion and plate model.
        property_name: Name of the thickness property (e.g., 'thickness').
        reconstruction_age: Last computed geological age (Ma), used for caching.

    Args:
        gplates_connector (pyGplatesConnector): Connector with plate model files.
            Must have `continental_polygons` and `static_polygons` set.
        continental_data: Present-day continental thickness data. Can be:
            - gtrack.PointCloud with the thickness property
            - Path to HDF5/NetCDF file
            - Tuple of (latlon_array, thickness_values_km)
        age_to_property (Callable): Function converting seafloor age (Myr) to thickness (km).
            Signature: age_to_property(ages: np.ndarray) -> np.ndarray
        time_step (float): Time step for ocean tracker (Myr). Default 1.0.
        n_points (int): Number of points for ocean tracker mesh. Default 10000.
        reinit_interval_myr (float): Reinitialize ocean tracker every N Myr. Default 50.0.
        k_neighbors (int): Number of neighbors for thickness interpolation. Default 50.
        property_name (str): Name of the thickness property. Default 'thickness'.
        r_outer (float): Outer radius of mesh in mesh units. Default 2.208.
            This is the radial coordinate of Earth's surface in the mesh.
        depth_scale (float): Physical depth (km) corresponding to 1 mesh unit. Default 2890.0.
            For Earth's mantle, 1 non-dimensional unit = 2890 km (mantle depth).
        transition_width (float): Width of tanh transition in km. Default 10.0.
            Controls the smoothness of the lithosphere-mantle boundary.
        default_thickness (float): Thickness (km) for points with no nearby data. Default 100.0.
        distance_threshold (float): Max angular distance for interpolation (radians on unit sphere).
            Points beyond this get default_thickness. Default 0.1.

    Examples:
        >>> def half_space_cooling(age_myr):
        ...     age_sec = age_myr * 3.15576e13
        ...     return 2.32 * np.sqrt(1e-6 * age_sec) / 1e3  # km
        >>>
        >>> lith_connector = LithosphereConnector(
        ...     gplates_connector=plate_model,
        ...     continental_data='SL2013sv.nc',
        ...     age_to_property=half_space_cooling,
        ...     r_outer=2.208,
        ...     depth_scale=2890.0,
        ...     transition_width=10.0,
        ... )
        >>> indicator = lith_connector.get_indicator(mesh_coords, ndtime)
    """

    def __init__(
        self,
        gplates_connector: "pyGplatesConnector",
        continental_data,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        time_step: float = 1.0,
        n_points: int = 10000,
        reinit_interval_myr: float = 50.0,
        k_neighbors: int = 50,
        property_name: str = "thickness",
        r_outer: float = 2.208,
        depth_scale: float = 2890.0,
        transition_width: float = 10.0,
        default_thickness: float = 100.0,
        distance_threshold: float = 0.1,
    ):
        self.gplates_connector = gplates_connector
        self.age_to_property = age_to_property
        self.k = k_neighbors
        self.property_name = property_name
        self.r_outer = r_outer
        self.depth_scale = depth_scale
        self.transition_width = transition_width
        self.default_thickness = default_thickness
        self.distance_threshold = distance_threshold
        self.reinit_interval = reinit_interval_myr

        # Pre-compute transition width in mesh units
        self._transition_width_nondim = transition_width / depth_scale

        # Validate connector has required polygon files
        if gplates_connector.continental_polygons is None:
            raise ValueError(
                "gplates_connector must have continental_polygons set. "
                "Pass continental_polygons to pyGplatesConnector constructor."
            )
        if gplates_connector.static_polygons is None:
            raise ValueError(
                "gplates_connector must have static_polygons set. "
                "Pass static_polygons to pyGplatesConnector constructor."
            )

        # Import gtrack components
        from gtrack import (
            SeafloorAgeTracker, PointRotator, PolygonFilter,
            PointCloud, TracerConfig
        )

        # Store PointCloud class for later use
        self._PointCloud = PointCloud

        # Store n_points for reinitialization
        self._n_points = n_points

        # Create ocean age tracker
        config = TracerConfig(
            time_step=time_step,
            default_mesh_points=n_points,
        )
        self._ocean_tracker = SeafloorAgeTracker(
            rotation_files=gplates_connector.rotation_filenames,
            topology_files=gplates_connector.topology_filenames,
            continental_polygons=gplates_connector.continental_polygons,
            config=config,
        )

        # Create point rotator for continental data
        self._rotator = PointRotator(
            rotation_files=gplates_connector.rotation_filenames,
            static_polygons=gplates_connector.static_polygons,
        )

        # Create polygon filter for continental regions
        self._polygon_filter = PolygonFilter(
            polygon_files=gplates_connector.continental_polygons,
            rotation_files=gplates_connector.rotation_filenames,
        )

        # Load and prepare continental data
        self._continental_present = self._load_continental_data(continental_data)

        # State management
        self._initialized = False
        self._last_reinit_age = None
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

    def ndtime2age(self, ndtime: float) -> float:
        """Convert non-dimensional time to geological age (Ma).

        Delegates to gplates_connector.
        """
        return self.gplates_connector.ndtime2age(ndtime)

    def age2ndtime(self, age: float) -> float:
        """Convert geological age (Ma) to non-dimensional time.

        Delegates to gplates_connector.
        """
        return self.gplates_connector.age2ndtime(age)

    def get_indicator(
        self,
        target_coords: np.ndarray,
        ndtime: float
    ) -> np.ndarray:
        """Get smooth lithosphere indicator at target coordinates for given time.

        Returns a 3D field that is ~1 inside the lithosphere and ~0 in the mantle,
        with a smooth tanh transition at the lithosphere base.

        Checks if time has changed significantly (using gplates_connector.delta_t)
        before recomputing. Returns cached result if time hasn't changed enough.

        Args:
            target_coords: (M, 3) array of mesh coordinates in mesh units.
            ndtime: Non-dimensional time.

        Returns:
            (M,) array of indicator values (0 to 1).
        """
        age = self.ndtime2age(ndtime)

        log(f"LithosphereConnector: Computing lithosphere indicator for {age:.2f} Ma.")

        # Check if we can use cached result
        if self.reconstruction_age is not None:
            if abs(age - self.reconstruction_age) < self.gplates_connector.delta_t:
                # Check if coordinates are the same (using hash of shape and sample)
                coords_hash = (target_coords.shape, target_coords[0, 0] if len(target_coords) > 0 else 0)
                if self._cached_result is not None and coords_hash == self._cached_coords_hash:
                    log(f"LithosphereConnector: Using cached result.")
                    return self._cached_result

        # Initialize ocean tracker if needed
        if not self._initialized:
            starting_age = self.gplates_connector.oldest_age
            log(f"LithosphereConnector: Initializing ocean tracker at {starting_age} Ma.")
            self._ocean_tracker.initialize(starting_age=starting_age)
            self._initialized = True
            self._last_reinit_age = starting_age

        # Check for reinitialisation
        if self._last_reinit_age is not None:
            if abs(self._last_reinit_age - age) >= self.reinit_interval:
                log(f"LithosphereConnector: Reinitializing ocean tracker at {age:.2f} Ma.")
                self._ocean_tracker.reinitialize(n_points=self._n_points)
                self._last_reinit_age = age

        # Get ocean lithosphere
        ocean_cloud = self._ocean_tracker.step_to(age)
        ocean_ages = ocean_cloud.get_property('age')
        ocean_values = self.age_to_property(ocean_ages)
        ocean_cloud.add_property(self.property_name, ocean_values)

        # Get rotated continental lithosphere
        cont_cloud = self._rotator.rotate(
            self._continental_present,
            from_age=0.0,
            to_age=age
        )

        # Combine into single PointCloud
        # warn=False because ocean cloud doesn't have plate_ids (expected behavior)
        # and only has 'age' property while continental has 'thickness'
        combined = self._PointCloud.concatenate([ocean_cloud, cont_cloud], warn=False)

        # Compute smooth lithosphere indicator
        result = self._compute_indicator(
            combined.xyz,
            combined.get_property(self.property_name),
            target_coords
        )

        # Cache result
        self.reconstruction_age = age
        self._cached_result = result
        self._cached_coords_hash = (target_coords.shape, target_coords[0, 0] if len(target_coords) > 0 else 0)

        return result

    def _load_continental_data(self, data):
        """Load and prepare continental data.

        Args:
            data: One of:
                - PointCloud with property matching self.property_name
                - Path to HDF5/NetCDF file (loaded via h5py)
                - Tuple of (latlon_array, values_array)

        Returns:
            PointCloud filtered to continental regions with plate IDs assigned.
        """
        PointCloud = self._PointCloud

        if hasattr(data, 'xyz') and hasattr(data, 'properties'):
            # Already a PointCloud
            cloud = data
        elif isinstance(data, (str, Path)):
            cloud = self._load_from_hdf5(data)
        elif isinstance(data, tuple) and len(data) == 2:
            latlon, values = data
            cloud = PointCloud.from_latlon(np.asarray(latlon))
            cloud.add_property(self.property_name, np.asarray(values))
        else:
            raise TypeError(
                f"Unsupported continental_data type: {type(data)}. "
                "Expected PointCloud, file path, or (latlon, values) tuple."
            )

        # Filter to continental regions at present day
        log(f"LithosphereConnector: Filtering continental data ({cloud.n_points} points).")
        cloud = self._polygon_filter.filter_inside(cloud, at_age=0.0)
        log(f"LithosphereConnector: After filtering: {cloud.n_points} continental points.")

        # Assign plate IDs for rotation
        cloud = self._rotator.assign_plate_ids(cloud, at_age=0.0, remove_undefined=True)
        log(f"LithosphereConnector: After plate ID assignment: {cloud.n_points} points.")

        return cloud

    def _load_from_hdf5(self, filepath):
        """Load data from HDF5/NetCDF file using h5py.

        Expected file structure:
            - 'lon': longitude array (degrees, 0-360 or -180 to 180)
            - 'lat': latitude array (degrees, -90 to 90)
            - 'z' or property_name: values array

        Args:
            filepath: Path to HDF5 or NetCDF file.

        Returns:
            PointCloud with property values.
        """
        PointCloud = self._PointCloud

        log(f"LithosphereConnector: Loading data from {filepath}.")

        with h5py.File(filepath, 'r') as f:
            lon = f['lon'][:]
            lat = f['lat'][:]

            # Try property_name first, then 'z' as fallback
            if self.property_name in f:
                values = f[self.property_name][:]
            elif 'z' in f:
                values = f['z'][:]
            else:
                raise KeyError(
                    f"File must contain '{self.property_name}' or 'z' dataset. "
                    f"Available datasets: {list(f.keys())}"
                )

        # Convert longitude from 0-360 to -180 to 180 if needed
        lon = np.where(lon > 180, lon - 360, lon)

        # Create meshgrid if 1D arrays (gridded data)
        if lon.ndim == 1 and lat.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            values = values.ravel()
        else:
            # Already 2D or flattened
            latlon = np.column_stack([lat.ravel(), lon.ravel()])
            values = values.ravel()

        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property(self.property_name, values)

        log(f"LithosphereConnector: Loaded {cloud.n_points} points from file.")

        return cloud

    def _compute_indicator(
        self,
        source_xyz: np.ndarray,
        source_thickness_km: np.ndarray,
        target_coords: np.ndarray
    ) -> np.ndarray:
        """Compute smooth lithosphere indicator for target coordinates.

        For each target point:
        1. Compute its radial position r
        2. Look up thickness at its (lon, lat) via inverse-distance interpolation
        3. Compute lithosphere base: lith_base_r = r_outer - thickness_nondim
        4. Return smooth indicator: 0.5 * (1 + tanh((r - lith_base_r) / transition_width))

        Args:
            source_xyz: (N, 3) source point coordinates (unit sphere).
            source_thickness_km: (N,) thickness values in km at source points.
            target_coords: (M, 3) target mesh coordinates.

        Returns:
            (M,) indicator values: ~1 inside lithosphere, ~0 in mantle.
        """
        # Compute radial position of each target point
        r_target = np.linalg.norm(target_coords, axis=1)

        # Normalize target to unit sphere for thickness lookup
        unit_target = target_coords / np.maximum(r_target[:, np.newaxis], 1e-10)

        # Normalize source points to unit sphere
        # (gtrack stores coordinates in meters at Earth's radius ~6.38e6 m)
        r_source = np.linalg.norm(source_xyz, axis=1)
        unit_source = source_xyz / np.maximum(r_source[:, np.newaxis], 1e-10)

        # Build KDTree from source points (now on unit sphere)
        tree = cKDTree(unit_source)
        dists, idx = tree.query(unit_target, k=self.k)

        # Interpolate thickness at each target's (lon, lat)
        epsilon = 1e-10

        if self.k == 1:
            thickness_km = source_thickness_km[idx].copy()
            too_far = dists > self.distance_threshold
            thickness_km[too_far] = self.default_thickness
        else:
            # Handle exact matches
            exact_match = dists[:, 0] < epsilon

            # Identify points with no nearby source data (gaps)
            too_far = dists[:, 0] > self.distance_threshold

            # Safe distances for division
            safe_dists = np.maximum(dists, epsilon)

            # Inverse distance weights
            weights = 1.0 / safe_dists
            weights /= weights.sum(axis=1, keepdims=True)

            # Weighted average thickness
            thickness_km = np.sum(weights * source_thickness_km[idx], axis=1)

            # For exact matches, use nearest value directly
            thickness_km[exact_match] = source_thickness_km[idx[exact_match, 0]]

            # For gaps (no nearby points), use default thickness
            thickness_km[too_far] = self.default_thickness

        # Convert thickness to mesh units
        thickness_nondim = thickness_km / self.depth_scale

        # Compute lithosphere base radius
        lith_base_r = self.r_outer - thickness_nondim

        # Compute smooth indicator using tanh
        # indicator = 0.5 * (1 + tanh((r - lith_base) / width))
        # This gives ~1 when r > lith_base (inside lithosphere)
        # and ~0 when r < lith_base (in mantle)
        indicator = 0.5 * (1.0 + np.tanh(
            (r_target - lith_base_r) / self._transition_width_nondim
        ))

        return indicator


class GplatesScalarFunction(fd.Function):
    """Firedrake Function for lithosphere indicator from plate reconstructions.

    Creates a 3D scalar field that is ~1 inside the lithosphere and ~0 in the
    mantle, with a smooth tanh transition at the lithosphere base. This can be
    used to modify viscosity or other material properties.

    The field is computed by:
    1. Tracking oceanic lithosphere ages forward in time (SeafloorAgeTracker)
    2. Rotating continental thickness data backward in time (PointRotator)
    3. Converting ages to thickness using a user-provided function
    4. Computing indicator based on radial position vs lithosphere base

    Attributes:
        lithosphere_connector: The connector providing indicator data.
        mesh_coords: Cached mesh coordinates for interpolation.

    Args:
        function_space: Scalar function space (e.g., Q).
        lithosphere_connector: LithosphereConnector instance.
        name: Optional name for the function.
        **kwargs: Additional arguments passed to Firedrake Function.

    Examples:
        >>> lithosphere_indicator = GplatesScalarFunction(
        ...     Q,
        ...     lithosphere_connector=lith_connector,
        ...     name="Lithosphere_Indicator"
        ... )
        >>> lithosphere_indicator.update_plate_reconstruction(ndtime=0.5)
        >>>
        >>> # Use to modify viscosity
        >>> effective_viscosity = mantle_viscosity * (1 + 1000 * lithosphere_indicator)
    """

    def __init__(
        self,
        function_space,
        lithosphere_connector: LithosphereConnector,
        name: str = None,
        **kwargs
    ):
        super().__init__(function_space, name=name, **kwargs)
        self.lithosphere_connector = lithosphere_connector

        # Cache mesh coordinates (NOT normalized - connector handles scaling)
        mesh = extract_unique_domain(self)
        # Create a VectorFunctionSpace matching the scalar space for coordinates
        # Use VectorElement to wrap the scalar element (works for TensorProductElements too)
        from finat.ufl import VectorElement
        scalar_element = function_space.ufl_element()
        vector_element = VectorElement(scalar_element)
        coords_space = fd.FunctionSpace(mesh, vector_element)
        coords_func = fd.Function(coords_space)
        coords_func.interpolate(fd.SpatialCoordinate(mesh))
        self.mesh_coords = coords_func.dat.data_ro_with_halos.copy()

    def update_plate_reconstruction(self, ndtime: float):
        """Update lithosphere indicator for given non-dimensional time.

        Delegates to lithosphere_connector.get_indicator() which handles:
        - Time caching (skip if time hasn't changed significantly)
        - Ocean tracker stepping
        - Continental rotation
        - Thickness interpolation
        - Smooth indicator computation

        Args:
            ndtime: Non-dimensional time.
        """
        with stop_annotating():
            values = self.lithosphere_connector.get_indicator(
                self.mesh_coords, ndtime
            )
            self.dat.data_with_halos[:] = values

        if annotate_tape():
            self.create_block_variable()
