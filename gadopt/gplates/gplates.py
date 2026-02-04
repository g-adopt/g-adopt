import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import firedrake as fd
import h5py
import numpy as np
import pygplates
from firedrake.ufl_expr import extract_unique_domain
from pyadjoint.tape import annotate_tape, stop_annotating
from scipy.spatial import cKDTree

from ..solver_options_manager import SolverConfigurationMixin
from ..utility import log, DEBUG, INFO, log_level, InteriorBC, is_continuous
from .connectors import IndicatorConnector


def debug_log(msg):
    """Log message only when GADOPT_LOGLEVEL=DEBUG."""
    if DEBUG >= log_level:
        log(msg)


__all__ = [
    "GplatesVelocityFunction",
    "GplatesScalarFunction",
    "IndicatorConnector",
    "LithosphereConnector",
    "LithosphereConfig",
    "LithosphereConnectorDefault",
    "CratonConnector",
    "CratonConfig",
    "CratonConnectorDefault",
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
                                 out=np.ones_like(original_magnitudes),
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


@dataclass
class LithosphereConfig:
    """Configuration for lithosphere indicator computation.

    Groups all tunable parameters for LithosphereConnector into a single
    configuration object. Provides sensible defaults for Earth's mantle.

    Use with LithosphereConnector:
        >>> config = LithosphereConfig(n_points=40000, transition_width=5.0)
        >>> connector = LithosphereConnector(gplates, data, age_func, config=config)

    Or override specific values with config_extra:
        >>> connector = LithosphereConnector(
        ...     gplates, data, age_func,
        ...     config_extra={"n_points": 40000}
        ... )

    Attributes
    ----------
    Ocean Tracker Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    time_step : float
        Time step for ocean age tracker in Myr. Default: 1.0
    n_points : int
        Number of points for ocean tracker mesh. Higher values give
        better resolution but slower computation. Default: 10000
    reinit_interval_myr : float
        Reinitialize ocean tracker every N Myr to prevent drift
        accumulation. Default: 50.0

    Interpolation Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    k_neighbors : int
        Number of nearest neighbors for thickness interpolation.
        Default: 50
    distance_threshold : float
        Maximum angular distance (radians on unit sphere) for valid
        interpolation. Points beyond this receive default_thickness.
        Default: 0.1 (~570 km at Earth's surface)
    default_thickness : float
        Thickness (km) assigned to points with no nearby data.
        Default: 100.0

    Mesh Geometry Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    r_outer : float
        Outer radius of mesh in non-dimensional units. This is the
        radial coordinate of Earth's surface in the mesh.
        Default: 2.208 (for r_inner=1.208, giving mantle depth ratio)
    depth_scale : float
        Physical depth (km) corresponding to 1 non-dimensional unit.
        For Earth's mantle: 2890 km. Default: 2890.0
    transition_width : float
        Width of tanh transition at lithosphere base in km. Controls
        smoothness of the indicator field. Default: 10.0

    Data Parameters
    ~~~~~~~~~~~~~~~
    property_name : str
        Name of the thickness property in data files. Default: 'thickness'

    Examples
    --------
    >>> # Use all defaults
    >>> config = LithosphereConfig()
    >>>
    >>> # High-resolution ocean tracking
    >>> config = LithosphereConfig(
    ...     n_points=40000,
    ...     time_step=0.5,
    ...     reinit_interval_myr=25.0,
    ... )
    >>>
    >>> # Different mesh geometry (e.g., different r_inner)
    >>> config = LithosphereConfig(
    ...     r_outer=2.5,
    ...     depth_scale=2890.0,
    ... )
    """

    # Ocean tracker parameters
    time_step: float = 1.0
    n_points: int = 10000
    reinit_interval_myr: float = 50.0

    # Interpolation parameters
    k_neighbors: int = 50
    distance_threshold: float = 0.1
    default_thickness: float = 100.0

    # Mesh geometry parameters
    r_outer: float = 2.208
    depth_scale: float = 2890.0
    transition_width: float = 10.0

    # Data parameters
    property_name: str = "thickness"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.n_points < 100:
            raise ValueError(f"n_points must be at least 100, got {self.n_points}")
        if self.reinit_interval_myr <= 0:
            raise ValueError(
                f"reinit_interval_myr must be positive, got {self.reinit_interval_myr}"
            )
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {self.k_neighbors}")
        if self.distance_threshold <= 0:
            raise ValueError(
                f"distance_threshold must be positive, got {self.distance_threshold}"
            )
        if self.default_thickness < 0:
            raise ValueError(
                f"default_thickness must be non-negative, got {self.default_thickness}"
            )
        if self.r_outer <= 0:
            raise ValueError(f"r_outer must be positive, got {self.r_outer}")
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")
        if self.transition_width <= 0:
            raise ValueError(
                f"transition_width must be positive, got {self.transition_width}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary, suitable for serialization.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LithosphereConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters. Unknown keys are ignored.

        Returns
        -------
        LithosphereConfig
            Configuration object with values from dictionary.
        """
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)

    def with_overrides(self, overrides: dict) -> "LithosphereConfig":
        """Create a new config with specified overrides.

        Parameters
        ----------
        overrides : dict
            Dictionary of parameter overrides.

        Returns
        -------
        LithosphereConfig
            New configuration with overrides applied.

        Examples
        --------
        >>> base = LithosphereConfig()
        >>> high_res = base.with_overrides({"n_points": 40000})
        """
        current = self.to_dict()
        current.update(overrides)
        return self.from_dict(current)


# Default configuration instance for LithosphereConnector.
# Import and inspect this to see all available parameters and their defaults:
#   from gadopt.gplates import LithosphereConnectorDefault
#   print(LithosphereConnectorDefault)
LithosphereConnectorDefault = LithosphereConfig()


class LithosphereConnector(IndicatorConnector):
    """Connector for lithosphere indicator field through geological time.

    Combines oceanic lithosphere (age-tracked via gtrack's SeafloorAgeTracker,
    converted to thickness) with continental lithosphere (back-rotated
    present-day thickness data via gtrack's PointRotator) to produce a smooth
    3D indicator field: 1 inside lithosphere, 0 in mantle.

    The indicator uses a tanh transition for numerical stability:
        indicator = 0.5 * (1 + tanh((r - lith_base) / transition_width))

    where lith_base = r_outer - thickness (in mesh units).

    Uses pyGplatesConnector for plate model access and time conversion.

    MPI Parallelization:
        When a communicator is provided, only rank 0 performs I/O and gtrack
        computations. The resulting thickness data is broadcast to all ranks,
        and each rank interpolates to its local mesh points. This avoids
        redundant I/O and computation across MPI ranks.

    Attributes:
        gplates_connector: The pyGplatesConnector instance for time conversion and plate model.
        config: The LithosphereConfig with all tunable parameters.
        reconstruction_age: Last computed geological age (Ma), used for caching.
        comm: MPI communicator (None for serial execution).

    Args:
        gplates_connector (pyGplatesConnector): Connector with plate model files.
            Must have `continental_polygons` and `static_polygons` set.
        continental_data: Present-day continental thickness data. Can be:
            - gtrack.PointCloud with the thickness property
            - Path to HDF5/NetCDF file
            - Tuple of (latlon_array, thickness_values_km)
        age_to_property (Callable): Function converting seafloor age (Myr) to thickness (km).
            Signature: age_to_property(ages: np.ndarray) -> np.ndarray
        config (LithosphereConfig, optional): Configuration object with all tunable
            parameters. If None, uses default LithosphereConfig().
        config_extra (dict, optional): Dictionary of parameter overrides to apply
            on top of config. Useful for tweaking specific values without creating
            a full config object.
        comm: MPI communicator for parallel execution. If provided, rank 0
            performs I/O and gtrack computations, then broadcasts results.
            If None (default), all operations run locally (serial mode).

    Examples:
        >>> def half_space_cooling(age_myr):
        ...     age_sec = age_myr * 3.15576e13
        ...     return 2.32 * np.sqrt(1e-6 * age_sec) / 1e3  # km
        >>>
        >>> # Using default configuration (serial)
        >>> connector = LithosphereConnector(
        ...     gplates_connector=plate_model,
        ...     continental_data='SL2013sv.nc',
        ...     age_to_property=half_space_cooling,
        ... )
        >>>
        >>> # With MPI parallelization (rank 0 computes, broadcasts to others)
        >>> connector = LithosphereConnector(
        ...     gplates_connector=plate_model,
        ...     continental_data='SL2013sv.nc',
        ...     age_to_property=half_space_cooling,
        ...     comm=mesh.comm,
        ... )
        >>>
        >>> # With custom config
        >>> config = LithosphereConfig(n_points=40000, transition_width=5.0)
        >>> connector = LithosphereConnector(
        ...     gplates_connector=plate_model,
        ...     continental_data='SL2013sv.nc',
        ...     age_to_property=half_space_cooling,
        ...     config=config,
        ...     comm=mesh.comm,
        ... )
        >>>
        >>> indicator = connector.get_indicator(mesh_coords, ndtime)
    """

    def __init__(
        self,
        gplates_connector: "pyGplatesConnector",
        continental_data,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        config: LithosphereConfig = None,
        config_extra: dict = None,
        comm=None,
    ):
        self.gplates_connector = gplates_connector
        self.age_to_property = age_to_property
        self.comm = comm
        self._is_root = (comm is None or comm.rank == 0)

        # Build effective configuration
        # Use module-level LithosphereConnectorDefault if no config provided
        if config is None:
            config = LithosphereConnectorDefault
        if config_extra is not None:
            config = config.with_overrides(config_extra)
        self.config = config

        # Pre-compute transition width in mesh units
        self._transition_width_nondim = config.transition_width / config.depth_scale

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

        # Import gtrack components (import on all ranks is fine, just don't instantiate)
        from gtrack import (
            SeafloorAgeTracker, PointRotator, PolygonFilter,
            PointCloud, TracerConfig
        )

        # Store PointCloud class for later use
        self._PointCloud = PointCloud

        # Only rank 0 creates gtrack objects and loads data
        # This avoids redundant I/O and memory usage across MPI ranks
        if self._is_root:
            # Create ocean age tracker
            tracer_config = TracerConfig(
                time_step=config.time_step,
                default_mesh_points=config.n_points,
            )
            self._ocean_tracker = SeafloorAgeTracker(
                rotation_files=gplates_connector.rotation_filenames,
                topology_files=gplates_connector.topology_filenames,
                continental_polygons=gplates_connector.continental_polygons,
                config=tracer_config,
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
        else:
            # Non-root ranks don't need gtrack objects
            self._ocean_tracker = None
            self._rotator = None
            self._polygon_filter = None
            self._continental_present = None

        # State management
        self._initialized = False
        self._last_reinit_age = None
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

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

        When running with MPI (comm is set), rank 0 performs all gtrack computations
        and broadcasts the resulting thickness data. All ranks then interpolate
        to their local mesh points.

        Args:
            target_coords: (M, 3) array of mesh coordinates in mesh units.
                Each MPI rank provides its local mesh coordinates.
            ndtime: Non-dimensional time.

        Returns:
            (M,) array of indicator values (0 to 1) for local mesh points.
        """
        age = self.ndtime2age(ndtime)

        debug_log(f"LithosphereConnector: Computing lithosphere indicator for {age:.2f} Ma.")

        # Validate requested age
        oldest_age = self.gplates_connector.oldest_age
        if age > oldest_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's oldest age "
                f"({oldest_age:.2f} Ma)."
            )
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future). "
                f"Ages must be >= 0 (present day)."
            )

        # Check if ocean tracker would need to go backward (invalid for SeafloorAgeTracker)
        # This is specific to LithosphereConnector because it tracks ocean floor ages forward in time
        if self._initialized and self._is_root:
            current_tracker_age = self._ocean_tracker.current_age
            if current_tracker_age is not None and age > current_tracker_age:
                raise ValueError(
                    f"Requested age {age:.2f} Ma is older than the ocean tracker's current "
                    f"position ({current_tracker_age:.2f} Ma). The ocean tracker can only "
                    f"evolve forward in time (decreasing age). Ages must be requested in "
                    f"decreasing order (e.g., 200, 150, 100, 50, 0 Ma)."
                )

        # Check if we can use cached result (same age within tolerance)
        if self.reconstruction_age is not None:
            if abs(age - self.reconstruction_age) < self.gplates_connector.delta_t:
                # Check if coordinates are the same (using hash of shape and sample)
                coords_hash = (target_coords.shape, target_coords[0, 0] if len(target_coords) > 0 else 0)
                if self._cached_result is not None and coords_hash == self._cached_coords_hash:
                    log(f"LithosphereConnector: Age {age:.2f} Ma unchanged (within delta_t={self.gplates_connector.delta_t}), using cached result.")
                    return self._cached_result

        # Rank 0 performs all gtrack operations, others wait for broadcast
        if self._is_root:
            # Initialize ocean tracker if needed
            if not self._initialized:
                starting_age = int(self.gplates_connector.oldest_age)
                debug_log(f"LithosphereConnector: Initializing ocean tracker at {starting_age} Ma.")
                self._ocean_tracker.initialize(starting_age=starting_age)
                self._initialized = True
                self._last_reinit_age = starting_age

            # Check for reinitialisation
            if self._last_reinit_age is not None:
                if abs(self._last_reinit_age - age) >= self.config.reinit_interval_myr:
                    debug_log(f"LithosphereConnector: Reinitializing ocean tracker at {age:.2f} Ma.")
                    self._ocean_tracker.reinitialize(n_points=self.config.n_points)
                    self._last_reinit_age = age

            # Get ocean lithosphere
            # Note: gtrack has a bug where step_to fails if int(current_age) == int(target_age)
            # due to float accumulation. We work around this by using integer ages.
            ocean_cloud = self._ocean_tracker.step_to(int(round(age)))
            ocean_ages = ocean_cloud.get_property('age')
            ocean_values = self.age_to_property(ocean_ages)
            ocean_cloud.add_property(self.config.property_name, ocean_values)

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

            # Extract numpy arrays for broadcast
            source_xyz = combined.xyz.copy()
            source_thickness = combined.get_property(self.config.property_name).copy()
        else:
            # Non-root ranks: placeholders for broadcast
            source_xyz = None
            source_thickness = None

        # Broadcast thickness data from rank 0 to all ranks
        if self.comm is not None:
            source_xyz = self.comm.bcast(source_xyz, root=0)
            source_thickness = self.comm.bcast(source_thickness, root=0)

        # All ranks compute indicator for their local mesh points
        result = self._compute_indicator(
            source_xyz,
            source_thickness,
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
            cloud.add_property(self.config.property_name, np.asarray(values))
        else:
            raise TypeError(
                f"Unsupported continental_data type: {type(data)}. "
                "Expected PointCloud, file path, or (latlon, values) tuple."
            )

        # Filter to continental regions at present day
        debug_log(f"LithosphereConnector: Filtering continental data ({cloud.n_points} points).")
        cloud = self._polygon_filter.filter_inside(cloud, at_age=0.0)
        debug_log(f"LithosphereConnector: After filtering: {cloud.n_points} continental points.")

        # Assign plate IDs for rotation
        cloud = self._rotator.assign_plate_ids(cloud, at_age=0.0, remove_undefined=True)
        debug_log(f"LithosphereConnector: After plate ID assignment: {cloud.n_points} points.")

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

        debug_log(f"LithosphereConnector: Loading data from {filepath}.")

        with h5py.File(filepath, 'r') as f:
            lon = f['lon'][:]
            lat = f['lat'][:]

            # Try property_name first, then 'z' as fallback
            if self.config.property_name in f:
                values = f[self.config.property_name][:]
            elif 'z' in f:
                values = f['z'][:]
            else:
                raise KeyError(
                    f"File must contain '{self.config.property_name}' or 'z' dataset. "
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
        cloud.add_property(self.config.property_name, values)

        debug_log(f"LithosphereConnector: Loaded {cloud.n_points} points from file.")

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
        dists, idx = tree.query(unit_target, k=self.config.k_neighbors)

        # Interpolate thickness at each target's (lon, lat)
        epsilon = 1e-10

        if self.config.k_neighbors == 1:
            thickness_km = source_thickness_km[idx].copy()
            too_far = dists > self.config.distance_threshold
            thickness_km[too_far] = self.config.default_thickness
        else:
            # Handle exact matches
            exact_match = dists[:, 0] < epsilon

            # Identify points with no nearby source data (gaps)
            too_far = dists[:, 0] > self.config.distance_threshold

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
            thickness_km[too_far] = self.config.default_thickness

        # Convert thickness to mesh units
        thickness_nondim = thickness_km / self.config.depth_scale

        # Compute lithosphere base radius
        lith_base_r = self.config.r_outer - thickness_nondim

        # Compute smooth indicator using tanh
        # indicator = 0.5 * (1 + tanh((r - lith_base) / width))
        # This gives ~1 when r > lith_base (inside lithosphere)
        # and ~0 when r < lith_base (in mantle)
        indicator = 0.5 * (1.0 + np.tanh(
            (r_target - lith_base_r) / self._transition_width_nondim
        ))

        return indicator


@dataclass
class CratonConfig:
    """Configuration for craton indicator computation.

    Groups all tunable parameters for CratonConnector into a single
    configuration object. Similar to LithosphereConfig but without
    ocean age tracking parameters.

    Use with CratonConnector:
        >>> config = CratonConfig(n_points=40000, transition_width=5.0)
        >>> connector = CratonConnector(gplates, polygons, thickness_data, config=config)

    Or override specific values with config_extra:
        >>> connector = CratonConnector(
        ...     gplates, polygons, thickness_data,
        ...     config_extra={"n_points": 40000}
        ... )

    Attributes
    ----------
    Sampling Parameters
    ~~~~~~~~~~~~~~~~~~~
    n_points : int
        Number of sample points on fibonacci sphere for polygon coverage.
        Higher values give better resolution of craton boundaries.
        Default: 20000

    Interpolation Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    k_neighbors : int
        Number of nearest neighbors for thickness interpolation.
        Default: 50
    distance_threshold : float
        Maximum angular distance (radians on unit sphere) for valid
        interpolation. Points beyond this receive default_thickness.
        Default: 0.1 (~570 km at Earth's surface)
    default_thickness : float
        Thickness (km) assigned to points with no nearby data.
        Default: 200.0 (typical cratonic root depth)

    Mesh Geometry Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    r_outer : float
        Outer radius of mesh in non-dimensional units.
        Default: 2.208 (for r_inner=1.208)
    depth_scale : float
        Physical depth (km) corresponding to 1 non-dimensional unit.
        For Earth's mantle: 2890 km. Default: 2890.0
    transition_width : float
        Width of tanh transition at craton base in km. Controls
        smoothness of the indicator field. Default: 10.0

    Data Parameters
    ~~~~~~~~~~~~~~~
    property_name : str
        Name of the thickness property in data files. Default: 'thickness'

    Examples
    --------
    >>> # Use all defaults
    >>> config = CratonConfig()
    >>>
    >>> # Higher resolution with sharper boundaries
    >>> config = CratonConfig(
    ...     n_points=50000,
    ...     transition_width=5.0,
    ... )
    """

    # Sampling parameters
    n_points: int = 20000

    # Interpolation parameters
    k_neighbors: int = 50
    distance_threshold: float = 0.1
    default_thickness: float = 200.0

    # Mesh geometry parameters
    r_outer: float = 2.208
    depth_scale: float = 2890.0
    transition_width: float = 10.0

    # Data parameters
    property_name: str = "thickness"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_points < 100:
            raise ValueError(f"n_points must be at least 100, got {self.n_points}")
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be at least 1, got {self.k_neighbors}")
        if self.distance_threshold <= 0:
            raise ValueError(
                f"distance_threshold must be positive, got {self.distance_threshold}"
            )
        if self.default_thickness < 0:
            raise ValueError(
                f"default_thickness must be non-negative, got {self.default_thickness}"
            )
        if self.r_outer <= 0:
            raise ValueError(f"r_outer must be positive, got {self.r_outer}")
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")
        if self.transition_width <= 0:
            raise ValueError(
                f"transition_width must be positive, got {self.transition_width}"
            )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CratonConfig":
        """Create configuration from dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)

    def with_overrides(self, overrides: dict) -> "CratonConfig":
        """Create a new config with specified overrides."""
        current = self.to_dict()
        current.update(overrides)
        return self.from_dict(current)


# Default configuration instance for CratonConnector.
CratonConnectorDefault = CratonConfig()


class CratonConnector(IndicatorConnector):
    """Connector for craton indicator field through geological time.

    Computes a smooth 3D indicator field for cratonic lithosphere by combining:
    - Craton polygons (back-rotated to target age via gtrack's PolygonFilter)
    - Cratonic thickness data (back-rotated via gtrack's PointRotator)

    The indicator is ~1 inside cratonic lithosphere, ~0 in mantle, with a
    smooth tanh transition at the craton base. This is similar to
    LithosphereConnector but specific to cratonic regions.

    Uses pyGplatesConnector for plate model access and time conversion.

    MPI Parallelization:
        When a communicator is provided, only rank 0 performs I/O and gtrack
        computations. The resulting thickness data is broadcast to all ranks,
        and each rank interpolates to its local mesh points.

    Attributes:
        gplates_connector: The pyGplatesConnector for time conversion and plate model.
        config: The CratonConfig with all tunable parameters.
        reconstruction_age: Last computed geological age (Ma), used for caching.
        comm: MPI communicator (None for serial execution).

    Args:
        gplates_connector (pyGplatesConnector): Connector with plate model files.
            Must have `static_polygons` set for rotation.
        craton_polygons: Path to craton polygon shapefile (e.g., shapes_cratons.shp).
        craton_thickness_data: Present-day cratonic thickness data. Can be:
            - gtrack.PointCloud with the thickness property
            - Path to HDF5/NetCDF file
            - Tuple of (latlon_array, thickness_values_km)
        config (CratonConfig, optional): Configuration object. If None, uses defaults.
        config_extra (dict, optional): Dictionary of parameter overrides.
        comm: MPI communicator for parallel execution.

    Examples:
        >>> # Using default configuration
        >>> connector = CratonConnector(
        ...     gplates_connector=plate_model,
        ...     craton_polygons='shapes_cratons.shp',
        ...     craton_thickness_data='craton_thickness.h5',
        ... )
        >>>
        >>> # With data as tuple
        >>> connector = CratonConnector(
        ...     gplates_connector=plate_model,
        ...     craton_polygons='shapes_cratons.shp',
        ...     craton_thickness_data=(latlon_array, thickness_km),
        ...     comm=mesh.comm,
        ... )
        >>>
        >>> indicator = connector.get_indicator(mesh_coords, ndtime)
    """

    def __init__(
        self,
        gplates_connector: "pyGplatesConnector",
        craton_polygons,
        craton_thickness_data,
        config: CratonConfig = None,
        config_extra: dict = None,
        comm=None,
    ):
        self.gplates_connector = gplates_connector
        self.comm = comm
        self._is_root = (comm is None or comm.rank == 0)

        # Build effective configuration
        if config is None:
            config = CratonConnectorDefault
        if config_extra is not None:
            config = config.with_overrides(config_extra)
        self.config = config

        # Pre-compute transition width in mesh units
        self._transition_width_nondim = config.transition_width / config.depth_scale

        # Validate connector has required polygon files
        if gplates_connector.static_polygons is None:
            raise ValueError(
                "gplates_connector must have static_polygons set. "
                "Pass static_polygons to pyGplatesConnector constructor."
            )

        # Import gtrack components
        from gtrack import PolygonFilter, PointRotator, PointCloud

        self._PointCloud = PointCloud

        # Only rank 0 creates gtrack objects and loads data
        if self._is_root:
            # Create polygon filter for craton regions
            self._craton_filter = PolygonFilter(
                polygon_files=craton_polygons,
                rotation_files=gplates_connector.rotation_filenames,
            )

            # Create point rotator for thickness data
            self._rotator = PointRotator(
                rotation_files=gplates_connector.rotation_filenames,
                static_polygons=gplates_connector.static_polygons,
            )

            # Load and prepare craton thickness data
            self._craton_present = self._load_craton_data(craton_thickness_data, craton_polygons)
        else:
            self._craton_filter = None
            self._rotator = None
            self._craton_present = None

        # State management
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

    def _load_craton_data(self, data, craton_polygons):
        """Load and prepare craton thickness data.

        Args:
            data: One of:
                - PointCloud with property matching self.config.property_name
                - Path to HDF5/NetCDF file
                - Tuple of (latlon_array, values_array)
            craton_polygons: Path to craton polygon file for filtering.

        Returns:
            PointCloud filtered to craton regions with plate IDs assigned.
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
            cloud.add_property(self.config.property_name, np.asarray(values))
        else:
            raise TypeError(
                f"Unsupported craton_thickness_data type: {type(data)}. "
                "Expected PointCloud, file path, or (latlon, values) tuple."
            )

        # Filter to craton regions at present day
        n_before = cloud.n_points
        log(f"CratonConnector: Filtering {n_before} continental points to craton polygons...")
        cloud = self._craton_filter.filter_inside(cloud, at_age=0.0)
        log(f"CratonConnector: After craton filtering: {cloud.n_points} points ({100*cloud.n_points/n_before:.1f}% retained)")

        # Assign plate IDs for rotation
        cloud = self._rotator.assign_plate_ids(cloud, at_age=0.0, remove_undefined=True)
        debug_log(f"CratonConnector: After plate ID assignment: {cloud.n_points} points.")

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

        debug_log(f"CratonConnector: Loading data from {filepath}.")

        with h5py.File(filepath, 'r') as f:
            lon = f['lon'][:]
            lat = f['lat'][:]

            # Try property_name first, then 'z' as fallback
            if self.config.property_name in f:
                values = f[self.config.property_name][:]
            elif 'z' in f:
                values = f['z'][:]
            else:
                raise KeyError(
                    f"File must contain '{self.config.property_name}' or 'z' dataset. "
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
        cloud.add_property(self.config.property_name, values)

        debug_log(f"CratonConnector: Loaded {cloud.n_points} points from file.")

        return cloud

    def get_indicator(
        self,
        target_coords: np.ndarray,
        ndtime: float
    ) -> np.ndarray:
        """Get smooth craton indicator at target coordinates for given time.

        Returns a 3D field that is ~1 inside cratonic lithosphere and ~0 outside,
        with a smooth tanh transition at the craton base.

        For each target point:
        1. Computes its radial position r
        2. Looks up thickness at its (lon, lat) from back-rotated craton data
        3. Computes craton base: craton_base_r = r_outer - thickness_nondim
        4. Returns smooth indicator: 0.5 * (1 + tanh((r - craton_base_r) / width))

        When running with MPI, rank 0 performs gtrack computations and
        broadcasts results. All ranks then interpolate to local mesh points.

        Args:
            target_coords: (M, 3) array of mesh coordinates.
            ndtime: Non-dimensional time.

        Returns:
            (M,) array of indicator values (0 to 1).
        """
        age = self.ndtime2age(ndtime)

        debug_log(f"CratonConnector: Computing craton indicator for {age:.2f} Ma.")

        # Validate requested age
        # For CratonConnector, rotation works for any valid age (no time-stepping constraint)
        oldest_age = self.gplates_connector.oldest_age
        if age > oldest_age:
            raise ValueError(
                f"Requested age {age:.2f} Ma is older than the plate model's oldest age "
                f"({oldest_age:.2f} Ma)."
            )
        if age < 0:
            raise ValueError(
                f"Requested age {age:.2f} Ma is negative (in the future). "
                f"Ages must be >= 0 (present day)."
            )

        # Check if we can use cached result (same age within tolerance)
        if self.reconstruction_age is not None:
            if abs(age - self.reconstruction_age) < self.gplates_connector.delta_t:
                coords_hash = (target_coords.shape, target_coords[0, 0] if len(target_coords) > 0 else 0)
                if self._cached_result is not None and coords_hash == self._cached_coords_hash:
                    log(f"CratonConnector: Age {age:.2f} Ma unchanged (within delta_t={self.gplates_connector.delta_t}), using cached result.")
                    return self._cached_result

        # Rank 0 performs gtrack operations
        if self._is_root:
            # Rotate craton thickness data to target age
            craton_cloud = self._rotator.rotate(
                self._craton_present,
                from_age=0.0,
                to_age=age
            )

            debug_log(f"CratonConnector: {craton_cloud.n_points} craton points at {age:.2f} Ma.")

            # Extract numpy arrays for broadcast
            source_xyz = craton_cloud.xyz.copy()
            source_thickness = craton_cloud.get_property(self.config.property_name).copy()
        else:
            source_xyz = None
            source_thickness = None

        # Broadcast thickness data from rank 0 to all ranks
        if self.comm is not None:
            source_xyz = self.comm.bcast(source_xyz, root=0)
            source_thickness = self.comm.bcast(source_thickness, root=0)

        # All ranks compute indicator for their local mesh points
        result = self._compute_indicator(source_xyz, source_thickness, target_coords)

        # Cache result
        self.reconstruction_age = age
        self._cached_result = result
        self._cached_coords_hash = (target_coords.shape, target_coords[0, 0] if len(target_coords) > 0 else 0)

        return result

    def _compute_indicator(
        self,
        source_xyz: np.ndarray,
        source_thickness_km: np.ndarray,
        target_coords: np.ndarray
    ) -> np.ndarray:
        """Compute smooth craton indicator for target coordinates.

        For each target point:
        1. Compute its radial position r
        2. Look up thickness at its (lon, lat) via inverse-distance interpolation
        3. Compute craton base: craton_base_r = r_outer - thickness_nondim
        4. Return smooth indicator: 0.5 * (1 + tanh((r - craton_base_r) / width))

        Points far from any craton data get indicator = 0 (not in craton).

        Args:
            source_xyz: (N, 3) source point coordinates (from gtrack, in meters).
            source_thickness_km: (N,) thickness values in km at source points.
            target_coords: (M, 3) target mesh coordinates.

        Returns:
            (M,) indicator values: ~1 inside craton, ~0 outside.
        """
        # Handle empty craton case
        if source_xyz is None or len(source_xyz) == 0:
            return np.zeros(len(target_coords))

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

        # Limit k to available source points (important for small craton datasets)
        k = min(self.config.k_neighbors, len(source_xyz))
        dists, idx = tree.query(unit_target, k=k)

        # Interpolate thickness at each target's (lon, lat)
        epsilon = 1e-10

        if k == 1:
            thickness_km = source_thickness_km[idx].copy()
            too_far = dists > self.config.distance_threshold
        else:
            # Handle exact matches
            exact_match = dists[:, 0] < epsilon

            # Identify points with no nearby source data (outside cratons)
            too_far = dists[:, 0] > self.config.distance_threshold

            # Safe distances for division
            safe_dists = np.maximum(dists, epsilon)

            # Inverse distance weights
            weights = 1.0 / safe_dists
            weights /= weights.sum(axis=1, keepdims=True)

            # Weighted average thickness
            thickness_km = np.sum(weights * source_thickness_km[idx], axis=1)

            # For exact matches, use nearest value directly
            thickness_km[exact_match] = source_thickness_km[idx[exact_match, 0]]

        # Convert thickness to mesh units
        thickness_nondim = thickness_km / self.config.depth_scale

        # Compute craton base radius
        craton_base_r = self.config.r_outer - thickness_nondim

        # Compute smooth indicator using tanh
        # indicator = 0.5 * (1 + tanh((r - craton_base) / width))
        # This gives ~1 when r > craton_base (inside craton)
        # and ~0 when r < craton_base (below craton)
        indicator = 0.5 * (1.0 + np.tanh(
            (r_target - craton_base_r) / self._transition_width_nondim
        ))

        # Points far from any craton data are outside cratons -> indicator = 0
        indicator[too_far] = 0.0

        return indicator


class GplatesScalarFunction(fd.Function):
    """Firedrake Function for scalar indicator fields from plate reconstructions.

    Creates a 3D scalar field that is ~1 in regions of interest and ~0 elsewhere,
    with smooth transitions at boundaries. This can be used to modify viscosity
    or other material properties based on plate tectonic reconstructions.

    Works with any IndicatorConnector subclass:
    - LithosphereConnector: Lithosphere indicator (ocean ages + continental thickness)
    - CratonConnector: Craton indicator (polygon-based)

    MPI Parallelization:
        For efficient parallel execution, pass `comm=mesh.comm` when creating
        the connector. This ensures rank 0 performs I/O and gtrack computations,
        then broadcasts results to all ranks.

    Attributes:
        indicator_connector: The IndicatorConnector providing indicator data.
        mesh_coords: Cached mesh coordinates for interpolation.

    Args:
        function_space: Scalar function space (e.g., Q).
        indicator_connector: Any IndicatorConnector subclass (LithosphereConnector,
            CratonConnector, etc.). For parallel execution, create with `comm=mesh.comm`.
        name: Optional name for the function.
        **kwargs: Additional arguments passed to Firedrake Function.

    Examples:
        >>> # Lithosphere indicator
        >>> lith_connector = LithosphereConnector(..., comm=mesh.comm)
        >>> lithosphere = GplatesScalarFunction(
        ...     Q,
        ...     indicator_connector=lith_connector,
        ...     name="Lithosphere"
        ... )
        >>>
        >>> # Craton indicator
        >>> craton_connector = CratonConnector(..., comm=mesh.comm)
        >>> cratons = GplatesScalarFunction(
        ...     Q,
        ...     indicator_connector=craton_connector,
        ...     name="Cratons"
        ... )
        >>>
        >>> # Update and use
        >>> lithosphere.update_plate_reconstruction(ndtime=0.5)
        >>> cratons.update_plate_reconstruction(ndtime=0.5)
        >>> effective_viscosity = base_viscosity * (1 + 1000 * lithosphere + 100 * cratons)
    """

    def __init__(
        self,
        function_space,
        indicator_connector: IndicatorConnector,
        name: str = None,
        **kwargs
    ):
        super().__init__(function_space, name=name, **kwargs)

        # Validate connector type
        if not isinstance(indicator_connector, IndicatorConnector):
            raise TypeError(
                f"indicator_connector must be an IndicatorConnector subclass, "
                f"got {type(indicator_connector).__name__}"
            )

        self.indicator_connector = indicator_connector

        # Cache mesh coordinates (NOT normalized - connector handles scaling)
        mesh = extract_unique_domain(self)

        # Warn if running in parallel without comm set on connector
        if mesh.comm.size > 1 and indicator_connector.comm is None:
            warnings.warn(
                f"Running in parallel but {type(indicator_connector).__name__} has no communicator. "
                "Each MPI rank will independently load data and compute gtrack operations. "
                "For efficiency, create connector with comm=mesh.comm to have rank 0 "
                "compute and broadcast results.",
                category=RuntimeWarning
            )

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
        """Update indicator field for given non-dimensional time.

        Delegates to indicator_connector.get_indicator() which handles:
        - Time caching (skip if time hasn't changed significantly)
        - Data loading and gtrack operations
        - Interpolation to mesh coordinates
        - Smooth indicator computation

        Args:
            ndtime: Non-dimensional time.
        """
        with stop_annotating():
            values = self.indicator_connector.get_indicator(
                self.mesh_coords, ndtime
            )
            self.dat.data_with_halos[:] = values

        if annotate_tape():
            self.create_block_variable()
