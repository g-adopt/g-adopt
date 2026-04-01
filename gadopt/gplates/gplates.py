import os
import re
import warnings
from dataclasses import dataclass
from typing import Callable

import firedrake as fd
import numpy as np
import pygplates
from firedrake import VectorElement
from mpi4py import MPI
from firedrake.ufl_expr import extract_unique_domain
from scipy.special import erf
from gtrack import (
    SeafloorAgeTracker, PointRotator, PolygonFilter,
    PointCloud, TracerConfig,
)
from pyadjoint.tape import annotate_tape, stop_annotating
from scipy.spatial import cKDTree

from ..solver_options_manager import SolverConfigurationMixin
from ..utility import log, DEBUG, INFO, log_level, InteriorBC, is_continuous
from .connectors import InterpolationConfig, IndicatorConfigBase, IndicatorConnector
from .gplatesfiles import ensure_reconstruction


__all__ = [
    "ensure_reconstruction",
    "GplatesVelocityFunction",
    "GplatesScalarFunction",
    "InterpolationConfig",
    "IndicatorConfigBase",
    "IndicatorConnector",
    "LithosphereConnector",
    "LithosphereConfig",
    "LithosphereGeotherm",
    "PolygonConnector",
    "PolygonConfig",
    "PolygonGeotherm",
    "ocean_erf_normalized",
    "continental_linear",
    "pyGplatesConnector",
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

        # When gplates_connector is None (e.g. during adjoint perturbations via
        # _ad_mul/_ad_add, or when Firedrake internally creates subfunctions),
        # skip all the plate reconstruction setup.
        if gplates_connector is None:
            return

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

    def _ad_mul(self, other):
        r = GplatesVelocityFunction(self.function_space())
        r.assign(other * self)
        return r

    def _ad_add(self, other):
        r = GplatesVelocityFunction(self.function_space())
        fd.Function.assign(r, self + other)
        return r


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
class LithosphereConfig(IndicatorConfigBase):
    """Configuration for lithosphere indicator computation.

    Groups all tunable parameters for LithosphereConnector into a single
    configuration object. Provides sensible defaults for Earth's mantle.

    Use with LithosphereConnector:
        >>> from gadopt.gplates import InterpolationConfig
        >>> config = LithosphereConfig(
        ...     n_points=40000,
        ...     interpolation=InterpolationConfig(kernel="gaussian", k_neighbors=200),
        ... )
        >>> connector = LithosphereConnector(gplates, data, age_func, config=config)

    Or override specific values with config_extra (flat keys are routed
    to InterpolationConfig automatically):
        >>> connector = LithosphereConnector(
        ...     gplates, data, age_func,
        ...     config_extra={"k_neighbors": 20, "kernel": "gaussian"}
        ... )

    Args:
        time_step: Time step for ocean age tracker in Myr. Default: 1.0.
        n_points: Number of points for ocean tracker mesh. Higher values
            give better resolution but slower computation. Default: 10000.
        reinit_interval_myr: Reinitialize ocean tracker every N Myr to
            prevent drift accumulation. Default: 50.0.
        interpolation: Interpolation parameters (kernel, k_neighbors,
            distance_threshold, etc.). See :class:`InterpolationConfig`.
            Default: ``InterpolationConfig(default_value=100.0)``.
        r_outer: Outer radius of mesh in non-dimensional units. This is
            the radial coordinate of Earth's surface in the mesh.
            Default: 2.208 (for r_inner=1.208, giving mantle depth ratio).
        depth_scale: Physical depth (km) corresponding to 1 non-dimensional
            unit. For Earth's mantle: 2890 km. Default: 2890.0.
        transition_width: Width of tanh transition at lithosphere base in
            km. Controls smoothness of the indicator field. Default: 10.0.
        property_name: Name of the thickness property in data files.
            Default: 'thickness'.
        gtrack_config: Optional dictionary of additional overrides
            for gtrack's TracerConfig. Any parameter accepted by
            TracerConfig can be passed here (e.g. ridge_sampling_degrees,
            velocity_delta_threshold, continental_cache_size). These are
            applied after time_step and n_points, so they can also
            override those if needed. Default: None.
        checkpoint_interval_myr: Save ocean tracker state every N Myr
            of evolution. Subsequent runs that find a checkpoint close to
            their starting age will load it instead of stepping from
            ``oldest_age``, dramatically reducing idle time on non-root
            MPI ranks. None disables checkpointing. Default: None.
        checkpoint_dir: Directory for checkpoint files. When
            checkpointing is enabled and this is None, defaults to
            ``./gtrack_checkpoints/``. Checkpoints capture tracer
            positions and ages only; changing config parameters between
            runs is safe.
    """

    # Ocean tracker parameters
    time_step: float = 1.0
    n_points: int = 10000
    reinit_interval_myr: float = 50.0

    # Interpolation
    interpolation: InterpolationConfig = None  # set in __post_init__

    # Mesh geometry parameters
    r_outer: float = 2.208
    depth_scale: float = 2890.0
    transition_width: float = 10.0

    # Data parameters
    property_name: str = "thickness"

    # Pass-through to gtrack's TracerConfig
    gtrack_config: dict | None = None

    # Garbage collection: call gc.collect() every N get_indicator calls; None disables
    gc_collect_frequency: int | None = 1

    # Checkpointing: save/load ocean tracker state to avoid long spin-up.
    # None disables checkpointing (default). Set to e.g. 10.0 to write a
    # checkpoint every 10 Myr of tracker evolution.
    checkpoint_interval_myr: float | None = None
    # Directory for checkpoint files. When checkpointing is enabled and
    # this is None, defaults to ./gtrack_checkpoints/.
    checkpoint_dir: str | None = None

    def __post_init__(self):
        if self.interpolation is None:
            self.interpolation = InterpolationConfig(default_value=100.0)
        super().__post_init__()
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.reinit_interval_myr <= 0:
            raise ValueError(
                f"reinit_interval_myr must be positive, got {self.reinit_interval_myr}"
            )
        if self.gc_collect_frequency is not None and self.gc_collect_frequency < 1:
            raise ValueError(
                f"gc_collect_frequency must be >= 1 or None, "
                f"got {self.gc_collect_frequency}"
            )
        if self.checkpoint_interval_myr is not None and self.checkpoint_interval_myr <= 0:
            raise ValueError(
                f"checkpoint_interval_myr must be positive or None, "
                f"got {self.checkpoint_interval_myr}"
            )


class LithosphereConnector(IndicatorConnector):
    """Connector for lithosphere indicator field through geological time.

    Combines oceanic lithosphere (age-tracked via gtrack's SeafloorAgeTracker,
    converted to thickness) with continental lithosphere (back-rotated
    present-day thickness data via gtrack's PointRotator) to produce a smooth
    3D indicator field: 1 inside lithosphere, 0 in mantle.

    The indicator uses a tanh transition for numerical stability:
        indicator = 0.5 * (1 + tanh((r - lith_base) / transition_width))

    where lith_base = r_outer - thickness (in mesh units).

    MPI Parallelization:
        When a communicator is provided, only rank 0 performs I/O and gtrack
        computations. The resulting thickness data is broadcast to all ranks,
        and each rank interpolates to its local mesh points.

    Args:
        gplates_connector (pyGplatesConnector): Connector with plate model files.
            Must have `continental_polygons` and `static_polygons` set.
        continental_data: Present-day continental thickness data. Can be:
            - gtrack.PointCloud with the thickness property
            - Path to HDF5/NetCDF file
            - Tuple of (latlon_array, thickness_values_km)
        age_to_property (Callable): Function mapping seafloor age arrays (Myr) to
            thickness arrays (km). Signature: np.ndarray -> np.ndarray.
        config (LithosphereConfig, optional): If None, uses LithosphereConfig().
        config_extra (dict, optional): Parameter overrides on top of config.
        comm: MPI communicator for parallel execution.

    Examples:
        >>> def half_space_cooling(age_myr):
        ...     age_sec = age_myr * 3.15576e13
        ...     return 2.32 * np.sqrt(1e-6 * age_sec) / 1e3  # km
        >>>
        >>> connector = LithosphereConnector(
        ...     gplates_connector=plate_model,
        ...     continental_data='SL2013sv.nc',
        ...     age_to_property=half_space_cooling,
        ...     comm=mesh.comm,
        ... )
        >>> indicator = connector.get_indicator(mesh_coords, ndtime)
    """

    def __init__(
        self,
        gplates_connector: "pyGplatesConnector",
        continental_data,
        age_to_property: Callable[[np.ndarray], np.ndarray],
        config: LithosphereConfig | None = None,
        config_extra: dict | None = None,
        # Ideally the same as mesh.comm; defaults to COMM_WORLD which
        # is correct for single-communicator runs.
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.gplates_connector = gplates_connector
        self.age_to_property = age_to_property
        self.comm = comm
        self._is_root = (comm.rank == 0)

        if config is None:
            config = LithosphereConfig()
        if config_extra is not None:
            config = config.with_overrides(config_extra)
        self.config = config

        self._transition_width_nondim = config.transition_width / config.depth_scale

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

        self._PointCloud = PointCloud

        if self._is_root:
            tracer_kwargs = {
                "time_step": config.time_step,
                "default_mesh_points": config.n_points,
            }
            if config.gtrack_config is not None:
                tracer_kwargs.update(config.gtrack_config)
            tracer_config = TracerConfig(**tracer_kwargs)
            self._ocean_tracker = SeafloorAgeTracker(
                rotation_files=gplates_connector.rotation_filenames,
                topology_files=gplates_connector.topology_filenames,
                continental_polygons=gplates_connector.continental_polygons,
                config=tracer_config,
            )
            self._rotator = PointRotator(
                rotation_files=gplates_connector.rotation_filenames,
                static_polygons=gplates_connector.static_polygons,
            )
            self._polygon_filter = PolygonFilter(
                polygon_files=gplates_connector.continental_polygons,
                rotation_files=gplates_connector.rotation_filenames,
            )
            self._continental_present = self._load_data(continental_data)
        else:
            self._ocean_tracker = None
            self._rotator = None
            self._polygon_filter = None
            self._continental_present = None

        self._initialized = False
        self._last_reinit_age = None
        self._last_checkpoint_age = None
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

        # Resolve checkpoint directory (rank 0 only needs it, but store on
        # all ranks so the config is inspectable).
        if config.checkpoint_interval_myr is not None:
            self._checkpoint_dir = (
                config.checkpoint_dir or "./gtrack_checkpoints"
            )
        else:
            self._checkpoint_dir = None

    def _validate_age_extra(self, age: float):
        """Check that ages are requested in decreasing order (forward in time).

        Uses reconstruction_age (set on all ranks) rather than the ocean
        tracker's internal state (rank 0 only) so that all ranks raise
        consistently and avoid MPI deadlocks.
        """
        if self._initialized and self.reconstruction_age is not None:
            if age > self.reconstruction_age:
                raise ValueError(
                    f"Requested age {age:.2f} Ma is older than the last computed "
                    f"age ({self.reconstruction_age:.2f} Ma). The ocean tracker "
                    f"can only evolve forward in time (decreasing age)."
                )

    @staticmethod
    def _find_best_checkpoint(checkpoint_dir, target_age):
        """Find the checkpoint file closest to (but not younger than) target_age.

        Scans checkpoint_dir for files matching ``ocean_checkpoint_<age>Ma.npz``
        and returns the path whose age is the smallest value >= target_age,
        i.e. the one that minimises the amount of forward stepping needed.

        Returns None if no suitable checkpoint exists or the directory is
        missing / empty.
        """
        if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
            return None

        pattern = re.compile(r"^ocean_checkpoint_(\d+)Ma\.npz$")
        best_path = None
        best_age = None

        for fname in os.listdir(checkpoint_dir):
            m = pattern.match(fname)
            if m is None:
                continue
            ckpt_age = int(m.group(1))
            if ckpt_age < target_age:
                continue
            if best_age is None or ckpt_age < best_age:
                best_age = ckpt_age
                best_path = os.path.join(checkpoint_dir, fname)

        return best_path

    def _save_checkpoint_if_due(self, age):
        """Write a checkpoint if the interval has been reached.

        Failures are logged but never crash the simulation.
        """
        if self._checkpoint_dir is None:
            return
        interval = self.config.checkpoint_interval_myr
        if interval is None:
            return
        if (self._last_checkpoint_age is not None
                and abs(self._last_checkpoint_age - age) < interval):
            return

        rounded_age = int(round(age))
        filepath = os.path.join(
            self._checkpoint_dir, f"ocean_checkpoint_{rounded_age}Ma.npz"
        )
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            self._ocean_tracker.save_checkpoint(filepath)
            self._last_checkpoint_age = rounded_age
            log(f"LithosphereConnector: Saved ocean checkpoint at "
                f"{rounded_age} Ma to {filepath}.", level=DEBUG)
        except Exception as exc:
            log(f"LithosphereConnector: Failed to save checkpoint at "
                f"{rounded_age} Ma: {exc}", level=INFO)

    def _prepare_sources(
        self,
        age: float,
        default_continental_age: float = 500.0,
        allow_reinit: bool = True,
    ) -> dict[str, np.ndarray]:
        """Step ocean tracker + rotate continental data, combine into source arrays.

        Returns xyz, thickness, and age arrays. The tanh indicator
        computation ignores the age key; the geotherm computation needs
        it for age-dependent temperature profiles.

        Args:
            age: Target geological age (Ma).
            default_continental_age: Age (Myr) assigned to continental
                points that have no tracked age. Default: 500.0.
            allow_reinit: Passed through to ``_step_ocean_to``.
                Geotherm connectors that share this tracker pass False
                so that only the indicator connector drives
                reinitialisation.
        """
        ocean_cloud = self._step_ocean_to(age, allow_reinit=allow_reinit)
        ocean_ages = ocean_cloud.get_property("age")
        ocean_cloud.add_property(
            self.config.property_name, self.age_to_property(ocean_ages)
        )

        cont_cloud = self._rotator.rotate(
            self._continental_present, from_age=0.0, to_age=age
        )
        cont_cloud.add_property(
            "age", np.full(cont_cloud.n_points, default_continental_age)
        )

        combined = PointCloud.concatenate([ocean_cloud, cont_cloud], warn=False)
        return {
            "xyz": combined.xyz.copy(),
            "thickness": combined.get_property(self.config.property_name).copy(),
            "age": combined.get_property("age").copy(),
        }

    def _step_ocean_to(self, age: float, allow_reinit: bool = True) -> "PointCloud":
        """Initialize/reinit ocean tracker and step to target age.

        Must be called on rank 0 only.

        On the first call, attempts to load the most recent checkpoint that
        is at or before the target age (if checkpointing is configured and
        a suitable file exists). Falls back to full initialisation from
        ``oldest_age`` when no checkpoint is available.

        Args:
            age: Target geological age (Ma).
            allow_reinit: If False, skip the periodic reinitialisation check.
                Geotherm connectors that share this tracker pass False so
                that only the indicator connector drives reinitialisation.
        """
        if not self._initialized:
            loaded = False
            best = self._find_best_checkpoint(self._checkpoint_dir, age)
            if best is not None:
                try:
                    self._ocean_tracker.load_checkpoint(best)
                    loaded_age = self._ocean_tracker.current_age
                    log(f"LithosphereConnector: Loaded ocean checkpoint at "
                        f"{loaded_age} Ma from {best}.", level=DEBUG)
                    self._last_reinit_age = loaded_age
                    self._last_checkpoint_age = loaded_age
                    loaded = True
                except Exception as exc:
                    log(f"LithosphereConnector: Failed to load checkpoint "
                        f"{best}: {exc}. Falling back to full "
                        f"initialisation.", level=INFO)

            if not loaded:
                # PyGPlates uses integer ages (Ma) for reconstruction.
                starting_age = int(self.gplates_connector.oldest_age)
                log(f"LithosphereConnector: Initializing ocean tracker at "
                    f"{starting_age} Ma.", level=DEBUG)
                self._ocean_tracker.initialize(starting_age=starting_age)
                self._last_reinit_age = starting_age

            self._initialized = True

        if allow_reinit and self._last_reinit_age is not None:
            if abs(self._last_reinit_age - age) >= self.config.reinit_interval_myr:
                log(f"LithosphereConnector: Reinitializing ocean tracker at "
                    f"{age:.2f} Ma.", level=DEBUG)
                self._ocean_tracker.reinitialize(n_points=self.config.n_points)
                self._last_reinit_age = age

        # PyGPlates works with integer reconstruction ages (Ma), so we
        # round to the nearest integer before stepping the tracker.
        cloud = self._ocean_tracker.step_to(int(round(age)))

        if allow_reinit:
            self._save_checkpoint_if_due(age)

        return cloud


@dataclass
class PolygonConfig(IndicatorConfigBase):
    """Configuration for polygon-based indicator computation.

    Groups all tunable parameters for PolygonConnector into a single
    configuration object. Similar to LithosphereConfig but without
    ocean age tracking parameters.

    Args:
        n_points: Number of sample points for polygon coverage. Default: 20000.
        interpolation: Interpolation parameters. See :class:`InterpolationConfig`.
            Default: ``InterpolationConfig(default_value=200.0)``.
        r_outer: Outer mesh radius in non-dimensional units. Default: 2.208.
        depth_scale: Physical depth (km) per non-dimensional unit. Default: 2890.0.
        transition_width: Tanh transition width in km. Default: 10.0.
        property_name: Thickness property name in data files. Default: 'thickness'.
    """

    n_points: int = 20000
    interpolation: InterpolationConfig = None  # set in __post_init__
    r_outer: float = 2.208
    depth_scale: float = 2890.0
    transition_width: float = 10.0
    property_name: str = "thickness"

    # Garbage collection: call gc.collect() every N get_indicator calls; None disables
    gc_collect_frequency: int | None = 1

    def __post_init__(self):
        if self.interpolation is None:
            self.interpolation = InterpolationConfig(default_value=200.0)
        super().__post_init__()
        if self.gc_collect_frequency is not None and self.gc_collect_frequency < 1:
            raise ValueError(
                f"gc_collect_frequency must be >= 1 or None, "
                f"got {self.gc_collect_frequency}"
            )


class PolygonConnector(IndicatorConnector):
    """Connector for polygon-based indicator field through geological time.

    Computes a smooth 3D indicator field for regions defined by polygon
    boundaries. The indicator is ~1 inside the region, ~0 outside, with a
    smooth tanh transition at the region base.

    Differs from the base-class tanh behaviour: points far from any source
    data receive indicator = 0 (outside the polygon region).

    Args:
        gplates_connector (pyGplatesConnector): Connector with plate model files.
            Must have `static_polygons` set for rotation.
        polygons: Path to polygon shapefile.
        thickness_data: Present-day thickness data (PointCloud, path, tuple, or scalar).
        config (PolygonConfig, optional): If None, uses PolygonConfig().
        config_extra (dict, optional): Parameter overrides.
        comm: MPI communicator for parallel execution.

    Examples:
        >>> connector = PolygonConnector(
        ...     gplates_connector=plate_model,
        ...     polygons='shapes_cratons.shp',
        ...     thickness_data='craton_thickness.h5',
        ...     comm=mesh.comm,
        ... )
        >>> indicator = connector.get_indicator(mesh_coords, ndtime)
    """

    def __init__(
        self,
        gplates_connector: "pyGplatesConnector",
        polygons,
        thickness_data,
        config: PolygonConfig | None = None,
        config_extra: dict | None = None,
        # Ideally the same as mesh.comm; defaults to COMM_WORLD which
        # is correct for single-communicator runs.
        comm: MPI.Comm = MPI.COMM_WORLD,
    ):
        self.gplates_connector = gplates_connector
        self.comm = comm
        self._is_root = (comm.rank == 0)

        if config is None:
            config = PolygonConfig()
        if config_extra is not None:
            config = config.with_overrides(config_extra)
        self.config = config

        self._transition_width_nondim = config.transition_width / config.depth_scale

        if gplates_connector.static_polygons is None:
            raise ValueError(
                "gplates_connector must have static_polygons set. "
                "Pass static_polygons to pyGplatesConnector constructor."
            )

        self._PointCloud = PointCloud

        if self._is_root:
            self._polygon_filter = PolygonFilter(
                polygon_files=polygons,
                rotation_files=gplates_connector.rotation_filenames,
            )
            self._rotator = PointRotator(
                rotation_files=gplates_connector.rotation_filenames,
                static_polygons=gplates_connector.static_polygons,
            )
            self._region_present = self._load_data(thickness_data)
        else:
            self._polygon_filter = None
            self._rotator = None
            self._region_present = None

        self._initialized = False
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

    def _prepare_sources(self, age: float) -> dict[str, np.ndarray]:
        """Rotate region data to target age."""
        region_cloud = self._rotator.rotate(
            self._region_present, from_age=0.0, to_age=age
        )
        log(f"PolygonConnector: {region_cloud.n_points} region points at "
            f"{age:.2f} Ma.", level=DEBUG)
        return {
            "xyz": region_cloud.xyz.copy(),
            "thickness": region_cloud.get_property(self.config.property_name).copy(),
        }

    def _apply_indicator(self, r_target, thickness_km, too_far):
        """Zero out indicator for points far from any polygon data."""
        indicator = super()._apply_indicator(r_target, thickness_km, too_far)
        indicator[too_far] = 0.0
        return indicator


# ---------------------------------------------------------------------------
# Default geotherm functions
# ---------------------------------------------------------------------------

def ocean_erf_normalized(depth_m, z_lab_m, age_myr=None, kappa=1e-6, **kwargs):
    """Normalized erf geotherm for oceanic lithosphere.

    Returns T_norm = erf(z / a) / erf(z_lab / a), where a = 2 * sqrt(kappa * t).
    T = 0 at the surface and T = 1 at the LAB depth.

    Args:
        depth_m: Depth below the surface in meters.
        z_lab_m: LAB depth in meters.
        age_myr: Seafloor age in Myr. Falls back to linear if None or zero.
        kappa: Thermal diffusivity in m^2/s. Default: 1e-6.

    Returns:
        Normalized temperature in [0, 1].
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)

    result = np.zeros_like(depth_m)

    if age_myr is None:
        valid = z_lab_m > 0
        result[valid] = depth_m[valid] / z_lab_m[valid]
        return np.clip(result, 0.0, 1.0)

    age_myr = np.asarray(age_myr, dtype=float)
    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13

    a = 2.0 * np.sqrt(kappa * np.maximum(age_sec, 1.0))

    valid = z_lab_m > 0
    erf_z = erf(depth_m[valid] / a[valid])
    erf_zlab = erf(z_lab_m[valid] / a[valid])
    safe = erf_zlab > 1e-10
    result[valid] = np.where(safe, erf_z / np.maximum(erf_zlab, 1e-10),
                             depth_m[valid] / z_lab_m[valid])

    return np.clip(result, 0.0, 1.0)


def continental_linear(depth_m, z_lab_m, **kwargs):
    """Linear geotherm for continental lithosphere.

    Returns T_norm = z / z_lab. T = 0 at surface, T = 1 at LAB.

    Args:
        depth_m: Depth below the surface in meters.
        z_lab_m: LAB depth in meters.

    Returns:
        Normalized temperature in [0, 1].
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)

    result = np.zeros_like(depth_m)
    valid = z_lab_m > 0
    result[valid] = depth_m[valid] / z_lab_m[valid]
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Geotherm connectors (composition wrappers around indicator connectors)
# ---------------------------------------------------------------------------

class LithosphereGeotherm(IndicatorConnector):
    """Normalized geotherm temperature from a shared LithosphereConnector.

    Wraps an existing ``LithosphereConnector`` to reuse its ocean age
    tracker and continental rotation machinery, avoiding a duplicate
    ``SeafloorAgeTracker``.  The wrapped connector must be updated
    (via ``get_indicator`` or ``update_plate_reconstruction``) before
    this geotherm at each time step, because the ocean tracker can only
    evolve forward in time (decreasing geological age).  If either
    connector advances the tracker, neither can go back.

    Configuration (interpolation resolution, transition width, etc.)
    is inherited from the wrapped connector.  If different parameters
    are needed, create a separate ``LithosphereConnector``.

    Values represent (T - Ts) / (Tlab - Ts) in [0, 1].

    Args:
        lithosphere_connector: An existing LithosphereConnector whose
            tracker and rotation infrastructure will be shared.
        geotherm: Geotherm function (default: ocean_erf_normalized).
        kappa: Thermal diffusivity in m^2/s. Default: 1e-6.
        default_continental_age: Age (Myr) assigned to continental
            points for the erf profile. Default: 500.0.

    Examples:
        >>> lith = LithosphereConnector(gplates, data, half_space, comm=mesh.comm)
        >>> geotherm = LithosphereGeotherm(lith, geotherm=ocean_erf_normalized)
        >>> T_erf = GplatesScalarFunction(Q, indicator_connector=geotherm)
    """

    def __init__(
        self,
        lithosphere_connector: LithosphereConnector,
        geotherm=None,
        kappa: float = 1e-6,
        default_continental_age: float = 500.0,
    ):
        self._source = lithosphere_connector

        # Delegate IndicatorConnector required attributes
        self.gplates_connector = lithosphere_connector.gplates_connector
        self.config = lithosphere_connector.config
        self.comm = lithosphere_connector.comm
        self._is_root = lithosphere_connector._is_root
        self._transition_width_nondim = lithosphere_connector._transition_width_nondim

        self._geotherm = geotherm or ocean_erf_normalized
        self._kappa = kappa
        self._default_continental_age = default_continental_age

        # Independent cache (geotherm values differ from indicator values)
        self._initialized = False
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

    def _validate_age_extra(self, age: float):
        """Delegate to the source connector's ocean-tracker check."""
        self._source._validate_age_extra(age)

    def _prepare_sources(self, age: float) -> dict[str, np.ndarray]:
        """Delegate to the source connector without triggering reinitialisation."""
        return self._source._prepare_sources(
            age,
            default_continental_age=self._default_continental_age,
            allow_reinit=False,
        )

    def _compute_indicator(self, sources, target_coords):
        """Apply geotherm function instead of tanh."""
        source_xyz = sources["xyz"]
        if source_xyz is None or len(source_xyz) == 0:
            return self._empty_indicator(len(target_coords))

        r_target = np.linalg.norm(target_coords, axis=1)
        depth_m = (self.config.r_outer - r_target) * self.config.depth_scale * 1e3

        (thickness_km, age_myr), too_far = self._interpolate(
            source_xyz, target_coords, sources["thickness"], sources["age"]
        )
        thickness_km[too_far] = self.config.interpolation.default_value
        age_myr[too_far] = self._default_continental_age

        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m, age_myr=age_myr, kappa=self._kappa)
        return np.clip(T_norm, 0.0, 1.0)


class PolygonGeotherm(IndicatorConnector):
    """Normalized geotherm temperature from a shared PolygonConnector.

    Wraps an existing ``PolygonConnector`` to reuse its polygon
    filtering and rotation machinery.  Configuration is inherited from
    the wrapped connector.

    Values represent (T - Ts) / (Tlab - Ts) in [0, 1].

    Args:
        polygon_connector: An existing PolygonConnector whose rotation
            and filtering infrastructure will be shared.
        geotherm: Geotherm function (default: continental_linear).

    Examples:
        >>> cont = PolygonConnector(gplates, polygons, data, comm=mesh.comm)
        >>> geotherm = PolygonGeotherm(cont, geotherm=continental_linear)
        >>> T_lin = GplatesScalarFunction(Q, indicator_connector=geotherm)
    """

    def __init__(
        self,
        polygon_connector: PolygonConnector,
        geotherm=None,
    ):
        self._source = polygon_connector

        # Delegate IndicatorConnector required attributes
        self.gplates_connector = polygon_connector.gplates_connector
        self.config = polygon_connector.config
        self.comm = polygon_connector.comm
        self._is_root = polygon_connector._is_root
        self._transition_width_nondim = polygon_connector._transition_width_nondim

        self._geotherm = geotherm or continental_linear

        # Independent cache
        self._initialized = False
        self.reconstruction_age = None
        self._cached_result = None
        self._cached_coords_hash = None

    def _prepare_sources(self, age: float) -> dict[str, np.ndarray]:
        """Delegate to the source connector."""
        return self._source._prepare_sources(age)

    def _empty_indicator(self, n):
        """Outside any polygon region, use mantle temperature."""
        return np.ones(n)

    def _compute_indicator(self, sources, target_coords):
        """Apply geotherm function instead of tanh."""
        source_xyz = sources["xyz"]
        if source_xyz is None or len(source_xyz) == 0:
            return self._empty_indicator(len(target_coords))

        r_target = np.linalg.norm(target_coords, axis=1)
        depth_m = (self.config.r_outer - r_target) * self.config.depth_scale * 1e3

        (thickness_km,), too_far = self._interpolate(
            source_xyz, target_coords, sources["thickness"]
        )

        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m)
        T_norm = np.clip(T_norm, 0.0, 1.0)

        # Points far from any source data -> mantle temperature
        T_norm[too_far] = 1.0
        return T_norm


class GplatesScalarFunction(fd.Function):
    """Firedrake Function for scalar indicator fields from plate reconstructions.

    Works with any IndicatorConnector subclass. Creates a 3D scalar field
    that is ~1 in regions of interest and ~0 elsewhere, with smooth
    transitions at boundaries.

    Args:
        function_space: Scalar function space.
        indicator_connector: Any IndicatorConnector subclass.
        name: Optional name for the function.
        **kwargs: Additional arguments passed to Firedrake Function.

    Examples:
        >>> lith_connector = LithosphereConnector(..., comm=mesh.comm)
        >>> lithosphere = GplatesScalarFunction(
        ...     Q, indicator_connector=lith_connector, name="Lithosphere")
        >>> lithosphere.update_plate_reconstruction(ndtime=0.5)
    """

    def __init__(
        self,
        function_space,
        *args,
        indicator_connector: IndicatorConnector | None = None,
        name: str | None = None,
        **kwargs
    ):
        super().__init__(function_space, *args, name=name, **kwargs)

        if indicator_connector is None:
            return

        if not isinstance(indicator_connector, IndicatorConnector):
            raise TypeError(
                f"indicator_connector must be an IndicatorConnector subclass, "
                f"got {type(indicator_connector).__name__}"
            )

        self.indicator_connector = indicator_connector

        mesh = extract_unique_domain(self)

        scalar_element = function_space.ufl_element()
        vector_element = VectorElement(scalar_element)
        coords_space = fd.FunctionSpace(mesh, vector_element)
        coords_func = fd.Function(coords_space)
        coords_func.interpolate(fd.SpatialCoordinate(mesh))
        self.mesh_coords = coords_func.dat.data_ro_with_halos.copy()

    def update_plate_reconstruction(self, ndtime: float):
        """Update indicator field for given non-dimensional time.

        Only updates when the reconstruction age has changed beyond delta_t.
        """
        connector = self.indicator_connector
        age = connector.ndtime2age(ndtime)

        if connector.reconstruction_age is not None:
            if abs(age - connector.reconstruction_age) < connector.delta_t:
                return

        with stop_annotating():
            values = connector.get_indicator(self.mesh_coords, ndtime)
            self.dat.data_with_halos[:] = values

        if annotate_tape():
            self.create_block_variable()

    def _ad_mul(self, other):
        r = GplatesScalarFunction(self.function_space())
        r.assign(other * self)
        return r

    def _ad_add(self, other):
        r = GplatesScalarFunction(self.function_space())
        fd.Function.assign(r, self + other)
        return r
