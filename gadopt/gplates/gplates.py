import warnings
from contextlib import nullcontext
import firedrake as fd
import numpy as np
from firedrake.ufl_expr import extract_unique_domain
from pyadjoint.tape import annotate_tape, stop_annotating
from scipy.spatial import cKDTree
from ..utility import log, upward_normal

import pygplates

__all__ = [
    "GplatesVelocityFunction",
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
        log(f"pyGplates: Updating surface velocities for {self.gplates_connector.ndtime2age(ndtime):.2f} Ma.")
        # Check if we need to update plate velocities at all
        if self.gplates_connector.reconstruction_age is not None:
            if abs(self.gplates_connector.ndtime2age(ndtime) - self.gplates_connector.reconstruction_age) < self.gplates_connector.delta_t:
                return

        # Assuming `self` is a Firedrake Function instance,
        self.dat.data_with_halos[self.dbc.nodes, :] = (
            self.gplates_connector.get_plate_velocities(
                self.boundary_coords, ndtime)
        )

        # Project tangential with appropriate annotation handling
        suitable_context = stop_annotating() if annotate_tape() else nullcontext()
        with suitable_context:
            self.remove_radial_component()

        # Create block variable only if annotating
        if annotate_tape():
            self.create_block_variable()


class GplatesVelocityFunction(GPlatesFunctionalityMixin, fd.Function):
    """Extends `firedrake.Function` to incorporate velocities calculated by
    Gplates, coming from plate tectonics reconstion.

    `GplatesVelocityFunction` is designed to associate a Firedrake function with a GPlates
    connector, allowing the integration of plate tectonics reconstructions. This is particularly
    useful when setting "top" boundary condition for the Stokes systems when performing
    data assimilation (sequential or adjoint).

    Attributes:
        dbc (firedrake.DirichletBC): A Dirichlet boundary condition that applies the function
            only to the top boundary.
        boundary_coords (numpy.ndarray): The coordinates of the function located at the
            "top" boundary, normalised, so that it is meaningful for pygplates.
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
    # Solver parameters for tangential projection, since this spherical mesh, use iterative solver
    tangential_project_solver_parameters = {
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "ksp_rtol": 1e-9,
        "ksp_atol": 1e-12,
        "ksp_max_it": 20,
        "ksp_converged_reason": None,
    }

    def __new__(cls, *args, **kwargs):
        # Ensure compatibility with Firedrake Function's __new__ method
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, function_space, *args, gplates_connector=None, top_boundary_marker="top", **kwargs):
        # Initialize as a Firedrake Function
        super().__init__(function_space, *args, **kwargs)

        # Cache all the necessary information that will be used to assign surface velocities
        # the marker for surface boundary. This is typically "top" in extruded mesh.
        self.top_boundary_marker = top_boundary_marker
        # establishing the DirichletBC that will be used to find surface nodes
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

    def remove_radial_component(self):
        """
        Project velocity to tangent plane by removing radial component.

        Args:
            quad_degree (int, optional): Quadrature degree for the projection. Defaults to 6.

        """

        # Get upward normal
        r = upward_normal(self.ufl_domain())

        # Project out radial component
        self.project(
            self - fd.inner(self, r) * r,
            solver_parameters=GplatesVelocityFunction.tangential_project_solver_parameters,
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

    def __init__(self,
                 rotation_filenames,
                 topology_filenames,
                 oldest_age,
                 delta_t=1.,
                 scaling_factor=1.0,
                 nseeds=1e5,
                 nneighbours=4,
                 kappa=1e-6):
        """An interface to pygplates, used for updating top Dirichlet boundary conditions
        using plate tectonic reconstructions.

        This class provides functionality to assign plate velocities at different geological
        ages to the boundary conditions specified with dbc. Due to potential challenges in
        identifying the plate id for a given point with pygplates, especially in high-resolution
        simulations, this interface employs a method of calculating velocities at a number
        of equidistant points on a sphere. It then interpolates these velocities for points
        assigned a plate id. A warning is raised for any point not assigned a plate id.

        Arguments:
            rotation_filenames (Union[str, List[str]]): Collection of rotation file names for pygplates.
            topology_filenames (Union[str, List[str]]): Collection of topology file names for pygplates.
            oldest_age (float): The oldest age present in the plate reconstruction model.
            delta_t (Optional[float]): The t window range outside which plate velocities are updated.
            scaling_factor (Optional[float]): Scaling factor for surface velocities.
            nseeds (Optional[int]): Number of seed points used in the Fibonacci sphere generation. Higher
                    seed point numbers result in finer representation of boundaries in pygpaltes. Notice that
                    the finer velocity variations will then result in more challenging Stokes solves.
            nneighbours (Optional[int]): Number of neighboring points when interpolating velocity for each grid point. Default is 4.
            kappa: (Optional[float]): Diffusion constant used for don-dimensionalising time. Default is 1e-6.

        Examples:
            >>> connector = pyGplatesConnector(rotation_filenames, topology_filenames, oldest_age)
            >>> connector.get_plate_velocities(ndtime=100)
        """

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

        # pyGplates velocity features
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
        for a given non-dimensional time, by integrating plate tectonic reconstructions from pyGplates.

        This method calculates new plate velocities.
        It utilizes the cKDTree data structure for efficient nearest neighbor
        searches to interpolate velocities onto the mesh nodes.

        Args:
            target_coords (array-like): Coordinates of the points at the top of the sphere.
            ndtime (float): The non-dimensional time for which plate velocities are to be calculated and assigned.
                This time is converted to geological age and used to extract relevant plate motions
                from the pyGplates model.

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
        meshnode_feature.set_name('Velocity Mesh Nodes from pygplates')

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

        # if pyGplates does not find a plate id for a point, assings NaN to the velocity
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
                                 out=np.ones_like(tangential_magnitudes),
                                 where=tangential_magnitudes != 0)

        res_u = res_u_tangential * scale_factor[:, np.newaxis]

        return res_u

    def _calc_velocities(self, velocity_domain_features, topology_features, rotation_model, age, delta_t):
        """Calculates velocity vectors for domain points at a specific geological age (Myrs before present day).

        This method calculates the velocities for all points in the velocity domain
        at the specified geological age, using pyGplates to account for tectonic plate
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
        # Note: pygplates can only deal with rounded ages in million years
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
                        # Note: pygplates can only deal with rounded ages in million years
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
                        warnings.warn("pygplates couldn't assign plate IDs to some seeds due to irregularities in the reconstruction model. G-ADOPT will interpolate the nearest values.", category=RuntimeWarning)
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
