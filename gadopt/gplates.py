import warnings
import firedrake as fd
import numpy as np
from firedrake.ufl_expr import extract_unique_domain
from pyadjoint.tape import annotate_tape
from scipy.spatial import cKDTree

import pygplates

__all__ = [
    "GplatesFunction",
    "pyGplatesConnector"
]


class GPlatesFunctionalityMixin:
    def update_plate_reconstruction(self, model_time):
        """A placeholder method to update the Function with data from GPlates.
        Updates the function based on plate tectonics data for a given model time.

        This method fetches the plate velocities from the GPlates connector based on
        the provided model time and applies them to the function's data at the top
        boundary nodes.

        Args:
            model_time (float): The model time for which to update the plate
                velocities. This should be non-dimensionalised time.
        """
        # Check if we need to update plate velocities at all
        if self.gplates_connector.reconstruction_time is not None:
            if abs(self.gplates_connector.ndtime2geotime(model_time) - self.gplates_connector.reconstruction_time) < self.gplates_connector.delta_time:
                return

        # Assuming `self` is a Firedrake Function instance,
        self.dat.data[self.dbc.nodes, :] = (
            self.gplates_connector.get_plate_velocities(
                self.boundary_coords, model_time)
        )

        # At this point the values are updated.
        # So we have to make sure it is shown correctly on tape if we are annotating
        if annotate_tape():
            self.create_block_variable()


class GplatesFunction(GPlatesFunctionalityMixin, fd.Function):
    """Extends `firedrake.Function` to incorporate velocities calculated by
    Gplates, coming from plate tectonics reconstion.

    `GplatesFunction` is designed to associate a Firedrake function with a GPlates
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
        function_space: The function space on which the GplatesFunction is defined.
        gplates_connector: An instance of a pyGplatesConnector, used to integrate
            GPlates functionality or data. See Documentation for pyGplatesConnector.
        top_boundary_marker (defaults to "top"): marker for the top boundary.
        val (optional): Initial values for the function. Defaults to None.
        name (str, optional): Name for the function. Defaults to None.
        dtype (data type, optional): Data type for the function. Defaults to ScalarType.

    Methods:
        update_plate_reconstruction(model_time):
            Updates the function values based on plate velocities from GPlates
            for a given model time.
            **Note** model time is non-dimensionalised

    Example:
        >>> # Assuming necessary imports and setup are done
        >>> gplates_function = GplatesFunction(V,
        ...                                    gplates_connector=pl_rec_model,
        ...                                    name="GplateVelocity")
        >>> gplates_function.update_plate_reconstruction(model_time=0.0)
    """
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


class pyGplatesConnector(object):
    # Non-dimensionalisation constants
    # Factor to non-dimensionalise gplates velocities: d/kappa
    velocity_non_dim_factor = 2890e3/1.0e-6
    # Factor to dimensionalise model time: d^2/kappa
    time_dim_factor = 2890e3**2/1.0e-6
    # 1 yr in seconds
    yrs2sec = 365*24*60*60
    # 1 Myr in seconds
    myrs2sec = 1e6*yrs2sec

    def __init__(self, rotation_filenames, topology_filenames, geologic_zero, delta_time=1., scaling_factor=1.0, nseeds=1000, nneighbours=3):
        """
        An interface to pygplates, used for updating top Dirichlet boundary conditions
        using plate tectonic reconstructions.

        This class provides functionality to assign plate velocities at different geological
        times to the boundary conditions specified with dbc. Due to potential challenges in
        identifying the plate id for a given point with pygplates, especially in high-resolution
        simulations, this interface employs a method of calculating velocities at a number
        of equidistant points on a sphere. It then interpolates these velocities for points
        assigned a plate id. A warning is raised for any point not assigned a plate id.

        Attributes:
            rotation_filenames (str or list of str): Collection of rotation file names for pygplates.
            topology_filenames (str or list of str): Collection of topology file names for pygplates.
            geologic_zero (float): The oldest time present in the plate reconstruction model.
            delta_time (float, optional): The time window range outside which plate velocities are updated. Defaults to 1.0.
            scaling_factor (float, optional): Scaling factor for surface velocities. Defaults to 1.0.
            nseeds (int, optional): Number of seed points used in the Fibonacci sphere generation. Defaults to 1000.
            nneighbours (int, optional): Number of neighboring points to consider in velocity calculations. Defaults to 3.

        Methods:
            assign_plate_velocities(model_time):
                Assigns plate velocities based on the specified model time.
            ndtime2geotime(ndtime):
                Converts non-dimensionalised time to geologic time.
            geotime2ndtime(geotime):
                Converts geologic time to non-dimensionalised time.

        Examples:
            >>> connector = pyGplatesConnector(rotation_filenames, topology_filenames, geologic_zero)
            >>> connector.get_plate_velocities(model_time=100)
        """

        # Rotation model(s)
        self.rotation_model = pygplates.RotationModel(rotation_filenames)

        # Topological plate polygon feature(s).
        self.topology_features = []
        for fname in topology_filenames:
            for f in pygplates.FeatureCollection(fname):
                self.topology_features.append(f)
        # geologic_zero is the same as model_time=0
        self.geologic_zero = geologic_zero

        # time velocity interpolations
        self.delta_time = delta_time

        # seeds are equidistantial points generate on a sphere
        self.seeds = self._fibonacci_sphere(samples=int(nseeds))
        # number of neighbouring points that will be used for interpolation

        self.nneighbours = nneighbours
        self.velocity_domain_features = (
            self._make_GPML_velocity_feature(
                self.seeds
            )
        )

        # last reconstruction time
        self.reconstruction_time = None

        # Flag to know when to recalculate surface velocities.
        self.recalculation_flg = False

        # Factor to scale plate velocities to RMS velocity of model,
        # the definition: RMS_Earth / RMS_Model is used
        # This is specially used in low-Rayleigh-number simulations
        self.scaling_factor = scaling_factor

    # setting the time that we are interested in
    def get_plate_velocities(self, target_coords, model_time):
        """
        Returns plate velocities for the specified target coordinates at the top boundary of a sphere,
        for a given model time, by integrating plate tectonic reconstructions from pyGplates.

        This method calculates new plate velocities.
        It utilizes the cKDTree data structure for efficient nearest neighbor
        searches to interpolate velocities onto the mesh nodes.

        Args:
            target_coords (array-like): Coordinates of the points at the top of the sphere.
            model_time (float): The model time for which plate velocities are to be calculated and assigned.
                This time is converted to geological time and used to extract relevant plate motions
                from the pyGplates model.

        Raises:
            Exception: If the requested reconstruction time is geologically negative, indicating an issue with
                the time conversion.

        Notes:
            - The method uses conversions between non-dimensionalised time and geologic time.
            - Velocities are non-dimensionalised and scaled for the simulation before being applied.
        """

        # Raising an error if the user is asking for invalid time
        if self.ndtime2geotime(ndtime=model_time) < 0:
            raise Exception(
                ("pyGplates: geologic time is being negative!"
                 f"maximum: {self.geologic_zero/(pyGplatesConnector.time_dim_factor/pyGplatesConnector.myrs2sec/self.scaling_factor)}")
            )

        # compute new velocities
        self.reconstruction_time = self.ndtime2geotime(model_time)
        self.interpolated_u = self._interpolate_seeds_u(target_coords)
        return self.interpolated_u

    def ndtime2geotime(self, ndtime):
        """
        Converts non-dimensionalized time to geologic time with respect to present-day (Ma).

        This function takes a non-dimensionalized time value and converts it to geologic time,
        measured in millions of years ago (Ma), with respect to the present day.

        Args:
            ndtime (float): The non-dimensionalized time to be converted.

        Returns:
            float: The converted geologic time in millions of years ago (Ma).
        """

        geotime = self.geologic_zero - float(ndtime) * pyGplatesConnector.time_dim_factor/pyGplatesConnector.myrs2sec/self.scaling_factor
        return geotime

    def geotime2ndtime(self, geotime):
        """ converts geologic time with respect to present-day (Ma) to non-dimensionalise time:

        Args:
            geotime (float): geologic time (before presentday in Myrs)

        Returns:
            float: non-dimensionalised time
         """
        ndtime = (self.geologic_zero - geotime)*(pyGplatesConnector.myrs2sec*self.scaling_factor / pyGplatesConnector.time_dim_factor)
        return ndtime

    # convert seeds to Gplate features
    def _make_GPML_velocity_feature(self, coords):
        """
        Creates a pygplates.GPML velocity feature from specified coordinates.

        This function takes a set of coordinates and converts them into a GPML velocity feature,
        suitable for use with pyGplates. The created feature is a collection of points on a sphere,
        each representing a node in a velocity mesh, used for calculating tectonic plate velocities
        at those points.

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
        """
        Interpolates seed velocities onto target coordinates.

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
            time=round(self.reconstruction_time, 0),
            delta_time=self.delta_time)

        seeds_u = np.array([i.to_xyz() for i in seeds_u]) *\
            ((1e-2 * pyGplatesConnector.velocity_non_dim_factor) / (self.scaling_factor * pyGplatesConnector.yrs2sec))

        # generate a KD-tree of the seeds points that have a numerical value
        tree = cKDTree(data=self.seeds[seeds_u[:, 0] == seeds_u[:, 0], :], leafsize=16)

        # find the neighboring points
        dists, idx = tree.query(x=target_coords, k=self.nneighbours)

        # Use np.where to avoid division by zero
        # Replace 0 distances with np.finfo(float).eps to avoid division
        # by zero while keeping original zeros intact for later use
        safe_dists = np.where(dists == 0, np.finfo(float).eps, dists)

        # Then, calculate the weighted average using safe_dists
        res_u = np.einsum(
            'i, ij->ij',
            1/np.sum(1/safe_dists, axis=1),
            np.einsum('ij, ijk ->ik', 1/safe_dists, seeds_u[seeds_u[:, 0] == seeds_u[:, 0]][idx])
        )

        # Now handle the case where points are too close to each other:
        res_u[dists[:, 0] <= 1e-8, :] = seeds_u[seeds_u[:, 0] == seeds_u[:, 0]][idx[dists[:, 0] <= 1e-8, 0]]

        return res_u

    def _calc_velocities(self, velocity_domain_features, topology_features, rotation_model, time, delta_time):
        """
        Calculates velocity vectors for domain points at a specific geological time.

        This method calculates the velocities for all points in the velocity domain
        at the specified geological time, using pyGplates to account for tectonic plate
        motions. It utilizes a plate partitioner to determine the tectonic plate each
        point belongs to and computes the velocity based on the rotation model and
        the change over the specified time interval.

        Args:
            velocity_domain_features: Domain features representing the points at which velocities are to be calculated.
            topology_features: Topological features of tectonic plates used in the partitioning.
            rotation_model: The rotation model defining plate movements over time.
            time (float): The geological time at which velocities are to be calculated.
            delta_time (float): The time interval over which velocity is calculated.

        Returns:
            list of pygplates.Vector3D: Velocity vectors for all domain points.

        Raises:
            RuntimeWarning: If no plate ID is found for a given point, indicating an issue with the reconstruction model.
        """
        # All domain points and associated (magnitude, azimuth, inclination) velocities for the current time.
        all_domain_points = []
        all_velocities = []

        # Partition our velocity domain features into our topological plate polygons at the current 'time'.
        # Note: pygplates can only deal with rounded ages in million years
        plate_partitioner = pygplates.PlatePartitioner(topology_features, rotation_model, round(time, 0))

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

                        # Get the stage rotation of partitioning plate from 'time + delta_time' to 'time'.
                        # Note: pygplates can only deal with rounded ages in million years
                        equivalent_stage_rotation = rotation_model.get_rotation(round(time, 0), partitioning_plate_id, round(time, 0) + delta_time)

                        # Calculate velocity at the velocity domain point.
                        # This is from 'time + delta_time' to 'time' on the partitioning plate.
                        # NB: velocity unit is fixed to cm/yr, but we convert it to m/yr and further on non-dimensionalise it later.
                        velocity_vectors = pygplates.calculate_velocities(
                            [velocity_domain_point],
                            equivalent_stage_rotation,
                            delta_time, velocity_units=pygplates.VelocityUnits.cms_per_yr)

                        # add it to the list
                        all_velocities.extend(velocity_vectors)
                    else:
                        warnings.warn("No plate id found. There is an issue with the reconstruciton model.", category=RuntimeWarning)
                        all_velocities.extend([pygplates.Vector3D(np.NaN, np.NaN, np.NaN)])

        return all_velocities

    def _fibonacci_sphere(self, samples):
        """
        Generates points on a sphere using the Fibonacci sphere algorithm, which
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

        y = 1 - (np.array(list(range(samples)))/(samples-1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * np.array(list(range(samples)))
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.array([[x[i], y[i], z[i]] for i in range(len(x))])
