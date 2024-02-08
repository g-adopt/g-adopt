import numpy as np
import pygplates
import firedrake as fd
from firedrake.ufl_expr import extract_unique_domain
import numpy
from gadopt.utility import log
from scipy.spatial import cKDTree
import warnings

# non-dimensionalisation constants
# Factor to non-dimensionalise gplates velocities: d/kappa
velocity_non_dim_factor = 2890e3/1.0e-6
# Factor to dimensionalise model time: d^2/kappa
time_dim_factor = 2890e3**2/1.0e-6
# 1 yr in seconds
yrs2sec = 365*24*60*60
# 1 Myr in seconds
myrs2sec = 1e6*yrs2sec


class pyGplatesConnector(object):
    delta_t = 0.1

    def __init__(self, rotation_filenames, topology_filenames, geologic_zero, dbc, delta_time=0.95, scaling_factor=1.0, nseeds=1000, nneighbours=3):
        """
        An interface to pygplates, used for updating top Dirichlet boundary condition
        using plate tectonic reconstructions.

        The class provides functionality to assign plate velocities at different geological
        times to the boundary conditions specified with dbc. As pygplates can be sometimes
        fidlly with finding the plate id for a given point, specially for high resolution
        simulations, we use [#nseeds] number equidistantial points on a sphere to calculate
        velocities, and then use those points that have been assigned a plate id, and interpolate
        between. Any time a point is not assigned with a plate ide, we raise a Warning.

        Parameters
        ----------
        rotation_filenames : str or list of str
            Collection of rotation file names for pygplates.
        topology_filenames : str or list of str
            Collection of topology file names for pygplates.
        geologic_zero : float
            The oldest time present in the plate reconstruction model.
        dbc : firedrake.bcs.DirichletBC
            The Dirichlet boundary condition object from Firedrake.
        delta_time : float, optional
            The time window range outside of which plate velocities will be updated.
            Defaults to 1.0.
        scaling_factor : float, optional
            Scaling factor for surface velocities. Defaults to 1.0.
        nseeds : int, optional
            Number of seed points used in the Fibonacci sphere generation. Defaults to 1000.
        nneighbours : int, optional
            Number of neighboring points to consider in velocity calculations. Defaults to 3.

        Methods
        -------
        assign_plate_velocities(model_time)
            Assigns plate velocities based on the specified model time.
        ndtime2geotime(ndtime)
            Converts non-dimensionalised time to geologic time.
        geotime2ndtime(geotime)
            Converts geologic time to non-dimensionalised time.

        Examples
        --------
        >>> connector = pyGplatesConnector(rotation_filenames, topology_filenames, geologic_zero, dbc)
        >>> connector.assign_plate_velocities(model_time=100)
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

        # time window for velocity interpolations
        # we round times to the first floating point
        # because higher precision in timing does not mean much in plate reconstruction models
        self.delta_time = delta_time

        # Assiging the DirichletBC object from firedrake
        self.dbc = dbc

        # store the coordinates of the function that need to be assigned
        # these are basically the mesh nodes on a sphere
        self.boundary_coords = fd.Function(
            dbc.function_space(),
            name="coordinates").interpolate(
                fd.SpatialCoordinate(
                    extract_unique_domain(dbc.function_arg))).dat.data_ro_with_halos[self.dbc.nodes]
        self.boundary_coords /= np.linalg.norm(self.boundary_coords, axis=1)[:, np.newaxis]

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
    def assign_plate_velocities(self, model_time):
        """
        Assigns plate velocities to the boundary condition function based on a specified
        model time, integrating plate tectonic reconstructions from pyGplates. This method
        adjusts the boundary condition of the simulation according to the
        velocities derived from plate tectonic models.

        The method calculates new plate velocities if the specified model time differs
        significantly (greater than delta_time) from the last reconstruction time. This
        ensures that the boundary conditions are updated only when necessary. It uses
        the cKDTree data structure for efficient nearest neighbor searches to interpolate
        velocities onto the mesh nodes.

        Parameters
        ----------
        model_time : float
            The model time for which plate velocities are to be calculated and assigned.
            This time is converted to geological time and used to extract relevant
            plate motions from the pyGplates model.

        Raises
        ------
        Exception
            If the requested reconstruction time is negative, indicating an issue with
            the plate reconstruction model or time conversion.

        Notes
        -----
        - The method uses non-dimensionalised time to geologic time conversions.
        - It assumes the presence of a valid Firedrake DirichletBC (dbc) object and
          pyGplates rotation and topology models within the class.
        - Velocities are non-dimensionalised and scaled for the simulation before
          being applied.
        """
        # stretch the dimensionalised time by plate_scaling_factor
        requested_reconstruction_time = self.ndtime2geotime(ndtime=model_time)

        # Raising an error if the user is asking for invalid time
        if requested_reconstruction_time < 0:
            raise Exception(
                ("pyGplates: requested geologic time is negative!"
                 f"maximum: {self.ndtime2geotime(0.0)}")
            )

        # Only calculate new velocities if, either it's the first time step,
        # or it has been longer than {delta_time} since last calculation
        # The boundary condition gets updated here
        if self.reconstruction_time is None or abs(requested_reconstruction_time - self.reconstruction_time) > self.delta_time:
            self.reconstruction_time = requested_reconstruction_time
            interpolated_u = self._interpolatae_seeds_u()
            self.dbc.function_arg.dat.data_with_halos[self.dbc.nodes] = interpolated_u
            log(f"pyGplates: Calculated surface velocities for {self.reconstruction_time} Ma.")

    def ndtime2geotime(self, ndtime):
        """ converts non-dimensised time to geologic time with respect to present-day (Ma)

        Args:
            ndtime (float): non-dimensionalise time

        Returns:
            float: geologic time
        """
        geotime = self.geologic_zero - float(ndtime) * time_dim_factor/myrs2sec/self.scaling_factor
        return geotime

    def geotime2ndtime(self, geotime):
        """ converts geologic time with respect to present-day (Ma) to non-dimensionalise time:

        Args:
            geotime (float): geologic time (before presentday in Myrs)

        Returns:
            float: non-dimensionalised time
         """
        ndtime = (self.geologic_zero - geotime)*(myrs2sec*self.scaling_factor / time_dim_factor)
        return ndtime

    # convert seeds to Gplate features
    def _make_GPML_velocity_feature(self, coords):
        """ function to make a velocity mesh nodes at an arbitrary set of points defined in
             coords[# of points, 3] = x, y, z"""

        # Add points to a multipoint geometry
        multi_point = pygplates.MultiPointOnSphere(
            [pygplates.PointOnSphere(x=coords[i, 0], y=coords[i, 1], z=coords[i, 2], normalise=True)
             for i in range(numpy.shape(coords)[0])]
        )

        # Create a feature containing the multipoint feature, and defined as MeshNode type
        meshnode_feature = pygplates.Feature(pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
        meshnode_feature.set_geometry(multi_point)
        meshnode_feature.set_name('Velocity Mesh Nodes from pygplates')

        output_feature_collection = pygplates.FeatureCollection(meshnode_feature)

        return output_feature_collection

    def _interpolatae_seeds_u(self):
        # calculate velocities here
        seeds_u = self._calc_velocities(
            velocity_domain_features=self.velocity_domain_features,
            topology_features=self.topology_features,
            rotation_model=self.rotation_model,
            time=round(self.reconstruction_time, 0),
            delta_time=pyGplatesConnector.delta_t)

        seeds_u = numpy.array([i.to_xyz() for i in seeds_u]) *\
            ((1e-2 * velocity_non_dim_factor) / (self.scaling_factor * yrs2sec))

        # generate a KD-tree of the seeds points that have a numerical value
        tree = cKDTree(data=self.seeds[seeds_u[:, 0] == seeds_u[:, 0], :], leafsize=16)

        # find the neighboring points
        dists, idx = tree.query(x=self.boundary_coords, k=self.nneighbours)

        # weighted average (by 1/distance) of the data
        res_u = numpy.einsum(
            'i, ij->ij',
            1/numpy.sum(1/dists, axis=1),
            numpy.einsum('ij, ijk ->ik', 1/dists, seeds_u[seeds_u[:, 0] == seeds_u[:, 0]][idx])
        )
        # if too close assign the value of the nearest point
        res_u[dists[:, 0] <= 1e-8, :] = seeds_u[seeds_u[:, 0] == seeds_u[:, 0]][idx[dists[:, 0] <= 1e-8, 0]]
        return res_u

    def _calc_velocities(self, velocity_domain_features, topology_features, rotation_model, time, delta_time):
        # All domain points and associated (magnitude, azimuth, inclination) velocities for the current time.
        all_domain_points = []
        all_velocities = []

        # Partition our velocity domain features into our topological plate polygons at the current 'time'.
        plate_partitioner = pygplates.PlatePartitioner(topology_features, rotation_model, time)

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
                        equivalent_stage_rotation = rotation_model.get_rotation(time, partitioning_plate_id, time + delta_time)

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
                        all_velocities.extend([pygplates.Vector3D(numpy.NaN, numpy.NaN, numpy.NaN)])

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

        phi = numpy.pi * (3. - numpy.sqrt(5.))  # golden angle in radians

        y = 1 - (numpy.array(list(range(samples)))/(samples-1)) * 2
        radius = numpy.sqrt(1 - y * y)
        theta = phi * numpy.array(list(range(samples)))
        x = numpy.cos(theta) * radius
        z = numpy.sin(theta) * radius
        return numpy.array([[x[i], y[i], z[i]] for i in range(len(x))])


def validate_topology_rotation(topology_filenames, rotation_filenames, test_times):
    """ Given the collection of topology filenames and rotation_filenames
        This function tests the plate reconstruction model for the given
        time window of test_times

        The main cause of issue is when a point on the unit sphere cannot be
        allocated to a plate id.

    Args:
        topology_filenames (list of strings): topology file names
        rotation_filenames (list of strings): rotation file names
        test_times (range of geologic times): it is a list/array of geologic times to be tested

    Raises:
        Exception: _description_
    """
    Xnodes = np.arange(-180, 180, 2)
    Ynodes = np.arange(-90, 90, 2)
    Xg_x, Yg_x = np.meshgrid(Xnodes, Ynodes)

    # Add points to a multipoint geometry
    multi_point = pygplates.MultiPointOnSphere(
        [(float(lat), float(lon)) for lat, lon in zip(Yg_x.flatten(), Xg_x.flatten())])

    # Create a feature containing the multipoint feature,
    # and defined as MeshNode type
    meshnode_feature = pygplates.Feature(
        pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
    meshnode_feature.set_geometry(multi_point)
    meshnode_feature.set_name('Velocity Mesh Nodes from pygplates')

    velocity_domain_features = pygplates.FeatureCollection(meshnode_feature)

    # Load one or more rotation files into a rotation model.
    rotation_model = pygplates.RotationModel(rotation_filenames)

    # Load the topological plate polygon features.
    topology_features = []
    for fname in topology_filenames:
        for f in pygplates.FeatureCollection(fname):
            topology_features.append(f)

    # All domain points and associated (magnitude, azimuth, inclination)
    # velocities for the current reconstruction_time.
    all_domain_points = []

    for reconstruction_time in test_times:
        # Partition our velocity domain features into our topological plate
        # polygons at the current 'reconstruction_time'.
        plate_partitioner = pygplates.PlatePartitioner(
            topology_features, rotation_model, reconstruction_time)

        for velocity_domain_feature in velocity_domain_features:

            # A velocity domain feature usually has a single geometry
            # but we'll assume it can be any number.
            # Iterate over them all.
            for velocity_domain_geometry in (
                    velocity_domain_feature.get_geometries()):

                for velocity_domain_point in velocity_domain_geometry.get_points():
                    all_domain_points.append(velocity_domain_point)
                    partitioning_plate = (
                        plate_partitioner.partition_point(velocity_domain_point))
                    if partitioning_plate is None:
                        raise Exception(
                            f"At {reconstruction_time} Ma no plate was"
                            f" found for {velocity_domain_point.to_lat_lon()}")
        log(f"Age: {reconstruction_time} is robust.")
