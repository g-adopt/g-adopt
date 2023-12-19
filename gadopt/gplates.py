import numpy as np
import pygplates
import firedrake as fd
import numpy
from gadopt.utility import log

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
    def __init__(self, rotation_filenames, topology_filenames, geologic_zero, dbc, delta_time=1., scaling_factor=1.0):
        """ An interface to pygplates, using which one can update given Dirichlet boundary condition given by dbc.

        Args:
            rotation_filenames (string or list of stringd): collection of rotation file names for pygplates
            topology_filenames (string or list of strings): collection of topology file names for pygplates
            geologic_zero (float): what is the oldest time present in the plate reconstruction model
            dbc (firedrake.bcs.DirichletBC): contains the information for boundary condition
            delta_time (float, optional): what is the window range out of which plate velocities will be updated Defaults to 1..
            scaling_factor (float, optional): Scaling factor for surface velocities. Defaults to 1.0.
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
        self.delta_time = delta_time

        # Assiging the DirichletBC object from firedrake
        self.dbc = dbc

        # velocity domain features hold values for each coordinate
        CoordFunction = fd.Function(
            dbc.function_space(),
            name="coordinates").interpolate(
                fd.SpatialCoordinate(
                    dbc.function_space().ufl_domain()
                )
        )
        self.velocity_domain_features = (
            self._make_GPML_velocity_feature(
                CoordFunction.dat.data_ro_with_halos[self.dbc.nodes]
            )
        )

        # the boundary condition to be updated
        self.boundary_condition = dbc.function_arg

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
        # stretch the dimensionalised time by plate_scaling_factor
        requested_reconstruction_time = self.ndtime2geotime(ndtime=model_time)

        if requested_reconstruction_time < 0:
            raise Exception(
                ("pyGplates: geologic time is being negative!"
                 f"maximum: {self.geologic_zero/(time_dim_factor/myrs2sec/self.scaling_factor)}")
            )

        log(f"pyGplates: Time {requested_reconstruction_time}.")

        # only calculate new velocities if, either it's the first time step, or there has been more than delta_time since last calculation
        # velocities are stored in cache
        if not self.reconstruction_time or abs(requested_reconstruction_time - self.reconstruction_time) > self.delta_time:
            self.reconstruction_time = requested_reconstruction_time
            velocities = self._obtain_velocities(reconstruction_time=self.reconstruction_time)
        self.boundary_condition.dat.data_with_halos[self.dbc.nodes] = velocities

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

    # computing the velocities for the seed points
    def _obtain_velocities(self, reconstruction_time):
        # calculate velocities here
        all_velocities = self._calc_velocities(
            velocity_domain_features=self.velocity_domain_features,
            topology_features=self.topology_features,
            rotation_model=self.rotation_model,
            time=reconstruction_time,
            delta_time=self.delta_time)

        return (numpy.array([i.to_xyz() for i in all_velocities]) *
                ((1e-2 * velocity_non_dim_factor) / (self.scaling_factor * yrs2sec)))

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
                        raise Exception("No plate id found. There is an issue with the reconstruciton model.")
                        all_velocities.extend([pygplates.Vector3D(numpy.NaN, numpy.NaN, numpy.NaN)])

        return all_velocities


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
