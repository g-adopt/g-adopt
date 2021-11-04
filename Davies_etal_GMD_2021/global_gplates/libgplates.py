import itertools
import os.path
import pygplates
import numpy
from firedrake.petsc import PETSc

# get the naming right
data_root = 'gplates_files/'

# List of dynamic plate boundaries
topology_filenames = []
topology_filenames.append(os.path.join(data_root, 'DynamicPolygons/Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml'))

# List of rotation models
rotation_filenames = []
rotation_filenames.append(os.path.join(data_root, 'Rotations/Global_EarthByte_230-0Ma_GK07_AREPS.rot'))

# Make sure rotation & topology files exist
if False in [os.path.isfile(fi) for fi in topology_filenames + rotation_filenames]:
    raise ValueError(
        "Unable to find:",
        list(itertools.compress(
            topology_filenames + rotation_filenames,
            [[not os.path.isfile(fi) for fi in topology_filenames+rotation_filenames]]
        ))
    )

# List of constants:

# Factor to scale plate velocities to RMS velocity of model,
# Notice: I use the definition (RMS_Earth / RMS_Model)
# This is specially used in low-Rayleigh-number simulations
plate_scaling_factor = 1.25
# Factor to non-dimensionalise gplates velocities: d/kappa
velocity_non_dim_factor = 2890e3/1.0e-6
# Factor to dimensionalise model time: d^2/kappa
time_dim_factor = 2890e3**2/1.0e-6
# 1 yr in seconds
yrs2sec = 365*24*60*60
# 1 Myr in seconds
myrs2sec = 1e6*yrs2sec


class pygplate_runner(object):
    def __init__(self, rotation_filenames, topology_filenames, geologic_zero, nseeds=1000, nneighbours=3, delta_time=1.):
        """
        A class for importing surface velocities from pygplates.
        Each time surface velocity points are generated for [nseeds] of equidistantial points at the surface.
        With these points a cKDTree obect is generated each time, which will be used to interpolate onto required grid points.
        Interpolation is a weighted average of [nneighbours] closest points, unless the closest points is closer than ~numerical zero.
            input:
                rotation_filenames: list of rotation files
                topology_filenames: list of dynamic polygon files
                geologic_zero:      what is the oldest geologic time available in this model
                nseeds:             number of seed points to generate
                nneighbours:        when interpolating between the seed points and mesh points, how many neighboring points should be used
                delta_time:         how many million years should it take before we read in plate velocities again (is passed on to pygplates for
                                    time averaging too)
        """

        # Rotation model(s)
        self.rotation_model = pygplates.RotationModel(rotation_filenames)

        if nneighbours < 2:
            raise ValueError('nneighbours should be at least 2')
        else:
            self.nneighbours = nneighbours

        # Topological plate polygon feature(s).
        self.topology_features = []
        for fname in topology_filenames:
            for f in pygplates.FeatureCollection(fname):
                self.topology_features.append(f)
        # geologic_zero is the same as model_time=0
        self.geologic_zero = geologic_zero

        # time window for velocity interpolations
        self.delta_time = delta_time

        # NB: Not sure if we really need to keep seeds at this stage
        self.seeds = self.__fibonacci_sphere(samples=int(nseeds))
        #
        self.velocity_domain_features = self.__make_GPML_velocity_feature(self.seeds)

        # last reconstruction time
        self.reconstruction_time = None

        # Flag to know when to recalculate surface velocities.
        self.recalculation_flg = False

    # setting the time that we are interested in
    def set_time(self, model_time):
        from scipy.spatial import cKDTree

        # Convert model time to age
        # stretch the dimensionalised time by plate_scaling_factor
        requested_reconstruction_time = self.geologic_zero - float(model_time) * time_dim_factor/myrs2sec/plate_scaling_factor

        if requested_reconstruction_time < 0:
            raise ValueError('pyGplates: geologic time is being negative! Terminating the code!')

        PETSc.Sys.Print(f"pyGplates: Time {requested_reconstruction_time}.", flush=True)

        # only calculate new velocities if, either it's the first time step, or there has been more than delta_time since last calculation
        # velocities are stored in cache
        if not self.reconstruction_time or abs(requested_reconstruction_time - self.reconstruction_time) > self.delta_time:
            self.reconstruction_time = requested_reconstruction_time
            self.cache = self.__compute_seeds_v()
            self.tree = cKDTree(data=self.seeds[self.cache[:, 0] == self.cache[:, 0], :], leafsize=16)
            self.cache = self.cache[self.cache[:, 0] == self.cache[:, 0], :]
            self.recalculation_flg = True
            if len(self.cache) == 0:
                PETSc.Sys.Print("pyGplates: [Warning] Cache size is zero", flush=True)
                self.recalculation_flg = False
            PETSc.Sys.Print(f"pyGplates: seeds for stage {self.reconstruction_time} Ma loaded.", flush=True)
        else:
            # we don't need to recalculate velocities any more
            self.recalculation_flg = False

    # main interface where velocities can be retrieved for points: coords
    def get_velocities(self, coords):
        # In case we have to re-interpolate velocities
        if self.recalculation_flg:
            # normalising the input
            coords = numpy.einsum('i, ij -> ij', 1/numpy.sqrt(numpy.sum(coords**2, axis=1)), coords)

            # find the neighboring points
            dists, idx = self.tree.query(x=coords, k=self.nneighbours)

            # weighted average (by 1/distance) of the data
            self.coords_v = numpy.einsum('i, ij->ij', 1/numpy.sum(1/dists, axis=1), numpy.einsum('ij, ijk ->ik', 1/dists, self.cache[idx]))

            # if too close assign the value of the nearest point
            self.coords_v[dists[:, 0] <= 1e-8, :] = self.cache[idx[dists[:, 0] <= 1e-8, 0]]

        # convert velocities from cm/year to scaled non-dimensional
        return self.coords_v * ((1e-2 * velocity_non_dim_factor) / (plate_scaling_factor * yrs2sec))

    # computing the velocities for the seed points
    def __compute_seeds_v(self):
        # calculate velocities here
        all_velocities = self.__calc_velocities(velocity_domain_features=self.velocity_domain_features,
                                                topology_features=self.topology_features,
                                                rotation_model=self.rotation_model,
                                                time=self.reconstruction_time,
                                                delta_time=self.delta_time)

        return numpy.array([i.to_xyz() for i in all_velocities])

    # convert seeds to Gplate features
    def __make_GPML_velocity_feature(self, coords):
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

        # NB: at this point, the feature could be written to a file using
        # output_feature_collection.write('myfilename.gpmlz')

        # for use within the notebook, the velocity domain feature is returned from the function
        return output_feature_collection

    def __calc_velocities(self, velocity_domain_features, topology_features, rotation_model, time, delta_time):
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
                        all_velocities.extend([pygplates.Vector3D(numpy.NaN, numpy.NaN, numpy.NaN)])

        return all_velocities

    def __fibonacci_sphere(self, samples):
        """
            Generating equidistancial points on a sphere
            Fibannoci_spheres
        """

        phi = numpy.pi * (3. - numpy.sqrt(5.))  # golden angle in radians

        y = 1 - (numpy.array(list(range(samples)))/(samples-1)) * 2
        radius = numpy.sqrt(1 - y * y)
        theta = phi * numpy.array(list(range(samples)))
        x = numpy.cos(theta) * radius
        z = numpy.sin(theta) * radius
        return numpy.array([[x[i], y[i], z[i]] for i in range(len(x))])


rec_model = pygplate_runner(
    rotation_filenames=rotation_filenames,
    topology_filenames=topology_filenames,
    geologic_zero=230,
    nseeds=100000,
    nneighbours=4,
    delta_time=1.0
)
