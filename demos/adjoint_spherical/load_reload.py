from gadopt import *
import numpy as np
from pykdtree.kdtree import KDTree
from pyshtools import SHCoeffs
from tfinterpy.idw import IDW

# Quadrature degree:
dx = dx(degree=6)

# Global parameters that will be used by all
rmin = 1.22
rmax = 2.22
ref_level = 7
nlayers = 64


def __main__():
    # generate and write out mesh
    mesh_finame = generate_spherical_mesh("spherical_mesh.h5")

    # load the mesh
    with CheckpointFile(mesh_finame, mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    # Reference temperature field
    Tobs = load_tomography_model(
        mesh,
        fi_name="./NEW_TEMP_lin_LLNLG3G_SLB_Q5_mt512_smooth_2.0_101.sph")

    # Average of temperature field
    Taverage = Function(Tobs.function_space(), name="Taverage")

    # Calculate the layer average of the initial state
    averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
    averager.extrapolate_layer_average(
        Taverage, averager.get_layer_average(Tobs))

    # setting up the viscosity function
    mu = viscosity_function(mesh)

    # Write out the file
    with CheckpointFile("linear_LLNLG3G_SLB_Q5_smooth_2.0_101.h5", mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(Tobs, name="Tobs")
        fi.save_function(Taverage, name="AverageTemperature")
        fi.save_function(mu, name="Viscosity")

    # Output for visualisation
    output = File("./linear_LLNL.pvd")
    output.write(Tobs, Taverage, mu)


def viscosity_function(mesh):
    # Set up function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)

    # tomography based temperature
    viscosity = Function(Q, name="Viscosity")

    # radius and coordinates
    r = Function(V).interpolate(SpatialCoordinate(mesh))
    rad = Function(Q).interpolate(sqrt(SpatialCoordinate(mesh) ** 2))

    # knowing how many extrusion layers we have
    vnodes = nlayers * 2 + 1
    rad_profile = np.array([np.average(rad.dat.data[i::vnodes])
                           for i in range(vnodes)])

    terra_mu = np.loadtxt("./ARCHIVE/mu_2_lith.visc_smoothened.rad")
    terra_rad = np.linspace(rmax, rmin, terra_mu.shape[0])
    dists, inds = KDTree(terra_rad).query(rad_profile, k=2)
    mu_1d = np.sum(1/dists * terra_mu[inds], axis=1)/np.sum(1/dists, axis=1)
    mu_1d[dists[:, 0] <= 1e-6] = terra_mu[inds[dists[:, 0] <= 1e-6, 0]]

    averager = LayerAveraging(mesh, r1d=rad_profile, cartesian=False, quad_degree=6)
    averager.extrapolate_layer_average(
        viscosity, mu_1d)

    return viscosity


def load_tomography_model(mesh, fi_name):
    # Loading temperature model
    LLNL_model = seismic_model(fi_name=fi_name)

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)

    # tomography based temperature
    Tobs = Function(Q, name="Tobs")

    # radius and coordinates
    r = Function(V).interpolate(SpatialCoordinate(mesh))
    rad = Function(Q).interpolate(sqrt(SpatialCoordinate(mesh) ** 2))

    # knowing how many extrusion layers we have
    vnodes = nlayers + 1
    rad_profile = np.array([np.average(rad.dat.data[i::vnodes])
                           for i in range(vnodes)])

    # load seismic tomogrpahy based temperature model
    LLNL_model.load_seismic_data(rads=rad_profile)
    LLNL_model.setup_mesh(r.dat.data[0::vnodes])

    # Assigning values to each layer
    for i in range(vnodes):
        Tobs.dat.data[i::vnodes] = LLNL_model.fill_layer(i)

    return Tobs


def generate_spherical_mesh(mesh_filename):
    # Set up geometry:

    resolution_func = np.ones((nlayers))

    # A gaussian shaped function
    def gaussian(center, c, a):
        return a * np.exp(
            -((np.linspace(rmin, rmax, nlayers) - center) ** 2) / (2 * c**2)
        )

    # building the resolution function
    for idx, r_0 in enumerate([rmin, rmax, rmax - 660 / 6370]):
        # gaussian radius
        c = 0.15
        # how different is the high res area from low res
        res_amplifier = 5.0
        resolution_func *= 1 / (1 + gaussian(center=r_0, c=c, a=res_amplifier))

    resolution_func *= 1.0 / np.sum(resolution_func)
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)

    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=resolution_func,
        extrusion_type="radial",
    )

    with CheckpointFile(mesh_filename, "w") as fi:
        fi.save_mesh(mesh=mesh)

    return mesh_filename


def rcf2sphfile_array_pyshtools(fname, lmax_calc=None):
    """
    The same as rcf2sphfile_pyshtools, but converting the
    output already to an array
    """
    with open(fname, mode="r") as f_id:
        line = f_id.readline()
        lmax = int(line.split(",")[0].split()[-1])
        nr = int(line.split(",")[-1].split()[1]) + 1
        lmax_calc = lmax if lmax_calc is None else lmax_calc
        sph_all = np.zeros((nr, 2, lmax_calc + 1, lmax_calc + 1))

        # Read in the comment for the name of the array
        line = f_id.readline()
        # Read in the radius
        line = f_id.readline()
        rshl = np.zeros(nr)
        for ir in range(nr):
            rshl[ir] = float(f_id.readline())
        # Read in the averages
        line = f_id.readline()
        if line != "# Averages\n":
            raise Exception(f'Error! Expect "# Averages". Got: {line}')
        for ir in range(nr):
            f_id.readline()
        line = f_id.readline()
        if line != "# Spherical Harmonics\n":
            raise Exception(
                f'Error! Expect "# Spherical Harmonics". Got: {line}')
        for ir in range(nr):
            line = f_id.readline()
            if line[0] != str("#"):
                raise Exception(f'Error! Expect "# Comment". Got: {line}')
            clm = SHCoeffs.from_zeros(lmax=lmax, normalization="ortho")
            for l in range(lmax + 1):
                for m in range(l + 1):
                    line = f_id.readline()
                    if m == 0:
                        clm.set_coeffs(float(line.split()[0]), l, 0)
                    else:
                        clm.set_coeffs(
                            [float(line.split()[0]), float(line.split()[1])],
                            [l, l],
                            [m, -m],
                        )
            sph_all[ir, :, :, :] = clm.coeffs[:,
                                              : lmax_calc + 1, : lmax_calc + 1]
    return rshl, sph_all


def cartesian_to_lonlat(x, y, z):
    """
    Convert Cartesian coordinates to longitude and latitude.

    Parameters:
    x, y, z : array-like
        Cartesian coordinates

    Returns:
    lon, lat : array-like
        Longitude and latitude in radians
    """

    lon = np.arctan2(y, x) * 180.0 / np.pi
    lat = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180.0 / np.pi

    return lon, lat


def lonlatto_cartesian(lat, lon, radius=1.0):
    """
    Convert latitude and longitude to 3D Cartesian coordinates.

    Parameters:
    - lat: latitude in degrees
    - lon: longitude in degrees
    - radius: radius of the Earth (default is 6371.0 km)

    Returns:
    - x, y, z: Cartesian coordinates
    """

    # Convert latitude and longitude from degrees to radians
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0

    # Calculate Cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    return x, y, z


class seismic_model(object):
    def __init__(self, fi_name):
        self.fi_name = fi_name
        self.coords = self.fibonacci_sphere(360 * 180)
        self.lon, self.lat = cartesian_to_lonlat(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        )

    def fill_layer(self, layer_index):
        data = SHCoeffs.from_array(self.sph[layer_index], normalization="4pi").expand(
            lon=self.lon, lat=self.lat
        )
        data = 1 / 3900.0 * data - 3.0 / 39.0
        return IDW(np.column_stack((self.coords, data)), mode="3d").execute(
            self.target_mesh
        )

    def fibonacci_sphere(self, samples):
        """
        Generating equidistancial points on a sphere
        Fibannoci_spheres
        """

        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        y = 1 - (np.array(list(range(samples))) / (samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * np.array(list(range(samples)))
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.column_stack((x, y, z))

    def load_seismic_data(self, rads, k=2):
        self.rads = rads
        rshl, sph = rcf2sphfile_array_pyshtools(
            fname=self.fi_name, lmax_calc=None)

        rshl = np.array(
            [1 / 2890e3 * r + (2.22 - 1 / 2890e3 * 6370e3) for r in rshl])
        tree = KDTree(rshl[:, np.newaxis])

        epsilon = 1e-10  # A small value to prevent division by zero

        dists, inds = tree.query(np.asarray(self.rads)[:, np.newaxis], k=k)

        # Add epsilon to dists to avoid division by zero
        self.sph = np.einsum(
            "i, iklm->iklm",
            1 / np.sum(1 / (dists + epsilon), axis=1),  # Add epsilon here
            np.einsum("ij, ijklm->iklm", 1 / (dists + epsilon), sph[inds]),  # And here
        )

        # Handle the case for very small distances separately
        self.sph[dists[:, 0] < 1e-6] = sph[inds[dists[:, 0] < 1e-6, 0]]

        self.lmax = self.sph[0].shape[1] - 1

    def setup_mesh(self, coords):
        self.target_mesh = coords


if __name__ == "__main__":
    __main__()
