from firedrake import *
import numpy as np
from pykdtree.kdtree import KDTree
from pyshtools import SHCoeffs
from tfinterpy.idw import IDW

# Quadrature degree:
dx = dx(degree=6)


def map_tomography_model():
    LLNL_model = seismic_model(
        fi_name=("./NEW_TEMP_lin_LLNLG3G_SLB_Q5_mt512_smooth_2.0_101.sph")
    )

    # Set up geometry:
    rmin, rmax, ref_level, nlayers = 1.22, 2.22, 7, 64

    # Variable radial resolution
    # Initiating layer heights with 1.
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

    # Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=(rmax - rmin) * resolution_func / np.sum(resolution_func),
        extrusion_type="radial",
    )

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.
    Q1 = FunctionSpace(mesh, "CG", 1)
    Qlayer = FunctionSpace(mesh2d, "CG", 1)
    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.

    Told = Function(Q, name="NewTemp")
    Tlayer = Function(Qlayer, name="LayerAverage")

    # Initialise from checkpoint:
    tic_dc = DumbCheckpoint("Temperature_State_0", mode=FILE_READ)
    tic_dc.load(Told, name="Temperature")
    tic_dc.close()

    # Initial condition for Stokes:
    uic_dc = DumbCheckpoint("Stokes_State_0", mode=FILE_READ)
    uic_dc.load(z, name="Stokes")
    uic_dc.close()

    r = Function(V, name="SpatialCoordinates").interpolate(SpatialCoordinate(mesh))
    rmag = Function(Q, name="Rmag").interpolate(sqrt(SpatialCoordinate(mesh) ** 2))
    ref_temperature = Function(Q, name="ReferenceTemperature")
    ref_temperatureQ1 = Function(Q1, name="ReferenceTemperature")
    average_temperature = Function(Q1, name="AverageTemperature")

    vnodes = nlayers * 2 + 1

    rads = np.array([np.average(rmag.dat.data[i::vnodes]) for i in range(vnodes)])
    LLNL_model.load_seismic_data(rads=rads)
    LLNL_model.setup_mesh(r.dat.data[0::vnodes])

    #
    for i in range(vnodes):
        ref_temperature.dat.data[i::vnodes] = LLNL_model.fill_layer(i)

    ref_temperatureQ1.project(
        ref_temperature,
        solver_parameters={
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "pc_type": "sor",
            "mat_type": "aij",
            "ksp_rtol": 1e-12,
        },
    )

    Rmin_area = assemble(Constant(1.0, domain=mesh2d) * dx)  # area of CMB

    def layer_average(T):
        vnodes = nlayers + 1
        hnodes = Qlayer.dim()  # n/o Q2 nodes in each horizontal layer
        assert hnodes * vnodes == Q1.dim()
        for i in range(vnodes):
            Tlayer.dat.data[:] = T.dat.data_ro[i::vnodes]
            # NOTE: this integral is performed on mesh2d, which always has r=Rmin, but we normalize
            average_temperature.dat.data[i::vnodes] = assemble(Tlayer * dx) / Rmin_area
        return average_temperature

    with CheckpointFile("Adjoint_CheckpointFile.h5", mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(Told, name="Temperature")
        fi.save_function(z, name="Stokes")
        fi.save_function(ref_temperatureQ1, name="ReferenceTemperature")
        fi.save_function(layer_average(ref_temperatureQ1), name="AverageTemperature")

    File("./output/Output.pvd").write(Told, ref_temperature, average_temperature)


def make_sample_mesh():
    # Set up geometry:
    rmin, rmax, ref_level, nlayers = 1.22, 2.22, 4, 16
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

    with CheckpointFile("mesh.h5", "w") as fi:
        fi.save_mesh(mesh=mesh)


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
            raise Exception(f'Error! Expect "# Spherical Harmonics". Got: {line}')
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
            sph_all[ir, :, :, :] = clm.coeffs[:, : lmax_calc + 1, : lmax_calc + 1]
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
        rshl, sph = rcf2sphfile_array_pyshtools(fname=self.fi_name, lmax_calc=None)

        rshl = np.array([1 / 2890e3 * r + (2.22 - 1 / 2890e3 * 6370e3) for r in rshl])
        tree = KDTree(rshl[:, np.newaxis])

        dists, inds = tree.query(np.asarray(self.rads)[:, np.newaxis], k=k)

        self.sph = np.einsum(
            "i, iklm->iklm",
            1 / np.sum(1 / dists, axis=1),
            np.einsum("ij, ijklm->iklm", 1 / dists, sph[inds]),
        )
        self.sph[dists[:, 0] < 1e-6] = sph[inds[dists[:, 0] < 1e-6, 0]]

        self.lmax = self.sph[0].shape[1] - 1

    def setup_mesh(self, coords):
        self.target_mesh = coords


if __name__ == "__main__":
    map_tomography_model()
