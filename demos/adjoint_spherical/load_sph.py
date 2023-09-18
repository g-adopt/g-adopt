from firedrake import *
import numpy as np
from pykdtree.kdtree import KDTree
from pyshtools import SHCoeffs

# Quadrature degree:
dx = dx(degree=6)


def map_tomography_model():
    LLNL_model = seismic_model(
        fi_name=(
            "/Users/sghelichkhani/Workplace/pythbox"
            "/DATA/SEISMIC_CONVERSIONS/"
            "NEW_TEMP_lin_LLNLG3G_SLB_Q5_mt512_smooth_2.0_101.sph"
        )
    )

    # make_sample_mesh()

    with CheckpointFile("mesh.h5", mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    V = VectorFunctionSpace(mesh, "CG", 1)
    Q1 = FunctionSpace(mesh, "CG", 1)
    r = Function(V, name="SpatialCoordinates").project(SpatialCoordinate(mesh))
    rmag = Function(Q1, name="Rmag").project(sqrt(SpatialCoordinate(mesh) ** 2))
    ref_temperature = Function(Q1, name="ReferenceTemperature")

    vnodes = 16 + 1

    rads = np.array([np.average(rmag.dat.data[i::vnodes]) for i in range(vnodes)])
    LLNL_model.load_seismic_data(rads=rads)
    LLNL_model.setup_mesh(r.dat.data[0::vnodes])

    #
    for i in range(vnodes):
        ref_temperature.dat.data[i::vnodes] = LLNL_model.fill_layer(i)

    File("Test.pvd").write(ref_temperature)


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
    ###############################################################

    # Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
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
            sys.exit(str('Error! Expect "# Averages". Got: %s' % line))
        for ir in range(nr):
            f_id.readline()
        line = f_id.readline()
        if line != "# Spherical Harmonics\n":
            sys.exit(str('Error! Expect "# Spherical Harmonics". Got: %s' % line))
        for ir in range(nr):
            line = f_id.readline()
            if line[0] != str("#"):
                sys.exit(str('Error! Expect "# Comment". Got: %s' % line))
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


class seismic_model(object):
    def __init__(self, fi_name):
        self.fi_name = fi_name

    def fill_layer(self, layer_index):
        data = (
            SHCoeffs.from_array(self.sph[layer_index], normalization="4pi")
            .expand()
            .data.flatten()
        )
        output = np.einsum("ij, ij->i", 1 / self.dists, data[self.inds]) / np.sum(
            1 / self.dists, axis=1
        )
        output[self.dists[:, 0] < 1e-6] = data[self.inds[self.dists[:, 0] < 1e-6, 0]]
        return output

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

        grid = SHCoeffs.from_zeros(lmax=self.lmax, normalization="ortho").expand()
        lons, lats = grid.lons(), grid.lats()
        lons[lons > 180.0] -= 360.0
        lons_x, lats_x = np.meshgrid(lons, lats)
        self.tree = KDTree(np.column_stack((lons_x.flatten(), lats_x.flatten())))
        self.mesh_lons = None
        self.mesh_lats = None

    def setup_mesh(self, coords, k=3):
        mesh_lons, mesh_lats = cartesian_to_lonlat(
            coords[:, 0], coords[:, 1], coords[:, 2]
        )
        self.dists, self.inds = self.tree.query(
            np.column_stack((mesh_lons, mesh_lats)), k=3
        )


if __name__ == "__main__":
    map_tomography_model()
