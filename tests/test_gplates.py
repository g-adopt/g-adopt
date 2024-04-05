import pickle
from gadopt import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector


def test_gplates():
    # Set up geometry:
    rmin, rmax, ref_level, nlayers = 1.22, 2.22, 5, 16

    # Construct a CubedSphere mesh and then extrude into a sphere
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=(rmax - rmin)/(nlayers-1),
        extrusion_type="radial",
    )

    V = VectorFunctionSpace(mesh, "CG", 2)

    # compute surface velocities
    rec_model = pyGplatesConnector(
        rotation_filenames=[
            '../demos/gplates_global/gplates_files/Zahirovic2022_CombinedRotations_fixed_crossovers.rot',
        ],
        topology_filenames=[
            '../demos/gplates_global/gplates_files/Zahirovic2022_PlateBoundaries.gpmlz',
            '../demos/gplates_global/gplates_files/Zahirovic2022_ActiveDeformation.gpmlz',
            '../demos/gplates_global/gplates_files/Zahirovic2022_InactiveDeformation.gpmlz',
        ],
        nseeds=1e5,
        nneighbours=4,
        geologic_zero=409,
        delta_time=1.0
    )

    gplates_function = GplatesFunction(V, gplates_connector=rec_model, top_boundary_marker="top")

    surface_rms = []

    for t in np.arange(409, 0, -50):
        gplates_function.update_plate_reconstruction(rec_model.geotime2ndtime(t))
        surface_rms.append(sqrt(assemble(inner(gplates_function, gplates_function) * ds_t)))

    # Loading reference plate velocities
    with open('test_gplates.pkl', 'rb') as file:
        ref_surface_rms = pickle.load(file)

    np.testing.assert_allclose(surface_rms, ref_surface_rms)
