from gadopt import *
from gadopt.gplates import pyGplatesConnector

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
u = Function(V, name="velocity")
dbc = DirichletBC(V, u, "top")

# compute surface velocities
rec_model = pyGplatesConnector(
    rotation_filenames=[
        './gplates_files/Zahirovic2022_CombinedRotations_fixed_crossovers.rot',
    ],
    topology_filenames=[
        './gplates_files/Zahirovic2022_PlateBoundaries.gpmlz',
        './gplates_files/Zahirovic2022_ActiveDeformation.gpmlz',
        './gplates_files/Zahirovic2022_InactiveDeformation.gpmlz',
    ],
    dbc=dbc,
    geologic_zero=409,
    delta_time=1.0
)

plog = ParameterLog('plt_rec_test.log', mesh)
plog.log_str("geologic time, u_rms")

for t in np.arange(409, 0, -50):
    rec_model.assign_plate_velocities(rec_model.geotime2ndtime(t))
    plog.log_str(f"{t}, {sqrt(assemble(inner(u, u) * ds_t))}")

plog.close()
