from gadopt import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector

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
plate_reconstruction_model = pyGplatesConnector(
    rotation_filenames=[
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/optimisation/1000_0_rotfile_MantleOptimised.rot"
    ],
    topology_filenames=[
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/250-0_plate_boundaries.gpml",
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/410-250_plate_boundaries.gpml",
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Convergence.gpml",
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Divergence.gpml",
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Topologies.gpml",
        "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Transforms.gpml",
    ],
    nneighbours=4,
    nseeds=1000,
    oldest_age=10000,
    delta_t=1.0
)

# Top velocity boundary condition
gplates_velocities = GplatesFunction(
    V,
    gplates_connector=plate_reconstruction_model,
    top_boundary_marker="top",
    name="GPlates_Velocity"
)

myfile = VTKFile("Velocities.pvd")

for t in np.arange(1000, 0, -50):
    gplates_velocities.update_plate_reconstruction(plate_reconstruction_model.age2ndtime(t))
    gplates_velocities.assign(gplates_velocities / pyGplatesConnector.velocity_dimDcmyr)
    myfile.write(gplates_velocities)
