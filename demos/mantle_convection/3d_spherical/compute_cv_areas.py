"""Load an extruded spherical mesh and write its layerwise nodal areas."""

from pathlib import Path
from sys import argv

from gadopt import CheckpointFile, VTKFile, layerwise_nodal_control_volumes


checkpoint_path = (
    Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("Final_State.h5")
)
with CheckpointFile(str(checkpoint_path), "r") as checkpoint:
    mesh = checkpoint.load_mesh("firedrake_default_extruded")

cv_area_3d = layerwise_nodal_control_volumes(mesh, name="cv_area_3d")
VTKFile("cv_area_3d.pvd").write(cv_area_3d)
