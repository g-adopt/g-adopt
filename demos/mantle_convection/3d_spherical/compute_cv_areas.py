"""Write layerwise nodal areas for the checkpoint used by the original script."""

from gadopt import CheckpointFile, VTKFile, layerwise_nodal_control_volumes


with CheckpointFile("initial_condition_mat_prop/Final_State.h5", "r") as checkpoint:
    mesh = checkpoint.load_mesh("firedrake_default_extruded")

cv_area_3d = layerwise_nodal_control_volumes(mesh, name="cv_area_3d")
VTKFile("cv_area_3d.pvd").write(cv_area_3d)
