from gadopt import CheckpointFile, FunctionSpace, Function, VTKFile, LayerAveraging
from pathlib import Path

# Getting all the files with the name callback
allfiles = Path("./").glob("callback_*.h5")
# Sort them
all_callback_files = sorted(
    [i for i in Path("./").glob("callback*")],
    key=lambda x: x.stem.split("_"[-1])
)
# Get the mesh for now
with CheckpointFile(all_callback_files[0].as_posix(), mode="r") as fi:
    mesh = fi.load_mesh("firedrake_default_extruded")
# making sure we know the mesh is spherical
mesh.cartesian = False

# Setting up layer averaging
averager = LayerAveraging(mesh, quad_degree=6)

Q = FunctionSpace(mesh, "CG", 1)
control = Function(Q, name="control")
state = Function(Q, name="state")
Tobs = Function(Q, name="Tobs")
Tave = Function(Q, name="Tobs")

#
vtk_fi = VTKFile("callbacks.pvd")

for filename in all_callback_files:
    with CheckpointFile(filename.as_posix(), mode="r") as fi:
        control.interpolate(fi.load_function(mesh, name="control"))
        state.interpolate(fi.load_function(mesh, name="state"))
        Tobs.interpolate(fi.load_function(mesh, name="observation"))
        averager.extrapolate_layer_average(Tave, averager.get_layer_average(Tobs))

    vtk_fi.write(control, state, Tobs)
