from gadopt import * 
from pathlib import Path


def visualise_callbacks(working_path = "./", output_name="callback.pvd"):
    """
    Visualising the callback functions which produce the control, and Tobs
    From this we generate a vtu file that has dimensional control and Tobs and their averages
    """
    # This is the path to the refence fields
    reference_path = Path(__file__).resolve().parent / "initial_condition_mat_prop/reference_fields.h5"

    # Getting all the files with the name callback
    allfiles = list((Path.cwd() / Path(working_path)).glob("callback_*.h5"))
    
    # Sort them
    allfiles = sorted(
        allfiles, key=lambda x: x.stem.split("_"[-1])
    )

    # Get the mesh for now
    with CheckpointFile(allfiles[0].as_posix(), mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    # making sure we know the mesh is spherical
    mesh.cartesian = False
    
    # Setting up layer averaging
    averager = LayerAveraging(mesh, quad_degree=6)
    
    Q = FunctionSpace(mesh, "CG", 1)
    control = Function(Q, name="Tcontrol") # Control being loaded
    control_dim = Function(Q, name="Tcontrol_dim")  # Dimensional copy of control
    Tobs = Function(Q, name="Tobs")  # Tobs (tomography)
    Tave_obs= Function(Q, name="Tave_obs")  # Tave of the 
   
    # Loading adiabatic reference fields
    tala_parameters_dict = {}

    with CheckpointFile(
        reference_path.as_posix(),
        mode="r",
    ) as fi:
        for key in ["Tbar"]:
            tala_parameters_dict[key] = fi.load_function(mesh, name=key)

    # Write out the field
    vtk_fi = VTKFile(output_name)
    
    for filename in allfiles:
        # Load control and Tobs
        with CheckpointFile(filename.as_posix(), mode="r") as fi:
            control.interpolate(fi.load_function(mesh, name="control"))
            Tobs.interpolate(fi.load_function(mesh, name="Tobs"))

        # dim of control 
        control_dim.interpolate(
            (control + tala_parameters_dict["Tbar"]) * Constant(3700.0) + Constant(300.0)
        )

        # Compute the average for 
        averager.extrapolate_layer_average(Tave_obs, averager.get_layer_average(Tobs))

        # Write out
        vtk_fi.write(control_dim, Tobs, Tave_obs)


def visualise_finalstates(working_path="./", output_name="finalstate.pvd"):
    """
    Visualising the callback functions which produce the control, and Tobs
    From this we generate a vtu file that has dimensional control and Tobs and their averages
    """

    # Getting all the files with the name callback
    allfiles = list((Path.cwd() / Path(working_path)).glob("FinalState_*.h5"))
    
    # Sort them
    allfiles = sorted(
        allfiles, key=lambda x: x.stem.split("_"[-1])
    )

    # Get the mesh for now
    with CheckpointFile(allfiles[0].as_posix(), mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    # making sure we know the mesh is spherical
    mesh.cartesian = False
    
    # Setting up layer averaging
    averager = LayerAveraging(mesh, quad_degree=6)
    
    Q = FunctionSpace(mesh, "CG", 1)
    FullT = Function(Q, name="T_final")  # Tobs (tomography)
    Tave = Function(Q, name="Tave_obs")  # Tave of the 
   
    # Loading adiabatic reference fields
    tala_parameters_dict = {}

    with CheckpointFile(
        reference_path.as_posix(),
        mode="r",
    ) as fi:
        for key in ["Tbar"]:
            tala_parameters_dict[key] = fi.load_function(mesh, name=key)

    # Write out the field
    vtk_fi = VTKFile(output_name)
    
    for filename in allfiles:
        # Load control and Tobs
        with CheckpointFile(filename.as_posix(), mode="r") as fi:
            control.interpolate(fi.load_function(mesh, name="control"))
            Tobs.interpolate(fi.load_function(mesh, name="Tobs"))

        # dim of control 
        control_dim.interpolate(
            (control + tala_parameters_dict["Tbar"]) * Constant(3700.0) + Constant(300.0)
        )

        # Compute the average for 
        averager.extrapolate_layer_average(Tave_obs, averager.get_layer_average(Tobs))

        # Write out
        vtk_fi.write(control_dim, Tobs, Tave_obs)




if __name__ == "__main__":
    visualise_callbacks()
