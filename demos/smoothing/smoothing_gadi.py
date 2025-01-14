from gadopt import *
from pathlib import Path


def __main__():
    # get the path to the base directory
    base_path = Path(__file__).resolve().parent

    # mesh/initial guess file is comming from a long-term simulation
    mesh_path = base_path / "../adjoint_spherical/initial_condition_mat_prop/Final_State.h5"

    # Load mesh from checkpoint
    with CheckpointFile(str(mesh_path), mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        T_simulation = f.load_function(mesh, name="Temperature")

    Q_CG = FunctionSpace(mesh, "CG", 1)
    T_simulation_GC = Function(Q_CG, name="Temperature")
    T_simulation_GC.interpolate(T_simulation)

    temp_bcs = {
        "bottom": {'T': 1.0 - 930. / 3700},
        "top": {'T': 0.0},
    }

    # Projection solver parameters for nullspaces:
    iterative_energy_solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_rtol": 1e-5,
        "pc_type": "sor",
    }

    # Adding Smoothing to Tobs
    smoother = DiffusiveSmoothingSolver(
        function_space=T_simulation_GC.function_space(),
        wavelength=0.05,
        bcs=temp_bcs,
        solver_parameters=iterative_energy_solver_parameters,
    )

    # acting smoothing on Tobs
    T_smooth = smoother.action(T_simulation_GC)

    output_path = base_path / "smoothed_temperature.h5"

    # Output for visualisation
    output = VTKFile(output_path.with_suffix(".pvd"))
    output.write(T_simulation, T_smooth)


if __name__ == "__main__":
    __main__()
