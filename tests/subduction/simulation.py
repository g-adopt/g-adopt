from mpi4py import MPI
from ufl.core.operator import Operator

from gadopt import *

import parameters as prms
from field_initialisation import initial_level_set, initial_temperature
from rheology import material_viscosity
from utility import garbage_collect, interpolate_fields, write_output


def set_up(
    mesh: MeshGeometry, time_step: Constant
) -> tuple[
    TimestepAdaptor,
    StokesSolver,
    EnergySolver,
    LevelSetSolver,
    dict[Function, Operator],
]:
    # Mesh
    mesh.cartesian = True
    boundary = get_boundary_ids(mesh)

    # Function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    if prms.free_surface:
        Z = MixedFunctionSpace([V, W, W])
    else:
        Z = MixedFunctionSpace([V, W])
    Q = FunctionSpace(mesh, "DG", 2, variant="equispaced")
    K = FunctionSpace(mesh, "DG", 2, variant="equispaced")

    # Functions
    stokes = Function(Z, name="Stokes")
    u_func, p_func = stokes.subfunctions[:2]
    u_func.rename("Velocity")
    p_func.rename("Pressure")
    if prms.free_surface:
        eta_func = stokes.subfunctions[2]
        eta_func.rename("Free surface")
    u = split(stokes)[0]
    T = Function(Q, name="Temperature")
    psi = Function(K, name="Level set")

    # Rheology
    mu_material, rheol_expr = material_viscosity(mesh, u, T, psi)

    field_expr = {}
    for name, expr in rheol_expr.items():
        field_expr[Function(W, name=name)] = expr

    # Boundary conditions
    temp_bcs = {
        boundary.bottom: {"T": prms.temperature_scaling(prms.T_pot)},
        boundary.top: {"T": prms.temperature_scaling(prms.T_surf)},
    }
    stokes_bcs = {
        boundary.bottom: {"uy": 0.0},
        boundary.left: {"ux": 0.0},
        boundary.right: {"ux": 0.0},
    }

    if prms.dimensionless:
        compositional_rayleigh = Function(W, name="Rayleigh number (compositional)")
        RaB_material = material_field(psi, [0.0, prms.Ra * prms.B], interface="sharp")
        field_expr[compositional_rayleigh] = RaB_material

        approximation_params = {
            "alpha": 1.0,
            "g": 1.0,
            "H": 0.0,
            "kappa": 1.0,
            "mu": mu_material,
            "Ra": prms.Ra,
            "RaB": RaB_material,
            "rho": 1.0,
            "delta_rho": 1.0,
            "T0": 0.0,
        }

        if prms.free_surface:
            RaFS_material = material_field(
                psi, [prms.Ra * B for B in prms.BFS], interface="sharp"
            )
            stokes_bcs[boundary.top] = {"free_surface": {"RaFS": RaFS_material}}
        else:
            stokes_bcs[boundary.top] = {"uy": 0.0}
    else:
        density = Function(W, name="Density")
        delta_rho_material = material_field(
            psi, [0.0, prms.rho_weak_layer - prms.rho_mantle], interface="sharp"
        )
        rho_material = (prms.rho_mantle + delta_rho_material) * (
            1 - prms.alpha * (T - prms.T_surf)
        )
        field_expr[density] = rho_material

        approximation_params = {
            "alpha": prms.alpha,
            "g": prms.g,
            "H": 0.0,
            "kappa": prms.kappa,
            "mu": mu_material,
            "Ra": 1.0,
            "RaB": 1.0,
            "rho": prms.rho_mantle,
            "delta_rho": delta_rho_material,
            "T0": prms.T_surf,
        }

        if prms.free_surface:
            delta_rho_fs = prms.rho_mantle + delta_rho_material - prms.rho_water
            stokes_bcs[boundary.top] = {"free_surface": {"delta_rho_fs": delta_rho_fs}}
        else:
            stokes_bcs[boundary.top] = {"uy": 0.0}

    if prms.free_surface:
        nullspace_args = {}
    else:
        stokes_nullspace = create_stokes_nullspace(Z)
        nullspace_args = {
            "nullspace": stokes_nullspace,
            "transpose_nullspace": stokes_nullspace,
        }

    # Adaptive time stepping
    tstep_adapt = TimestepAdaptor(
        time_step,
        u,
        V,
        target_cfl=0.55 * prms.subcycles,
        maximum_timestep=prms.myr_to_seconds / prms.time_scale,
    )

    # Solvers
    approximation = BoussinesqApproximation(**approximation_params)

    stokes_solver = StokesSolver(
        stokes,
        approximation,
        T,
        dt=time_step,
        bcs=stokes_bcs,
        solver_parameters="direct",
        solver_parameters_extra={
            "snes_linesearch_type": "bisection",
            "snes_max_it": 200,
            "snes_rtol": 1e-5,
        },
        **nullspace_args,
    )

    energy_solver = EnergySolver(
        T, u, approximation, time_step, ImplicitMidpoint, bcs=temp_bcs
    )

    epsilon = interface_thickness(K, min_cell_edge_length=True)
    epsilon = MPI.COMM_WORLD.allreduce(epsilon.dat.data_ro.min(), MPI.MIN)
    adv_kwargs = {"u": u, "timestep": time_step, "subcycles": prms.subcycles}
    reini_kwargs = {"epsilon": epsilon}
    level_set_solver = LevelSetSolver(
        psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs
    )

    # Collect and clean objects associated with the old mesh
    garbage_collect(mesh)

    return tstep_adapt, stokes_solver, energy_solver, level_set_solver, field_expr


def initialise(
    mesh: MeshGeometry, time_step: Constant
) -> tuple[Function, Function, Function, dict[Function, Operator]]:
    # Setup solvers
    _, stokes_solver, energy_solver, level_set_solver, field_expr = set_up(
        mesh, time_step
    )
    # Extract solution fields
    stokes = stokes_solver.solution
    T = energy_solver.solution
    psi = level_set_solver.solution
    # Initialise transported fields (temperature and level set)
    initial_temperature(T)
    initial_level_set(psi)
    # Initialise Stokes fields (velocity and pressure)
    stokes_solver.solve()

    # Collect and clean objects associated with the old mesh
    garbage_collect(mesh)

    return stokes, T, psi, field_expr


def time_loop(
    mesh: MeshGeometry,
    mesh_fields: dict[str, dict[str, bool | Function]],
    time_now: Constant,
    time_step: Constant,
    step: int,
    output_file: VTKFile,
) -> tuple[dict[str, Function], int]:
    tstep_adapt, stokes_solver, energy_solver, level_set_solver, field_expr = set_up(
        mesh, time_step
    )

    stokes = stokes_solver.solution
    T = energy_solver.solution
    psi = level_set_solver.solution

    mesh_fields = interpolate_fields(mesh, mesh_fields, [stokes, T, psi])

    # Time loop
    for _ in range(prms.iterations):
        tstep_adapt.update_timestep()

        level_set_solver.solve()
        energy_solver.solve()
        stokes_solver.solve()

        # Increment iteration count and time
        step += 1
        time_now.assign(time_now + time_step)

        # Write output
        if step % prms.output_frequency == 0:
            write_output(
                output_file,
                float(time_now) * prms.time_scale / prms.myr_to_seconds,
                *stokes.subfunctions,
                T,
                psi,
                field_expressions=field_expr,
            )

        # Check if simulation has completed
        if float(time_now) >= prms.time_end:
            # Checkpoint solution fields to disk
            with CheckpointFile("final_state.h5", "w") as final_checkpoint:
                final_checkpoint.save_mesh(mesh)
                final_checkpoint.save_function(stokes, name="Stokes")
                final_checkpoint.save_function(T, name="Temperature")
                final_checkpoint.save_function(psi, name="Level set")

            log("Reached end of simulation -- exiting time-step loop")

            break

    # Collect and clean objects associated with the old mesh
    garbage_collect(mesh)

    return mesh_fields, step
