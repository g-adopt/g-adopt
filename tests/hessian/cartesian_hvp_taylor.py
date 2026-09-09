"""Second-order (Hessian) Taylor test for the Cartesian (box) adjoint case.

The existing test at ``tests/adjoint/taylor_test.py`` only checks the first-order
Taylor remainder, so it certifies the gradient and says nothing about the Hessian.
This script rebuilds the same reduced functional and runs ``taylor_to_dict``, which
reports all three remainders at once:

    R0 = |J(m + eps h) - J(m)|                              -> order 1
    R1 = |J(m + eps h) - J(m) - eps <dJ, h>|                -> order 2
    R2 = |J(m + eps h) - J(m) - eps <dJ, h> - eps^2/2 <h, H h>|  -> order 3

R2 converging at order 3 is the statement that the Hessian-vector product is right.
A rate that settles near 2 means the Hessian is wrong by an O(1) amount, because the
term 0.5 * eps^2 * <h, (H - H_computed) h> never cancels. A rate that decays towards 0
with a flat residual is a noise floor instead, not a wrong Hessian.

``taylor_to_dict`` evaluates one Hessian-vector product at the expansion point and
reuses it for every epsilon, so this is a test of a single H*h, which is exactly the
object the optimiser will consume.

Note on the pure regularisation terms: for ``damping`` and ``smoothing`` the objective
is an exact quadratic form in the control, because T0 is a linear projection of Tic.
The third-order remainder is then identically zero and R2 sits at roundoff. The rate
line for those two cases is the slope of machine noise and carries no information; the
meaningful check there is that R2 stays at roundoff, which this script asserts instead
of checking a rate.

Usage:
    python cartesian_hvp_taylor.py [case] [scheduler] [--perturbation white|smooth]
                                   [--seed N] [--timesteps N]

    case        one of damping, smoothing, Tobs, uobs, uimposed (default: all)
    scheduler   one of noscheduler, fullmemory, fullstorage (default: noscheduler)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *

# The tests/adjoint directory holds cases.py and the forward checkpoint that this
# script reuses, so make it importable and locate its artefacts from here.
ADJOINT_DIR = Path(__file__).resolve().parent.parent / "adjoint"
sys.path.insert(0, str(ADJOINT_DIR))

from cases import cases, schedules  # noqa: E402

# The Taylor machinery itself is shared with the cylindrical driver.
from taylor_diagnostics import make_perturbation, taylor_remainders, report  # noqa: E402

# Match the quadrature degree used by the first-order test so that the functional
# assembled here is bit-for-bit the same object.
ds_t = ds_t(degree=6)
dx = dx(degree=6)


def build_reduced_functional(case, scheduler_name, max_timesteps=None):
    """Populate the tape for one objective term of the box case.

    This is ``tests/adjoint/taylor_test.py:rectangle_taylor_test`` up to the point
    where the reduced functional exists, with the Taylor test itself removed and an
    optional cap on the number of timesteps added for the time-bisection experiment.

    Args:
        case (str): objective functional term, one of "damping", "smoothing",
            "Tobs", "uobs", "uimposed".
        scheduler_name (str): checkpointing schedule, a key of ``schedules``.
        max_timesteps (int, optional): if given, run only this many timesteps
            instead of the full forward history. Used to test whether a Hessian
            error grows with the length of the trajectory.

    Returns:
        tuple: (ReducedFunctional, Function) the reduced functional and the control
            value (the initial guess) at which to expand.
    """
    checkpoint_filename = ADJOINT_DIR / "adjoint-demo-checkpoint-state.h5"

    # Clear the tape so the adjoint reflects only the forward problem built here.
    tape = get_working_tape()
    tape.clear_tape()

    if not annotate_tape():
        continue_annotation()

    if scheduler_name == "fullstorage":
        enable_disk_checkpointing()

    # Schedules only apply to the cases that actually step in time.
    if case in ["Tobs", "uobs", "uimposed"] and schedules[scheduler_name] is not None:
        tape.enable_checkpointing(schedules[scheduler_name])

    with CheckpointFile(str(checkpoint_filename), "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = True

    # On an extruded mesh the vertical boundaries are tagged "top" and "bottom".
    boundary = get_boundary_ids(mesh)

    with CheckpointFile(str(checkpoint_filename), "r") as f:
        temperature_timestepping_info = f.get_timestepping_history(mesh, "Temperature")
        Tobs = f.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
        Tobs.rename("Observed Temperature")
        Tic_ref = f.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
        Tic_ref.rename("Reference Initial Temperature")

    # Function spaces: Q2-Q1 Taylor-Hood for Stokes, Q2 for temperature.
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 2)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(Q, name="Temperature")

    # Isoviscous Boussinesq: mu = 1, so Stokes is linear. This is the property that
    # makes the box case the clean reference against the nonlinear cylindrical case.
    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)
    timesteps = int(temperature_timestepping_info["index"][-1]) + 1

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # All boundaries free slip; heated from below (T = 1), cooled from above (T = 0).
    uobs = Function(V, name="Observed_Velocity")
    stokes_bcs = {
        boundary.bottom: {"uy": 0},
        boundary.top: {"uy": 0} if case != "uimposed" else {"u": uobs},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
    }
    temp_bcs = {
        boundary.bottom: {"T": 1.0},
        boundary.top: {"T": 0.0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(
        z,
        approximation,
        T,
        bcs=stokes_bcs,
        constant_jacobian=True,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
    )

    # The purely regularisation cases need no trajectory, so they start at the last
    # step and take a single pass; the misfit cases run the whole history.
    initial_timestep = 0 if case in ["Tobs", "uobs"] else timesteps - 1

    # The control lives in Q1, coarser than the Q2 temperature space.
    Q1 = FunctionSpace(mesh, "CG", 1)

    with CheckpointFile(str(checkpoint_filename), "r") as f:
        Taverage = f.load_function(mesh, "Average_Temperature", idx=0)
    Tic = Function(Q1, name="Initial_Condition_Temperature").assign(Taverage)

    # The smoothing term acts in the control space, so its boundary conditions are
    # imposed in Q1 rather than in the Q2 temperature space.
    T0_bcs = [DirichletBC(Q1, 0., boundary.top), DirichletBC(Q1, 1., boundary.bottom)]
    T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)

    control = Control(Tic)

    T.project(Tic, bcs=energy_solver.strong_bcs)

    # Optionally shorten the trajectory. A Hessian error that appears only once
    # several steps have accumulated points at the time loop or the checkpointing,
    # not at the physics of a single step.
    final_timestep = timesteps
    if max_timesteps is not None and case in ["Tobs", "uobs"]:
        final_timestep = min(timesteps, initial_timestep + max_timesteps)

    u_misfit = 0.0

    for time_idx in tape.timestepper(iter(range(initial_timestep, final_timestep))):
        with CheckpointFile(str(checkpoint_filename), "r") as f:
            uobs.assign(f.load_function(mesh, name="Velocity", idx=time_idx))

        stokes_solver.solve()
        energy_solver.solve()
        if case == "uobs":
            u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)

    # Normalisation terms, so that every objective term is O(1).
    damping = assemble((T0 - Taverage) ** 2 * dx)
    norm_damping = assemble(Taverage**2 * dx)
    smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
    norm_obs = assemble(Tobs**2 * dx)
    norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

    t_misfit = assemble((T - Tobs) ** 2 * dx)

    if case in ["Tobs", "uimposed"]:
        objective = t_misfit
    elif case == "uobs":
        objective = norm_obs * u_misfit / (final_timestep - initial_timestep) / norm_u_surface
    elif case == "damping":
        objective = norm_obs * damping / norm_damping
    else:
        objective = norm_obs * smoothing / norm_smoothing

    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    return reduced_functional, Tic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=None, choices=cases)
    parser.add_argument("scheduler", nargs="?", default="noscheduler", choices=list(schedules))
    parser.add_argument("--perturbation", default="white", choices=["white", "smooth"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="cap the forward trajectory at this many steps")
    parser.add_argument("--eps0", type=float, default=1.0,
                        help="largest perturbation size in the Taylor expansion")
    parser.add_argument("--levels", type=int, default=8,
                        help="number of halvings of epsilon")
    args = parser.parse_args()

    selected = [args.case] if args.case else cases

    for case in selected:
        reduced_functional, Tic = build_reduced_functional(case, args.scheduler, args.timesteps)
        delta = make_perturbation(Tic.function_space(), args.perturbation, args.seed)

        result = taylor_remainders(reduced_functional, Tic, delta,
                                   eps0=args.eps0, levels=args.levels)

        if MPI.COMM_WORLD.rank == 0:
            report(case, result)

        # The remainder loop runs under stop_annotating; switch taping back on so the
        # next case in the loop can populate a fresh tape.
        continue_annotation()


if __name__ == "__main__":
    main()
