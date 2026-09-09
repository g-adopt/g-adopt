"""Second-order (Hessian) Taylor test for the 2D cylindrical adjoint case.

This is the cylindrical counterpart of ``cartesian_hvp_taylor.py``. It reuses the real
problem setup from ``tests/adjoint_2d_cylindrical/inverse.py`` rather than restating it,
so the tape under test is the same tape the existing first-order test builds. The only
thing this script changes is the length of the trajectory, and only through the single
parameter that controls it.

Why the trajectory is shortened. The full case runs 125 timesteps of a nonlinear Stokes
problem on a 128-layer annulus, which is a 16-core job. A 5-step run costs a few minutes
in serial and still contains every feature that the box case lacks: the strain-rate and
temperature dependent viscosity, the ``newtonls`` nonlinear solve, the DQ2 temperature
space and the rotational nullspace. If the Hessian is wrong because of the rheology, it
is already wrong after five steps. If instead the rate only degrades as steps
accumulate, the fault is in the time loop or the checkpointing, and the rheology is
innocent. Those two outcomes are what this run separates.

Shortening the run needs a matching reference state, because the control is initialised
from, and the observation is taken from, timestep ``max_timesteps - 1`` of the forward
run. A 125-step ``Checkpoint_State.h5`` therefore cannot be reused for a 5-step inverse
problem. Use ``--make-reference`` once to generate the matching short forward run.

Usage:
    # generate the 5-step reference state (once)
    python cylindrical_hvp_taylor.py --timesteps 5 --make-reference

    # run the Hessian Taylor test on the temperature misfit term
    python cylindrical_hvp_taylor.py --timesteps 5 --eps0 0.05 --levels 7
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from mpi4py import MPI

from gadopt import *
from gadopt.inverse import *

# The cylindrical case is a directory of plain scripts, not a package, and both
# forward.py and inverse.py open "Checkpoint230.h5" and "Checkpoint_State.h5" by bare
# filename. So the directory has to be importable, and the process has to run in a
# directory that holds those files.
CYLINDRICAL_DIR = Path(__file__).resolve().parent.parent / "adjoint_2d_cylindrical"
sys.path.insert(0, str(CYLINDRICAL_DIR))

from taylor_diagnostics import (  # noqa: E402
    make_perturbation, taylor_remainders, report, write_json,
    hessian_vs_finite_difference, report_fd,
)

# The weightings that select a single objective term, in the order
# (alpha_T, alpha_u, alpha_d, alpha_s). A non-positive weight switches a term off.
# These are the same tuples as tests/adjoint_2d_cylindrical/cases.py.
CASE_WEIGHTS = {
    "Tobs": (+1, -1, -1, -1),
    "uobs": (-1, +1, -1, -1),
    "damping": (-1, -1, +1, -1),
    "smoothing": (-1, -1, -1, +1),
}


def prepare_rundir(rundir):
    """Create a working directory that the cylindrical scripts can run in.

    ``forward.py`` reads its starting temperature from ``Checkpoint230.h5`` in the
    current directory, and writes ``Checkpoint_State.h5`` there. Rather than write
    build products into the tracked test directory, run in a separate directory and
    link the one input that is tracked.

    The file is copied and not symlinked. On a Lustre filesystem the ompio MPI-IO
    component queries the stripe layout of the file it opens, and that query fails on
    a symlink with "get_stripe failed: 61", after which the parallel HDF5 open fails.
    The file is 4 MB, so a copy costs nothing.

    Only rank 0 touches the filesystem, and the other ranks wait on a barrier. Every
    rank running the same "test then create" sequence is a race: several ranks see the
    file missing at the same moment and all try to make it.

    Args:
        rundir (Path): the directory to run in. Created if it does not exist.

    Returns:
        Path: the same directory, with Checkpoint230.h5 copied into it.
    """
    rundir = Path(rundir).resolve()
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        rundir.mkdir(parents=True, exist_ok=True)

        target = rundir / "Checkpoint230.h5"
        # A symlink left by an older version of this script is replaced by a copy.
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            shutil.copyfile(CYLINDRICAL_DIR / "Checkpoint230.h5", target)

    # Hold every rank here until the directory and the link exist, so that no rank
    # chdirs into a directory that is not ready.
    comm.Barrier()

    return rundir


def set_max_timesteps(max_timesteps):
    """Shorten the reference run and the inverse problem to a few timesteps.

    ``forward.py`` and ``inverse.py`` both take the trajectory length from
    ``get_reference_values()["max_timesteps"]``. ``inverse.py`` imported that function
    by value, so it holds its own reference and both modules have to be redirected.

    Nothing else in the returned dictionary is touched, so the rheology, the Rayleigh
    number and the timestep are exactly those of the full case.

    Args:
        max_timesteps (int): the number of timesteps to run.
    """
    import forward
    import inverse

    original = forward.get_reference_values

    def shortened():
        values = original()
        values["max_timesteps"] = max_timesteps
        return values

    forward.get_reference_values = shortened
    inverse.get_reference_values = shortened


def override_viscosity(kind, delta_fraction=0.05):
    """Replace the production viscosity with a simpler rheology, to isolate a suspect.

    The production law in ``forward.get_viscosity`` is a depth- and temperature-
    dependent linear branch combined by a harmonic mean with a strain-rate dependent
    plastic branch, then clipped from below:

        mu = conditional(mu_eff > mu_min, mu_eff, mu_min)

    That clip is continuous but not continuously differentiable, so the functional is
    C1 and not C2 in the control. UFL differentiates the two branches separately,
    which is right almost everywhere for the first derivative but drops the delta
    layer the second derivative carries on the switching surface. No Hessian can then
    be exact, and the error shows up as a fixed eps^2 term in the second-order Taylor
    remainder while the gradient stays clean.

    Two replacements are offered, both keeping the depth and temperature dependence
    exactly as the production function builds it.

    "smooth" keeps the plastic branch and replaces only the clip with the standard
    smooth maximum, 0.5*(a + b + sqrt((a-b)^2 + delta^2)), which is C-infinity and
    converges to the hard maximum as delta goes to zero. This isolates the clip.

    "temperature" drops the plastic branch and the clip together, leaving the linear
    depth- and temperature-dependent viscosity alone. This is the cheapest and
    sharpest test: the viscosity no longer depends on the velocity, so Stokes becomes
    linear and g-adopt selects ksponly instead of newtonls, removing the strain-rate
    floor, the clip and the nonlinear solve in one step. If the eps^2 defect survives
    that, none of the rheology is responsible.

    Sia confirmed the clamp carries no physical meaning, so smoothing it is a real fix
    rather than a workaround; dropping the plastic branch is a diagnostic, not a fix.

    Args:
        kind (str): "smooth" or "temperature".
        delta_fraction (float): for "smooth", the smoothing width as a fraction of
            mu_min. Small enough not to change the physics, large enough to make the
            second derivative bounded on the mesh.
    """
    import forward
    import inverse

    def linear_viscosity(r, T):
        """The depth- and temperature-dependent branch, as the production law builds it.

        Args:
            r: radial coordinate.
            T: temperature field.

        Returns:
            The linear viscosity, with the 410 km and 660 km steps and the Arrhenius
            temperature dependence.
        """
        geometry_parameters = forward.get_geometry_parameters()
        reference_values = forward.get_reference_values()

        # A step function used to build the viscosity jumps at the phase boundaries.
        def step_func(centre, mag, increasing=True, sharpness=50):
            return mag * (
                0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
            )

        mu_lin = reference_values["mu_0"]
        for line, step in zip(
            [5.0 * (geometry_parameters["rmax"] - r), 1.0, 1.0],
            [
                step_func(geometry_parameters["r_660"], 30, False),
                step_func(geometry_parameters["r_410"], 10, False),
                step_func(geometry_parameters["rmax"], 10, True),
            ],
        ):
            mu_lin += line * step

        # Arrhenius temperature dependence, a factor of mu_T across the full range.
        mu_lin *= exp(-ln(Constant(reference_values["mu_T"])) * T)

        return mu_lin

    def effective_viscosity(r, T, u):
        """The production viscosity up to, but not including, the mu_min clip.

        The linear branch combined by a harmonic mean with the strain-rate dependent
        plastic branch. The 1e-10 floor inside the strain-rate norm is smooth, so it
        is left exactly as the production law has it.

        Args:
            r: radial coordinate.
            T: temperature field.
            u: velocity field.

        Returns:
            mu_eff, the unclipped effective viscosity.
        """
        reference_values = forward.get_reference_values()
        geometry_parameters = forward.get_geometry_parameters()

        mu_lin = linear_viscosity(r, T)

        eps = sym(grad(u))
        epsii = sqrt(inner(eps, eps) + 1e-10)
        sigma_y = (reference_values["sigma_y"]
                   + reference_values["sigma_y_depth"] * (geometry_parameters["rmax"] - r))
        mu_plast = 0.1 + (sigma_y / epsii)
        return 2 * (mu_lin * mu_plast) / (mu_lin + mu_plast)

    def temperature_only(r, T, u):
        # No dependence on u at all, so Stokes is linear and runs under ksponly.
        return linear_viscosity(r, T)

    def tanh_floor(r, T, u):
        """The clip as a tanh blend between the two branches.

        Writing x = mu_eff - mu_min and w = 0.5*(1 + tanh(x/delta)),

            mu = w*mu_eff + (1 - w)*mu_min

        which tends to mu_eff when x >> delta, to mu_min when x << -delta, and equals
        mu_min exactly at the crossing. It is C-infinity and cannot overflow, because
        tanh is bounded.

        One caveat that the square-root form does not share: this is the swish
        function in disguise, and swish is not monotone. For mu_eff slightly below
        mu_min the blend dips below the floor, reaching a minimum of -0.139*delta. At
        delta = 0.02 the viscosity dips to 0.397 rather than 0.4, which is physically
        negligible but does break the one guarantee the floor exists to provide.

        It is here so that the result can be checked against a smoother of a different
        shape. If the eps^2 defect collapses for both this and the square-root form,
        the finding is about the kink and not about the particular smoothing.
        """
        reference_values = forward.get_reference_values()
        mu_min = reference_values["mu_min"]
        mu_eff = effective_viscosity(r, T, u)

        delta = Constant(delta_fraction * mu_min)
        weight = 0.5 * (1 + tanh((mu_eff - mu_min) / delta))
        return weight * mu_eff + (1 - weight) * mu_min

    def hard_floor(r, T, u):
        """The former production clip: a hard conditional at mu_min.

        Continuous, not continuously differentiable. This is the control run: the
        functional is C1 and not C2, so the second-order Taylor remainder must show a
        fixed eps^2 defect D no matter how correct the Hessian-vector product is.
        """
        reference_values = forward.get_reference_values()
        mu_min = reference_values["mu_min"]
        mu_eff = effective_viscosity(r, T, u)
        return conditional(mu_eff > mu_min, mu_eff, mu_min)

    def smooth_floor(r, T, u):
        """The clip as a square-root (Cauchy) smooth maximum.

            max(a, b) ~ 0.5*(a + b + sqrt((a - b)^2 + delta^2))

        C-infinity, no overflow, and cheap. The property that makes it the default
        here is algebraic: the result is always at least max(a, b), because
        sqrt((a-b)^2 + delta^2) >= |a - b|. So it can only overshoot the floor, never
        undershoot it, which matters because the floor exists to keep the Stokes
        operator conditioned. The overshoot is delta/2 at the crossing and decays
        like delta^2/(4|a-b|) away from it, so the change is tightly localised around
        the switching surface.
        """
        reference_values = forward.get_reference_values()
        mu_min = reference_values["mu_min"]
        mu_eff = effective_viscosity(r, T, u)

        delta = Constant(delta_fraction * mu_min)
        return 0.5 * (mu_eff + mu_min + sqrt((mu_eff - mu_min) ** 2 + delta ** 2))

    replacement = {"hard": hard_floor,
                   "smooth": smooth_floor,
                   "tanh": tanh_floor,
                   "temperature": temperature_only}[kind]

    # inverse.py imported get_viscosity by value, so it holds its own reference and
    # both modules have to be redirected.
    forward.get_viscosity = replacement
    inverse.get_viscosity = replacement


def make_reference(max_timesteps):
    """Run the cylindrical forward model to produce a matching reference state.

    Writes ``Checkpoint_State.h5`` in the current directory, holding the initial
    temperature, the layer average, the velocity at every step, and the temperature at
    the final step. Those are the fields the inverse problem reads back.

    Args:
        max_timesteps (int): the number of timesteps to run.
    """
    import forward

    set_max_timesteps(max_timesteps)
    forward.run_forward()


def build_reduced_functional(case, max_timesteps):
    """Build the cylindrical reduced functional for one objective term.

    This delegates entirely to ``inverse.generate_inverse_problem``, so the forward
    model, the viscosity, the solvers, the nullspaces and the functional are the
    production ones and not a copy.

    Args:
        case (str): the objective term, a key of ``CASE_WEIGHTS``.
        max_timesteps (int): the number of timesteps to run.

    Returns:
        tuple: (ReducedFunctional, Function) the reduced functional and the control
            value at which to expand the Taylor series.
    """
    import inverse

    set_max_timesteps(max_timesteps)

    alpha_T, alpha_u, alpha_d, alpha_s = CASE_WEIGHTS[case]

    # No checkpointing schedule: the whole tape is held in memory, so nothing is
    # recomputed. A schedule is a separate variable and is tested separately.
    inverse_problem = inverse.generate_inverse_problem(
        alpha_T=alpha_T,
        alpha_u=alpha_u,
        alpha_d=alpha_d,
        alpha_s=alpha_s,
        checkpointing_schedule=None,
        uimposed=False,
    )

    return inverse_problem["reduced_functional"], inverse_problem["control"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="Tobs", choices=list(CASE_WEIGHTS))
    parser.add_argument("--timesteps", type=int, default=5,
                        help="length of the forward trajectory")
    parser.add_argument("--make-reference", action="store_true",
                        help="run the forward model first to build Checkpoint_State.h5")
    parser.add_argument("--rundir", default=None,
                        help="directory to run in (default: run_cylindrical next to this script)")
    parser.add_argument("--perturbation", default="white", choices=["white", "smooth"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eps0", type=float, default=0.05,
                        help="largest perturbation size in the Taylor expansion")
    parser.add_argument("--levels", type=int, default=7,
                        help="number of halvings of epsilon")
    parser.add_argument("--viscosity", default="production",
                        choices=["production", "hard", "smooth", "tanh", "temperature"],
                        help="which rheology to use. 'production' is forward.py as "
                             "it stands, which applies the mu_min floor as a "
                             "square-root smooth maximum with the width set in "
                             "get_reference_values. 'hard' restores the former "
                             "conditional clip, as the control run. 'smooth' is the "
                             "same form as production with the width taken from "
                             "--delta-fraction; 'tanh' uses a tanh blend; "
                             "'temperature' drops the plastic branch and the clip "
                             "entirely, which also makes Stokes linear. Anything but "
                             "'production' requires --make-reference, since the "
                             "rheology changes.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="number of timed calls of functional, derivative and "
                             "Hessian (default 2, 2, 3). Use 1 for a run whose only "
                             "purpose is the Taylor sweep.")
    parser.add_argument("--json", default=None,
                        help="write the result, rates, two-term fit and timings to "
                             "this JSON file (rank 0 only)")
    parser.add_argument("--delta-fraction", type=float, default=0.05,
                        help="smoothing width of the viscosity floor, as a fraction "
                             "of mu_min. It must comfortably exceed the excursion in "
                             "mu that the smallest epsilon produces (about epsilon "
                             "itself here), or the test still probes a kink.")
    parser.add_argument("--snes-rtol", type=float, default=None,
                        help="override the Newton tolerance for the nonlinear Stokes "
                             "solve. The tape treats each recorded solve as exact, so "
                             "Newton truncation is a systematic error in the taped "
                             "second derivative. Tightening this is the decisive test "
                             "of whether that truncation explains the R2 defect.")
    parser.add_argument("--fd-check", action="store_true",
                        help="compare H h against a finite difference of the gradient "
                             "instead of running the Taylor expansion of J")
    args = parser.parse_args()

    if args.snes_rtol is not None:
        # Mutate the shared default dictionary before any solver is built, so that
        # StokesSolver picks the tightened tolerance up when it selects newtonls.
        import gadopt.stokes_integrators as stokes_integrators
        stokes_integrators.newton_stokes_solver_parameters["snes_rtol"] = args.snes_rtol
        if MPI.COMM_WORLD.rank == 0:
            print(f"Stokes snes_rtol overridden to {args.snes_rtol:g}")

    if args.viscosity != "production":
        override_viscosity(args.viscosity, args.delta_fraction)
        if MPI.COMM_WORLD.rank == 0:
            print(f"viscosity overridden: {args.viscosity}, "
                  f"delta = {args.delta_fraction} * mu_min")

    # The JSON path is resolved before the chdir below, so that a relative path
    # means relative to where the command was typed and not to the run directory.
    if args.json:
        args.json = str(Path(args.json).resolve())

    rundir = args.rundir or (Path(__file__).resolve().parent / "run_cylindrical")
    rundir = prepare_rundir(rundir)
    os.chdir(rundir)

    if args.make_reference:
        make_reference(args.timesteps)
        # The forward run leaves nothing on the tape that the inverse problem should
        # inherit, and generate_inverse_problem clears the tape anyway.

    reduced_functional, control = build_reduced_functional(args.case, args.timesteps)
    delta = make_perturbation(control.function_space(), args.perturbation, args.seed)

    label = (f"cylindrical {args.case}, {args.timesteps} timesteps, "
             f"viscosity={args.viscosity}(delta={args.delta_fraction}), "
             f"snes_rtol={args.snes_rtol or 'default'}")

    if args.fd_check:
        # An independent check of the second-order adjoint that does not rely on the
        # Taylor expansion of J at all.
        result = hessian_vs_finite_difference(reduced_functional, control, delta)
        if MPI.COMM_WORLD.rank == 0:
            report_fd(label, result)
    else:
        repeat_kwargs = {}
        if args.repeats is not None:
            repeat_kwargs = dict(n_repeat_functional=args.repeats,
                                 n_repeat_derivative=args.repeats,
                                 n_repeat_hessian=args.repeats)
        result = taylor_remainders(reduced_functional, control, delta,
                                   eps0=args.eps0, levels=args.levels, **repeat_kwargs)
        if MPI.COMM_WORLD.rank == 0:
            report(label, result)
            if args.json:
                import firedrake
                write_json(args.json, label, result, extra={
                    "argv": sys.argv,
                    "ranks": MPI.COMM_WORLD.size,
                    "firedrake": os.path.dirname(firedrake.__file__),
                    "firedrake_tag": os.environ.get("FIREDRAKE_TAG"),
                    "pbs_jobid": os.environ.get("PBS_JOBID"),
                })


if __name__ == "__main__":
    main()
