"""Check the Hessian-vector product of the box demo's own tape, with a test that can fail.

``tests/hessian/cartesian_hvp_taylor.py`` checks the Hessian of a box problem that
differs from this demo: it uses a continuous (CG2) temperature with strong Dirichlet
conditions, while ``problem.py`` uses a discontinuous (DQ2) temperature with weak
conditions. So a pass or a failure of that harness says nothing certain about the tape
that the campaign runs. This script checks that tape directly, on the tiny case
(``forward.py --steps 10 --resolution 16``), in seconds:

1. Second-order Taylor remainders along a fixed direction ``h``:
   ``R2(eps) = |J(m + eps h) - J(m) - eps <dJ, h> - eps^2 / 2 <h, H h>|``.
   With a correct gradient and Hessian, ``R2`` falls as ``eps^3`` (rate 3) until the
   round-off of ``J`` takes over. A wrong Hessian gives rate 2.
2. A central difference of the gradient:
   ``(dJ(m + eps h) - dJ(m - eps h)) / (2 eps)`` against ``H h``. The relative error
   falls as ``eps^2`` with a correct Hessian. A wrong Hessian leaves it at a constant.

The problem is the S2 configuration: prior operator sigma 31.594026758304967 and
L 0.22071129807892717 as the metric, objective weights zero, first guess the observed
temperature, window 5. ``m`` is that first guess. ``h`` is a smooth field (a sum of
low sine modes with fixed coefficients), zero on the walls, so that the expansion stays
inside the bounds 0 <= m <= 1 that the optimiser uses for every eps of the sequence.

Usage (from a directory that holds the tiny reference state):
    python check_box_hessian.py REFERENCE.h5

Exit status 0 when the last two Taylor rates are above 2.8 and the last central
difference error is below 1e-5 and falls by at least a factor 10 over the sequence,
1 otherwise.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firedrake import Function, SpatialCoordinate, pi, sin  # noqa: E402
from pyadjoint import stop_annotating  # noqa: E402

from problem import build_problem  # noqa: E402


def smooth_direction(V):
    """A smooth direction in the control space that vanishes on the walls.

    The box is the unit square. ``sin(k pi y)`` is zero at y = 0 and y = 1, the bottom
    and top walls, so the direction does not move the wall values. Three modes with
    fixed coefficients make the direction the same on every machine and every run.

    Args:
        V: the control function space (CG1).

    Returns:
        Function: the direction ``h``, with maximum about 0.1, so that ``m + eps h``
        stays in [0, 1] for eps <= 0.1 wherever ``m`` is at least 0.01 from a bound.
    """
    x, y = SpatialCoordinate(V.mesh())
    h = Function(V, name="Direction")
    h.interpolate(0.05 * sin(pi * y) * sin(pi * x)
                  + 0.03 * sin(2 * pi * y) * sin(3 * pi * x)
                  + 0.02 * sin(3 * pi * y) * sin(2 * pi * x))
    return h


def pairing(cofunction, function):
    """The dual pairing <cofunction, function>, a plain number.

    Both live on the same dof layout (CG1 and its dual), so the pairing is the dot
    product of the two data arrays. The run is serial, so no reduction is needed.
    """
    return float(np.dot(cofunction.dat.data_ro, function.dat.data_ro))


def main(reference):
    """Run both checks and return True when both hold."""
    problem = build_problem(reference, window=5, alpha_d=0, alpha_s=0,
                            inner_product="regularisation", initial_guess="observed",
                            prior_sigma=31.594026758304967, prior_L=0.22071129807892717)
    rf = problem.reduced_functional
    V = problem.Tic.function_space()
    with stop_annotating():
        m = problem.Tic.copy(deepcopy=True)
        h = smooth_direction(V)
        print(f"m in [{m.dat.data_ro.min():.4f}, {m.dat.data_ro.max():.4f}], "
              f"h in [{h.dat.data_ro.min():.4f}, {h.dat.data_ro.max():.4f}]")

        # Value, gradient and Hessian action at m. The gradient is the dual object
        # (a Cofunction), and so is the Hessian action, so both pair with h directly.
        J0 = float(rf(m))
        g0 = rf.derivative()
        Hh = rf.hessian(h)
        dJh = pairing(g0, h)
        hHh = pairing(Hh, h)
        Hh_arr = Hh.dat.data_ro.copy()
        Hh_norm = np.linalg.norm(Hh_arr)
        print(f"J(m) = {J0:.15e}, <dJ, h> = {dJh:.6e}, <h, H h> = {hHh:.6e}, "
              f"|H h| = {Hh_norm:.6e}")

        eps_list = [1e-1 / 2 ** k for k in range(6)]
        r2, cd = [], []
        for eps in eps_list:
            # Taylor remainder: J at m + eps h against the quadratic model.
            mp = Function(V).assign(m + eps * h)
            Jp = float(rf(mp))
            gp = rf.derivative().dat.data_ro.copy()
            r2.append(abs(Jp - J0 - eps * dJh - 0.5 * eps ** 2 * hHh))
            # Central difference of the gradient against H h.
            mm = Function(V).assign(m - eps * h)
            rf(mm)
            gm = rf.derivative().dat.data_ro.copy()
            fd = (gp - gm) / (2 * eps)
            cd.append(np.linalg.norm(fd - Hh_arr) / Hh_norm)

        rates = [np.log(r2[i] / r2[i + 1]) / np.log(2) for i in range(len(r2) - 1)]
        print("eps          " + " ".join(f"{e:10.3e}" for e in eps_list))
        print("R2           " + " ".join(f"{r:10.3e}" for r in r2))
        print("R2 rate                 " + " ".join(f"{r:10.3f}" for r in rates))
        print("|fd - Hh|/|Hh| " + " ".join(f"{c:10.3e}" for c in cd))

    taylor_ok = min(rates[-2:]) > 2.8
    cd_ok = cd[-1] < 1e-5 and cd[0] / cd[-1] > 10
    print(f"Taylor rate test: {'ok' if taylor_ok else 'FAILED'} "
          f"(last two rates {rates[-2]:.3f}, {rates[-1]:.3f}, need > 2.8)")
    print(f"central difference test: {'ok' if cd_ok else 'FAILED'} "
          f"(last error {cd[-1]:.3e}, need < 1e-5; reduction {cd[0] / cd[-1]:.1f}, "
          f"need > 10)")
    return taylor_ok and cd_ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
