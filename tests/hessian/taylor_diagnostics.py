"""Shared machinery for the second-order (Hessian) Taylor diagnostics.

The Taylor remainders are

    R0 = |J(m + eps h) - J(m)|                                   -> order 1
    R1 = |J(m + eps h) - J(m) - eps <dJ, h>|                     -> order 2
    R2 = |J(m + eps h) - J(m) - eps <dJ, h> - eps^2/2 <h, H h>|  -> order 3

R2 converging at order 3 is the statement that the Hessian-vector product is right.
A rate that settles near 2 means the Hessian is wrong by an O(1) amount, because the
term 0.5 * eps^2 * <h, (H - H_computed) h> never cancels. A rate that decays towards
0 with a flat residual is a roundoff floor instead, and says nothing either way.

Both failure modes are real and they look different, but only if epsilon is swept over
a wide enough range to separate them. ``pyadjoint.taylor_to_dict`` is hard-wired to
four epsilons starting at 0.01, and for these problems that window is too narrow: at
the large end the perturbation can still be pre-asymptotic, and at the small end R2 can
already be at roundoff. The functions here take the range as an argument and report the
residuals relative to |J| so the floor is visible.
"""

import time

import numpy as np

from gadopt import *
from gadopt.inverse import *


def make_perturbation(function_space, kind="white", seed=None):
    """Build the direction h in which the Taylor expansion is taken.

    Args:
        function_space: the control space (Q1 here).
        kind (str): "white" reproduces the direction used by the existing first-order
            test, ``np.random.random`` on the nodal values. That field is grid-scale
            noise, it is not zero mean (every entry lies in [0, 1)), and it is not
            mesh independent. "smooth" is a zero-mean Gaussian field passed through
            one Helmholtz smoothing solve, which removes the mean and damps the
            roughest modes, where a discrete Hessian is least accurate.
        seed (int, optional): seed for the random generator, so a run repeats.

    Returns:
        Function: the perturbation direction, L2-normalised for "smooth" and left
            unnormalised for "white" so that it matches the existing test exactly.
    """
    delta = Function(function_space, name="Delta_Temperature")

    if kind == "white":
        # Deliberately identical to tests/adjoint/taylor_test.py, including the use
        # of the global numpy random state when no seed is given.
        if seed is not None:
            np.random.seed(seed)
        delta.dat.data[:] = np.random.random(delta.dat.data.shape)
        return delta

    if kind != "smooth":
        raise ValueError(f"unknown perturbation kind: {kind}")

    rng = np.random.default_rng(seed)
    raw = Function(function_space)
    raw.dat.data[:] = rng.standard_normal(raw.dat.data.shape)

    # One Helmholtz solve, (1 - L^2 grad^2) h = raw, acts as a low-pass filter with
    # a correlation length L. L is set to a few cells so that the direction is
    # resolved by the mesh rather than by the node spacing.
    length_scale = Constant(0.05)
    trial = TrialFunction(function_space)
    test = TestFunction(function_space)
    a = (trial * test + length_scale**2 * dot(grad(trial), grad(test))) * dx
    L_form = raw * test * dx
    # Keep this solve off the tape. stop_annotating works in every firedrake
    # version in use; the annotate keyword of solve does not exist in PR #4638.
    with stop_annotating():
        solve(a == L_form, delta)

    # Remove the mean, so the direction does not simply translate the control.
    one = Function(function_space).assign(1.0)
    volume = assemble(one * dx)
    mean = assemble(delta * dx) / volume
    delta.assign(delta - Constant(mean))

    # Normalise in L2, so epsilon has a consistent meaning between runs.
    norm = float(assemble(delta**2 * dx) ** 0.5)
    delta.assign(delta / Constant(norm))

    return delta


def taylor_remainders(reduced_functional, m, h, eps0=1.0, levels=8,
                      n_repeat_functional=2, n_repeat_derivative=2, n_repeat_hessian=3):
    """Compute the three Taylor remainders over a configurable range of epsilon.

    This does what ``pyadjoint.taylor_to_dict`` does, with one difference that matters
    here. ``taylor_to_dict`` is hard-wired to four epsilons starting at 0.01, and for
    the box misfit terms R2 has already reached roundoff at the first of them, so the
    reported rates are the slope of noise and say nothing about the Hessian. Starting
    at a larger epsilon and taking more levels gives several decades of signal before
    the floor, so a wrong Hessian (rate tending to 2) can be told apart from a noise
    floor (flat residual, rate falling to 0).

    One Hessian-vector product is computed at the expansion point and reused for every
    epsilon, exactly as the optimiser would use it.

    Args:
        reduced_functional (ReducedFunctional): the functional to expand.
        m (Function): the expansion point in control space.
        h (Function): the perturbation direction.
        eps0 (float): the largest perturbation size.
        levels (int): the number of halvings of epsilon.
        n_repeat_functional (int): how many times to evaluate and time the functional
            at the expansion point before the derivative is taken.
        n_repeat_derivative (int): how many times to evaluate and time the derivative.
        n_repeat_hessian (int): how many times to evaluate and time the Hessian-vector
            product. The product from the last call is the one used in R2.

    Returns:
        dict: with keys "Jm" (the functional at the expansion point), "eps", and
            "R0", "R1", "R2", each a list of residuals in order of decreasing epsilon.
            The key "timings" holds the wall-clock cost in seconds of every call of
            the functional, the derivative and the Hessian-vector product, as a list
            per key in call order, plus the total time of the epsilon sweep. That is
            what a cost model for the optimiser needs.
    """
    # Every timed operation is repeated. The first call of each includes kernel
    # compilation, cache warm-up and, for the cached adjoint solvers of firedrake PR
    # #4638, the construction of the adjoint and tangent-linear problems. The later
    # calls are the steady cost that an optimiser pays per iteration. Both are kept,
    # as a list per key in order of the calls, because the discussion about the cost
    # of a Hessian-vector product hinges on which of the two one quotes.
    timings = {"functional": [], "derivative": [], "hessian": []}

    with stop_annotating():
        for _ in range(n_repeat_functional):
            start = time.perf_counter()
            Jm = reduced_functional(m)
            timings["functional"].append(time.perf_counter() - start)

        # First derivative, contracted with the direction. This replays the tape
        # backwards once, so it costs about one adjoint sweep.
        for _ in range(n_repeat_derivative):
            start = time.perf_counter()
            dJ = reduced_functional.derivative()
            timings["derivative"].append(time.perf_counter() - start)
        dJdm = h._ad_dot(dJ)

        # A single Hessian-vector product, contracted with the same direction. This is
        # one tangent-linear sweep plus one second-order adjoint sweep, so it should
        # cost about twice the derivative.
        for _ in range(n_repeat_hessian):
            start = time.perf_counter()
            Hh = reduced_functional.hessian(h)
            timings["hessian"].append(time.perf_counter() - start)
        hHh = h._ad_dot(Hh)

        epsilons = [eps0 / 2**i for i in range(levels)]
        out = {"Jm": Jm, "eps": epsilons, "dJdm": dJdm, "hHh": hHh,
               "R0": [], "R1": [], "R2": [], "timings": timings}

        start = time.perf_counter()
        for eps in epsilons:
            Jp = reduced_functional(m._ad_add(h._ad_mul(eps)))
            out["R0"].append(abs(Jp - Jm))
            out["R1"].append(abs(Jp - Jm - eps * dJdm))
            out["R2"].append(abs(Jp - Jm - eps * dJdm - 0.5 * eps**2 * hHh))
        timings["epsilon_sweep"] = time.perf_counter() - start

    # Leave the functional evaluated at the expansion point, as taylor_to_dict does.
    reduced_functional(m)

    return out


def rates(residuals):
    """Convert a list of Taylor residuals at halving epsilon into local rates.

    ``taylor_to_dict`` halves epsilon between successive entries, so the local
    convergence order between two entries is log2(R_i / R_{i+1}).

    Args:
        residuals (list): the residual values, in order of decreasing epsilon.

    Returns:
        list: the local rates, one shorter than the input.
    """
    out = []
    for a, b in zip(residuals[:-1], residuals[1:]):
        # A residual that has hit roundoff can be zero or can rise again; report a
        # NaN there rather than crashing, so the caller sees the noise floor.
        if a <= 0.0 or b <= 0.0:
            out.append(float("nan"))
        else:
            out.append(np.log2(a / b))
    return out


def two_term_fit(eps, R2):
    """Fit R2 = |C * eps^3 + D * eps^2| to the two smallest values of epsilon.

    The R2 rate on its own is misleading. C is the true third-order term and D is
    the defect in <h, H h>. When C and D carry opposite signs, R2 passes close to a
    zero and the local rate rises above 3, which hides the defect. Reading D directly
    avoids that. With two unknowns and two residuals the fit is exact, but R2 is an
    absolute value, so the sign of each residual is unknown. All four sign
    combinations are returned. When the defect is small the signed remainder has the
    same sign at both epsilons, so the "++" combination is the one to read for a
    clean run, and the "+-" and "-+" combinations are the ones that show a sign
    change between the two epsilons.

    A ratio of consecutive R2 values of 8 means a pure cubic. The ratios are returned
    too, because they are the quickest thing to read.

    Args:
        eps (list): the epsilons, in decreasing order.
        R2 (list): the second-order residuals at those epsilons.

    Returns:
        dict: "fits", a list of (C, D) for the four sign combinations of the two
            residuals, "ratios", the consecutive ratios R2[i] / R2[i+1], and "pair",
            the two epsilons used.
    """
    e1, e2 = eps[-2], eps[-1]
    r1, r2 = R2[-2], R2[-1]
    fits = []
    for s1 in (+1, -1):
        for s2 in (+1, -1):
            # Solve [e1^3 e1^2; e2^3 e2^2] [C; D] = [s1 r1; s2 r2].
            A = np.array([[e1**3, e1**2], [e2**3, e2**2]])
            b = np.array([s1 * r1, s2 * r2])
            C, D = np.linalg.solve(A, b)
            fits.append((float(C), float(D)))
    ratios = []
    for a, b in zip(R2[:-1], R2[1:]):
        ratios.append(float(a / b) if b > 0 else float("nan"))
    return {"fits": fits, "ratios": ratios, "pair": [e1, e2]}


def write_json(path, label, result, extra=None):
    """Write a Taylor result and its metadata to a JSON file.

    The tables in the log are for reading. The JSON is for the script that assembles
    the comparison across trajectory lengths, rheologies and firedrake builds, so that
    no number is copied by hand.

    Args:
        path (str or Path): the file to write. Overwritten.
        label (str): the run description printed at the top of the report.
        result (dict): the dictionary returned by ``taylor_remainders``.
        extra (dict, optional): metadata to store alongside, for example the command
            line, the rank count and the firedrake build.
    """
    import json

    payload = {"label": label, "result": result}
    payload["fit"] = two_term_fit(result["eps"], result["R2"])
    payload["rates"] = {key: rates(result[key]) for key in ("R0", "R1", "R2")}
    if extra:
        payload["extra"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def report(case, result):
    """Print the residuals, the local rates, and the size of the noise floor.

    The residuals are also printed relative to |J| at the expansion point, because
    that is what says whether a residual is real or is roundoff. Double precision
    puts a floor of about 1e-16 * |J| on any remainder, so a relative residual near
    1e-15 carries no information about the Hessian, whatever its rate looks like.

    Args:
        case (str): the objective term this result belongs to.
        result (dict): the dictionary returned by ``taylor_remainders``.
    """
    Jm = result["Jm"]
    floor = 1e-16 * abs(Jm)

    print(f"\n=== case: {case} ===")
    print(f"J(m)      = {Jm:.8e}   (roundoff floor on any residual is about {floor:.2e})")
    print(f"<dJ, h>   = {result['dJdm']:.8e}")
    print(f"<h, H h>  = {result['hHh']:.8e}")
    print(f"eps       = " + "  ".join(f"{e:.3e}" for e in result["eps"]))

    for key, expected in (("R0", 1), ("R1", 2), ("R2", 3)):
        res = result[key]
        rate = rates(res)
        print(f"{key} (expect order {expected})")
        print("  residual:  " + "  ".join(f"{r:.4e}" for r in res))
        print("  rel to J:  " + "  ".join(f"{r / abs(Jm):.2e}" for r in res))
        print("  rate:      " + "     ".join(f"{r:.3f}" for r in rate))
        # Flag the entries that have sunk into roundoff, so a rate computed from
        # them is not read as a statement about the Hessian.
        noise = ["*" if r < 100 * floor else " " for r in res]
        print("  at floor:  " + "     ".join(f"{n}    " for n in noise))

    # The two-term fit, which is the number to read instead of the R2 rate. See
    # two_term_fit for why.
    fit = two_term_fit(result["eps"], result["R2"])
    print("two-term fit R2 = |C eps^3 + D eps^2| on the two smallest eps")
    print("  ratios R2[i]/R2[i+1] (8 = pure cubic):  "
          + "  ".join(f"{r:.2f}" for r in fit["ratios"]))
    for (C, D), signs in zip(fit["fits"], ("++", "+-", "-+", "--")):
        rel = abs(D) / abs(result["hHh"]) if result["hHh"] != 0 else float("nan")
        print(f"  signs {signs}:  C = {C:+.4e}   D = {D:+.4e}   |D|/<h,Hh> = {rel:.2e}")

    # Wall-clock cost, which is the other half of what the TAO work needs to know.
    # Every call is listed, in order: the first includes compilation and warm-up,
    # the later ones are the steady per-iteration cost.
    t = result.get("timings", {})
    if t:
        print("timings (s), one entry per call in order")
        for key in ("functional", "derivative", "hessian"):
            if key in t:
                print(f"  {key:14s} " + "  ".join(f"{v:9.2f}" for v in t[key]))
        if "epsilon_sweep" in t:
            print(f"  {'epsilon_sweep':14s} {t['epsilon_sweep']:9.2f}")
        # The cost of a Hessian-vector product relative to a plain forward solve is
        # the number that decides whether Newton-Krylov can pay for itself. Quote the
        # steady values, that is the last call of each.
        if t.get("functional") and t.get("hessian") and t["functional"][-1] > 0:
            print(f"  steady hessian / functional     {t['hessian'][-1] / t['functional'][-1]:.2f}")
        if t.get("functional") and t.get("derivative") and t["functional"][-1] > 0:
            print(f"  steady derivative / functional  {t['derivative'][-1] / t['functional'][-1]:.2f}")




def hessian_vs_finite_difference(reduced_functional, m, h, epsilons=(1e-2, 1e-3, 1e-4)):
    """Check the Hessian-vector product against a finite difference of the gradient.

    This is independent of the Taylor expansion of J. The Hessian-vector product is
    the directional derivative of the gradient, so

        H h = lim_{eps -> 0} (grad J(m + eps h) - grad J(m)) / eps

    and the two sides can be compared directly, component by component, in the control
    space. That separates two things the Taylor test cannot separate:

    - If H h agrees with the finite difference to several digits, the second-order
      adjoint is right, and any failure of the R2 rate comes from the smoothness of J
      itself (an inexact forward solve, a kink in the rheology), not from the Hessian.
    - If H h disagrees, the second-order adjoint is genuinely wrong, and the size of
      the disagreement says by how much.

    The finite difference is itself only first-order accurate and carries a roundoff
    error that grows as 1/eps, so it has a best epsilon somewhere in the middle of the
    range. Several epsilons are used, and the best agreement over the range is what
    counts, not the agreement at the smallest one.

    Each epsilon costs one gradient evaluation, which is one forward sweep plus one
    adjoint sweep.

    Args:
        reduced_functional (ReducedFunctional): the functional to differentiate.
        m (Function): the expansion point in control space.
        h (Function): the direction.
        epsilons (tuple): the finite-difference step sizes to try.

    Returns:
        dict: with "Hh_norm", the L2 norm of the exact Hessian-vector product, and
            "rows", one entry per epsilon holding the norm of the finite difference,
            the norm of the difference between the two, and their relative difference.
    """
    with stop_annotating():
        # Order matters. pyadjoint needs the functional evaluated and the first-order
        # adjoint swept before the second-order adjoint has the adjoint inputs it
        # multiplies against; calling hessian() on a cold tape leaves tlm/adj values
        # unset and fails inside evaluate_hessian_component. This is the same order
        # taylor_remainders uses.
        reduced_functional(m)

        # The gradient at the expansion point, held for every finite difference.
        g0 = reduced_functional.derivative().riesz_representation(riesz_map="l2")

        # The exact product, via the tangent-linear and second-order adjoint sweeps.
        Hh = reduced_functional.hessian(h)
        Hh_vec = Hh.riesz_representation(riesz_map="l2")

        Hh_norm = float(np.sqrt(Hh_vec.dat.inner(Hh_vec.dat)))
        out = {"Hh_norm": Hh_norm, "rows": []}

        for eps in epsilons:
            # Gradient at the perturbed control.
            reduced_functional(m._ad_add(h._ad_mul(eps)))
            g1 = reduced_functional.derivative().riesz_representation(riesz_map="l2")

            # The one-sided difference quotient, which approximates H h.
            fd = g1.copy(deepcopy=True)
            fd -= g0
            fd /= eps

            diff = fd.copy(deepcopy=True)
            diff -= Hh_vec

            fd_norm = float(np.sqrt(fd.dat.inner(fd.dat)))
            diff_norm = float(np.sqrt(diff.dat.inner(diff.dat)))

            out["rows"].append({
                "eps": eps,
                "fd_norm": fd_norm,
                "diff_norm": diff_norm,
                "relative": diff_norm / Hh_norm if Hh_norm > 0 else float("nan"),
            })

    # Leave the functional evaluated at the expansion point.
    reduced_functional(m)

    return out


def report_fd(case, result):
    """Print the comparison of H h against the finite difference of the gradient.

    Args:
        case (str): the objective term this result belongs to.
        result (dict): the dictionary returned by ``hessian_vs_finite_difference``.
    """
    print(f"\n=== Hessian vs finite-difference gradient: {case} ===")
    print(f"||H h||            = {result['Hh_norm']:.8e}")
    print(f"{'eps':>10s}  {'||fd||':>14s}  {'||fd - H h||':>14s}  {'relative':>12s}")
    for row in result["rows"]:
        print(f"{row['eps']:10.3e}  {row['fd_norm']:14.6e}  "
              f"{row['diff_norm']:14.6e}  {row['relative']:12.3e}")
    # The smallest relative difference over the range is the meaningful number: the
    # finite difference is first-order accurate at large eps and noisy at small eps.
    best = min(row["relative"] for row in result["rows"])
    print(f"best relative agreement: {best:.3e}")
