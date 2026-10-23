r"""The Extended Burgers modulus and its fit by N parallel Maxwell elements.

GIAMIP Exp09 and Exp12 use the Extended Burgers model (EB) of Lau (2023)
and Ivins et al. (2022) in the mantle below 90 km. G-ADOPT represents a
linear rheology as N Maxwell elements in parallel, one internal variable per
element (`gadopt.CompressibleInternalVariableApproximation` with lists of
shear moduli and viscosities). This script finds the N elements that best
represent EB, in the normalised form of Lau et al. (2026, Text S1), and
writes them to `eb_fit.json` for the prepare scripts of Exp09 and Exp12.
The design is `NOTES/HANDOUT-TRANSIENT-NONLINEAR-DESIGN-2026-10-20.md`,
sections 5, 6 and 12.3.

THE NORMALISED EB MODULUS
    Times are in units of the local Maxwell time tau_M = eta_0 / mu_inf and
    moduli in units of the unrelaxed modulus mu_inf. With s' = tau_M s the
    Laplace variable (or s' = i omega tau_M for the complex modulus), the EB
    creep function in the Laplace domain gives the compliance

        J(s') = 1 + 1/s' + Delta * int_{tau_L}^{tau_H} D(tau) / (1 + s' tau) dtau

    with the absorption-band distribution

        D(tau) = alpha tau^(alpha - 1) / (tau_H^alpha - tau_L^alpha),

    and the modulus is mu_EB(s') = 1 / J(s'). The three terms are the
    instant elastic response (1), the steady-state dashpot of viscosity
    eta_0 (1/s'), and the anelastic band of strength Delta. Lau (2023)
    Eq. 20 as printed has the wrong sign in front of s'^2 B(s'): its short-
    time limit is about 1 / (1 + 2 Delta) instead of 1. This script uses the
    compliance above (design handout section 5, item 1).

    The substitution x = tau^alpha turns the band integral into

        1 / (x_H - x_L) * int_{x_L}^{x_H} dx / (1 + s' x^(1/alpha)),

    whose integrand is smooth (the integrable singularity tau^(alpha - 1) of
    D is gone), so adaptive quadrature reaches round-off. With tau_L = 0 it
    has the closed form 2F1(1, alpha; 1 + alpha; -s' tau_H) (Gauss
    hypergeometric function), which the tests compare with the quadrature.

THE GENERALISED MAXWELL MODEL
    N Maxwell elements in parallel, element j with modulus mu_j and Maxwell
    time tau_j = eta_j / mu_j, have the modulus

        mu_N(s') = sum_j mu_j s' tau_j / (1 + s' tau_j).

    Its short-time limit is sum_j mu_j (the instant modulus of the run) and
    its long-time limit is s' sum_j mu_j tau_j, a dashpot of viscosity
    sum_j eta_j. The long-term viscosity must be eta_0 of the Maxwell twin
    (design handout section 5, item 3), so in normalised form

        sum_j mu_j tau_j = 1.

THE FIT (Lau 2024, Sect. 3.3 and 4.2.1, Eqs 27-29 and 33)
    The tau_j are fixed, log-spaced between TAU_MIN = 1e-2 and 1 + Delta
    (design handout section 6.3). The mu_j then enter linearly, and the fit
    is a linear least-squares problem: the real and the imaginary parts of
    mu_N(i omega) - mu_EB(i omega) at N_FREQ = 1000 frequencies, log-spaced
    in omega tau_M from 1e-1 to 1e2, with absolute (unweighted) residuals,
    subject to the equality constraint sum_j mu_j tau_j = 1. The constrained
    problem is solved exactly through its KKT system (`constrained_lsq`), so
    the constraint holds to round-off. An element whose modulus comes out
    below MU_FLOOR = 1e-6 (or negative) is dropped and the fit is repeated
    with the remaining elements: a zero modulus gives the Maxwell time
    eta_j / mu_j = 0 / 0 in the solver.

    The fit band covers the times that the GIAMIP runs resolve. The upper
    end 1 + Delta is the relaxation time of the band's long end (the
    modulus there is about 1 / (1 + Delta)); an upper end of 10 makes the
    constrained fit miss by 34-55 % near omega tau_M = 0.3-0.5 (handout
    section 6.3).

    ERROR MEASURES. The headline error is the largest relative error of the
    complex modulus over the fit frequencies, max |mu_N - mu_EB| / |mu_EB|.
    The JSON and the plot also give the 2-norm relative error over the band
    and the largest relative error of the imaginary part alone (the
    dissipation). With 3 or 4 elements on a fixed grid the largest error
    depends strongly on the upper end of the grid: on the grid end 2.9 of
    the handout this code gives 3.2 % (N = 3) and 3.4 % (N = 4); on the grid
    end 3.0 (`--tau_max 3.0`) it gives 4.1 % and 1.7 %, the numbers quoted in
    handout section 6.3. The ranking of N = 3 and N = 4 changes with it.

    FREE ELEMENT TIMES. For comparison only, `fit_free` also optimises the
    element times (log tau_j, Nelder-Mead on the largest relative error, the
    sum constraint kept). The prepare scripts use the `selected` fit, which
    is a fixed-grid fit unless `--select_free` is given.

DIMENSIONAL ELEMENTS
    In a cell with unrelaxed modulus mu_inf and Maxwell-twin viscosity eta_0,
    element j has mu_j = mu_inf * mu_bar_j and eta_j = eta_0 * mu_bar_j *
    tau_bar_j, where (mu_bar_j, tau_bar_j) are the normalised values of the
    JSON file. The sum of the eta_j is eta_0 because of the constraint.

USAGE
    python eb_fit.py [--n 3 4] [--select 3] [--tau_max 2.9] [--select_free]
                     [--output_dir .]

    Writes `eb_fit.json` (the selected fit, every fixed-grid fit and every
    free-time fit) and `eb_fit.png`: the real and imaginary parts of the
    modulus and their relative errors against omega tau_M (with the 10 yr
    step of the midpoint upper and lower mantle marked), and the relaxation
    modulus in the time domain against t / tau_M. Sia chooses N from the
    plot (decision S17); the prepare scripts read the `selected` fit.
"""

import argparse
import hashlib
import json
import os

import numpy as np
from scipy import integrate, special

#: The EB parameters of Exp09 and Exp12 (design handout section 6.2):
#: the band exponent (Jackson & Faul 2010), the band strength (Lau & Faul
#: 2019, lower mantle, applied to the whole mantle below 90 km, decision
#: S22) and the upper end of the band in units of tau_M (decision S22).
ALPHA = 0.274
DELTA = 1.9
TAU_H = 1.0
#: The lower end of the band, units of tau_M: 0, the limit of Lau (2023)
#: Eq. 21 (design handout section 6.3). The handout suggests 1e-12 for a
#: quadrature in tau, where D(tau) is singular at 0. The quadrature here is
#: in x = tau^alpha, where the integrand is regular at x = 0, so the exact
#: lower end is used. A lower end of 1e-12 would change the band integral by
#: a relative amount of about x_L / x_H = 1e-12^0.274 = 5e-4 (the tests
#: check this size).
TAU_L = 0.0

#: The fixed element times of the fit: log-spaced from TAU_MIN to 1 + DELTA.
TAU_MIN = 1.0e-2
#: The frequencies of the fit: N_FREQ values of omega tau_M, log-spaced from
#: OMEGA_MIN to OMEGA_MAX.
N_FREQ = 1000
OMEGA_MIN = 1.0e-1
OMEGA_MAX = 1.0e2
#: The smallest admissible element modulus (units of mu_inf).
MU_FLOOR = 1.0e-6


def band_integral_quadrature(s, alpha=ALPHA, tau_h=TAU_H, tau_l=TAU_L):
    """The band integral int D(tau) / (1 + s tau) dtau by adaptive quadrature.

    After the substitution x = tau^alpha the integral is

        1 / (x_H - x_L) * int_{x_L}^{x_H} dx / (1 + s x^(1/alpha)),

    with a smooth integrand. The real and imaginary parts are integrated
    separately for a complex `s`.

    Args:
      s: the normalised Laplace variable s' (real or complex scalar).
      alpha, tau_h, tau_l: the band exponent and its ends (units of tau_M).

    Returns:
      The integral, real or complex like `s`.
    """
    x_l, x_h = tau_l ** alpha, tau_h ** alpha

    def f(x):
        return 1.0 / (1.0 + s * x ** (1.0 / alpha))

    options = dict(epsabs=0.0, epsrel=1e-13, limit=500)
    re = integrate.quad(lambda x: np.real(f(x)), x_l, x_h, **options)[0]
    if np.iscomplexobj(s):
        im = integrate.quad(lambda x: np.imag(f(x)), x_l, x_h, **options)[0]
        return (re + 1j * im) / (x_h - x_l)
    return re / (x_h - x_l)


def band_integral_closed(s, alpha=ALPHA, tau_h=TAU_H):
    """The band integral for tau_L = 0 in closed form.

    int_0^{tau_H} alpha tau^(alpha-1) / (tau_H^alpha (1 + s tau)) dtau
    = 2F1(1, alpha; 1 + alpha; -s tau_H), from the Euler integral of the
    Gauss hypergeometric function (substitute tau = tau_H t).
    """
    return special.hyp2f1(1.0, alpha, 1.0 + alpha, -s * tau_h)


def mu_eb(s, alpha=ALPHA, delta=DELTA, tau_h=TAU_H, tau_l=TAU_L,
          closed=True):
    """The normalised EB modulus mu_EB(s') = 1 / J(s').

    Args:
      s: s' = tau_M s, a scalar or an array, real (Laplace domain) or complex
        (i omega tau_M for the complex modulus).
      alpha, delta, tau_h, tau_l: the EB parameters (times in units of tau_M).
      closed: use the hypergeometric closed form (only for tau_l = 0); else
        the quadrature.

    Returns:
      mu_EB in units of mu_inf, the shape of `s`.
    """
    s = np.asarray(s)
    if closed and tau_l == 0.0:
        band = band_integral_closed(s, alpha, tau_h)
    else:
        band = np.vectorize(
            lambda v: band_integral_quadrature(v, alpha, tau_h, tau_l),
            otypes=[complex if np.iscomplexobj(s) else float])(s)
    # Compliance: elastic 1, dashpot 1/s', anelastic band of strength Delta.
    compliance = 1.0 + 1.0 / s + delta * band
    return 1.0 / compliance


def mu_maxwell(s, mu, tau):
    """The modulus of parallel Maxwell elements, sum_j mu_j s tau_j / (1 + s tau_j)."""
    s = np.asarray(s)[..., None]
    mu, tau = np.asarray(mu, dtype=float), np.asarray(tau, dtype=float)
    return np.sum(mu * s * tau / (1.0 + s * tau), axis=-1)


def constrained_lsq(A, b, c, d):
    """Minimise ||A x - b||^2 subject to c . x = d, exactly, by the KKT system.

        [ 2 A^T A   c ] [ x      ]   [ 2 A^T b ]
        [ c^T       0 ] [ lambda ] = [ d       ]

    The system is small (N + 1 rows); `numpy.linalg.lstsq` solves it, so a
    nearly singular A^T A (close element times) still gives the minimum-norm
    answer, and the constraint row is satisfied to round-off.

    Args:
      A: the (m, n) matrix, b: the (m,) right-hand side, c: the (n,)
        constraint row, d: the constraint value.

    Returns:
      x, shape (n,).
    """
    n = A.shape[1]
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = 2.0 * A.T @ A
    K[:n, n] = c
    K[n, :n] = c
    rhs = np.concatenate([2.0 * A.T @ b, [d]])
    return np.linalg.lstsq(K, rhs, rcond=None)[0][:n]


def frequencies(n_freq=N_FREQ, omega_min=OMEGA_MIN, omega_max=OMEGA_MAX):
    """The normalised frequencies omega tau_M of the fit, log-spaced."""
    return np.logspace(np.log10(omega_min), np.log10(omega_max), n_freq)


def errors(mu, tau, target, omega):
    """The error measures of a fit against the target at the frequencies.

    Returns:
      A dict: `max_rel_error` (max |mu_N - mu_EB| / |mu_EB|) and the
      frequency where it is (`max_rel_error_omega`), `l2_rel_error`
      (||mu_N - mu_EB|| / ||mu_EB|| over the frequencies), and
      `max_rel_error_imag` (max |Im mu_N - Im mu_EB| / |Im mu_EB|).
    """
    model = mu_maxwell(1j * omega, mu, tau)
    diff = model - target
    rel = np.abs(diff) / np.abs(target)
    k = int(np.argmax(rel))
    return {"max_rel_error": float(rel[k]),
            "max_rel_error_omega": float(omega[k]),
            "l2_rel_error": float(np.linalg.norm(diff)
                                  / np.linalg.norm(target)),
            "max_rel_error_imag": float(np.max(np.abs(diff.imag)
                                               / np.abs(target.imag)))}


def fit(n=None, tau=None, alpha=ALPHA, delta=DELTA, tau_h=TAU_H, omega=None,
        tau_max=None):
    """Fit parallel Maxwell elements to EB (Lau 2024 with the sum constraint).

    Args:
      n: the number of elements on the log grid from TAU_MIN to `tau_max`.
      tau: an explicit list of normalised element times instead of `n`.
      alpha, delta, tau_h: the EB parameters.
      omega: the normalised frequencies, or None for `frequencies()`.
      tau_max: the upper end of the grid, default 1 + delta.

    Returns:
      A dict: `n`, `tau_bar`, `mu_bar` (lists), `tau_bar_max` (the grid
      end, None for an explicit grid), `sum_mu_bar` (the instant modulus of
      the fit), `sum_mu_tau` (the normalised long-term viscosity, 1 to
      round-off), the measures of `errors`, and `dropped` (element times
      removed by the floor).

    Raises:
      ValueError: if neither or both of `n` and `tau` are given.
      RuntimeError: if every element falls below the floor.
    """
    if (n is None) == (tau is None):
        raise ValueError("give either n or tau")
    omega = frequencies() if omega is None else np.asarray(omega)
    grid_end = None
    if tau is None:
        grid_end = float(1.0 + delta if tau_max is None else tau_max)
        tau = np.logspace(np.log10(TAU_MIN), np.log10(grid_end), n)
    tau = [float(t) for t in tau]
    s = 1j * omega
    target = mu_eb(s, alpha, delta, tau_h)
    dropped = []
    while True:
        if not tau:
            raise RuntimeError("every element fell below the modulus floor")
        t = np.asarray(tau)
        # Column j: the modulus of element j with unit modulus, split into
        # real and imaginary rows (absolute misfit, Lau 2024 Eq. 28).
        column = (s[:, None] * t) / (1.0 + s[:, None] * t)
        A = np.vstack([column.real, column.imag])
        b = np.concatenate([target.real, target.imag])
        mu = constrained_lsq(A, b, t, 1.0)
        if not (mu < MU_FLOOR).any():
            break
        # Drop the element with the smallest modulus and fit again.
        worst = int(np.argmin(mu))
        dropped.append(tau.pop(worst))
    return {"n": len(tau), "tau_bar": t.tolist(), "mu_bar": mu.tolist(),
            "tau_bar_max": grid_end,
            "sum_mu_bar": float(mu.sum()),
            "sum_mu_tau": float(np.dot(mu, t)),
            **errors(mu, t, target, omega),
            "dropped": dropped}


def fit_free(n, alpha=ALPHA, delta=DELTA, tau_h=TAU_H, omega=None):
    """Fit n elements with free element times, for comparison with the grid fit.

    The moduli are the constrained least-squares moduli of `fit` for given
    times; the times (as log10 tau_j, sorted) are optimised by Nelder-Mead
    to minimise the largest relative error over the band, starting from the
    log grid of `fit(n)`. A set of times for which the floor drops an
    element counts as a bad point.

    Returns:
      The dict of `fit` at the optimised times, with `free_times` True.
    """
    from scipy import optimize

    omega = frequencies() if omega is None else np.asarray(omega)
    start = np.log10(fit(n, alpha=alpha, delta=delta, tau_h=tau_h,
                         omega=omega)["tau_bar"])

    def cost(logtau):
        try:
            f = fit(tau=10.0 ** np.sort(logtau), alpha=alpha, delta=delta,
                    tau_h=tau_h, omega=omega)
        except RuntimeError:
            return 1e3
        return f["max_rel_error"] + (1e3 if f["dropped"] else 0.0)

    best = optimize.minimize(cost, start, method="Nelder-Mead",
                             options={"xatol": 1e-6, "fatol": 1e-9,
                                      "maxiter": 4000})
    out = fit(tau=10.0 ** np.sort(best.x), alpha=alpha, delta=delta,
              tau_h=tau_h, omega=omega)
    out["free_times"] = True
    return out


def relaxation_eb(t, alpha=ALPHA, delta=DELTA, tau_h=TAU_H):
    """The EB relaxation modulus G(t) / mu_inf at times t (units of tau_M).

    G(t) is the inverse Laplace transform of mu_EB(s') / s' (the stress
    under a unit step of strain). It is computed with `mpmath.invertlaplace`
    (Talbot contour) and the closed form of the band integral, at 20 digits.
    """
    import mpmath as mp

    def laplace(s):
        band = mp.hyp2f1(1, alpha, 1 + alpha, -s * tau_h)
        return 1 / (s * (1 + 1 / s + delta * band))

    # 20 digits for this computation only; the global precision is kept.
    with mp.workdps(20):
        return np.array([float(mp.invertlaplace(laplace, float(x),
                                                method="talbot"))
                         for x in t])


def relaxation_maxwell(t, mu, tau):
    """The relaxation modulus of parallel Maxwell elements, sum_j mu_j exp(-t / tau_j)."""
    t = np.asarray(t)[:, None]
    return np.sum(np.asarray(mu) * np.exp(-t / np.asarray(tau)), axis=-1)


#: The midpoint Maxwell times of Exp09 (design handout section 3.1), yr, and
#: the shortest step of the runs (decision S18), yr. The plot marks where
#: the 10 yr step sits in the normalised frequency of each layer:
#: omega tau_M = tau_M / dt.
MIDPOINT_TAU_M_YR = {"upper mantle": 140.0, "lower mantle": 1600.0}
SHORTEST_STEP_YR = 10.0


def plot(fits, path, alpha=ALPHA, delta=DELTA, tau_h=TAU_H):
    """Plot target and fits: modulus against omega tau_M, relaxation against t / tau_M.

    Five panels: the real and the imaginary part of the modulus, the
    relative error of the complex modulus and of the imaginary part alone,
    over a wider range than the fit band (1e-3 to 1e4, band shaded, the
    element frequencies 1 / tau_j of the selected fits ticked, the 10 yr
    step of the midpoint upper and lower mantle marked), and the relaxation
    modulus G(t) / mu_inf against t / tau_M.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega = np.logspace(-3, 4, 1500)
    target = mu_eb(1j * omega, alpha, delta, tau_h)
    # The Maxwell twin (one element, mu = 1, tau = 1) for reference.
    twin = mu_maxwell(1j * omega, [1.0], [1.0])
    fig, axes = plt.subplots(5, 1, figsize=(8, 17))
    colours = plt.get_cmap("tab10").colors

    def style(f):
        return ":" if f.get("free_times") else "--"

    def label(f):
        kind = "free times" if f.get("free_times") else "grid"
        return f"N = {f['n']} ({kind})"

    for ax, part, name in ((axes[0], np.real, "Re mu / mu_inf"),
                           (axes[1], np.imag, "Im mu / mu_inf")):
        ax.semilogx(omega, part(target), "k-", lw=2.0, label="EB (target)")
        ax.semilogx(omega, part(twin), "k:", lw=1.0, label="Maxwell twin")
        for i, f in enumerate(fits):
            ax.semilogx(omega, part(mu_maxwell(1j * omega, f["mu_bar"],
                                               f["tau_bar"])),
                        style(f), color=colours[i % 10], lw=1.5,
                        label=label(f))
        ax.set_ylabel(name)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"EB alpha = {alpha}, Delta = {delta}, tau_H = "
                      f"{tau_h} tau_M; fit band shaded")
    for i, f in enumerate(fits):
        model = mu_maxwell(1j * omega, f["mu_bar"], f["tau_bar"])
        axes[2].loglog(omega, np.abs(model - target) / np.abs(target),
                       style(f), color=colours[i % 10],
                       label=f"{label(f)}: max {100 * f['max_rel_error']:.1f} "
                             f"%, 2-norm {100 * f['l2_rel_error']:.1f} %, "
                             f"sum mu {f['sum_mu_bar']:.3f}")
        axes[3].loglog(omega, np.abs(model.imag - target.imag)
                       / np.abs(target.imag), style(f), color=colours[i % 10],
                       label=f"{label(f)}: max in band "
                             f"{100 * f['max_rel_error_imag']:.1f} %")
        # The element frequencies 1 / tau_j of this fit.
        for tau in f["tau_bar"]:
            axes[2].axvline(1.0 / tau, color=colours[i % 10], lw=0.6,
                            ls=style(f), ymax=0.08)
    axes[2].set_ylabel("|mu_N - mu_EB| / |mu_EB|")
    axes[3].set_ylabel("|Im mu_N - Im mu_EB| / |Im mu_EB|")
    for ax in axes[:4]:
        ax.axvspan(OMEGA_MIN, OMEGA_MAX, color="0.9", zorder=0)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(omega[0], omega[-1])
        for name, tau_m in MIDPOINT_TAU_M_YR.items():
            x = tau_m / SHORTEST_STEP_YR
            ax.axvline(x, color="0.4", lw=1.0, ls="-.")
        ax.set_xlabel("omega tau_M")
    for name, tau_m in MIDPOINT_TAU_M_YR.items():
        axes[0].text(tau_m / SHORTEST_STEP_YR, 0.05,
                     f" 10 yr step, {name}\n (tau_M {tau_m:.0f} yr)",
                     fontsize=7, rotation=90, va="bottom",
                     bbox=dict(facecolor="white", edgecolor="none",
                               alpha=0.8))
    axes[2].legend(loc="lower left", fontsize=8)
    axes[3].legend(loc="lower left", fontsize=8)

    # The relaxation modulus in the time domain.
    t = np.logspace(-3, 1.5, 60)
    g_eb = relaxation_eb(t, alpha, delta, tau_h)
    axes[4].semilogx(t, g_eb, "k-", lw=2.0, label="EB")
    axes[4].semilogx(t, np.exp(-t), "k:", lw=1.0, label="Maxwell twin")
    for i, f in enumerate(fits):
        axes[4].semilogx(t, relaxation_maxwell(t, f["mu_bar"], f["tau_bar"]),
                         style(f), color=colours[i % 10], lw=1.5,
                         label=label(f))
    for name, tau_m in MIDPOINT_TAU_M_YR.items():
        axes[4].axvline(SHORTEST_STEP_YR / tau_m, color="0.4", lw=1.0,
                        ls="-.")
    axes[4].set_xlabel("t / tau_M (dash-dot: 10 yr in the midpoint upper "
                       "and lower mantle)")
    axes[4].set_ylabel("G(t) / mu_inf")
    axes[4].legend(loc="lower left", fontsize=8)
    axes[4].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sha256(path):
    """The SHA-256 of a file, hexadecimal."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main(argv=None):
    """Fit, write the JSON and the plot."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, nargs="+", default=[3, 4],
                   help="the numbers of elements to fit")
    p.add_argument("--select", type=int, default=3,
                   help="the N that the prepare scripts use (decision S17)")
    p.add_argument("--tau_max", type=float, default=1.0 + DELTA,
                   help="the upper end of the element-time grid, units of "
                        "tau_M (handout 6.3: 1 + Delta)")
    p.add_argument("--select_free", action="store_true",
                   help="select the free-time fit of N = --select instead of "
                        "the grid fit")
    p.add_argument("--no_free", action="store_true",
                   help="skip the free-time fits")
    p.add_argument("--output_dir", default=os.path.dirname(
        os.path.abspath(__file__)))
    args = p.parse_args(argv)
    if args.select not in args.n:
        raise SystemExit(f"--select {args.select} is not one of --n {args.n}")
    if args.select_free and args.no_free:
        raise SystemExit("--select_free needs the free-time fits")
    fits = [fit(n, tau_max=args.tau_max) for n in args.n]
    free = [] if args.no_free else [fit_free(n) for n in args.n]
    for f in fits + free:
        print(f"N = {f['n']}{' free' if f.get('free_times') else ''}: tau_bar "
              f"{np.round(f['tau_bar'], 5).tolist()}, mu_bar "
              f"{np.round(f['mu_bar'], 5).tolist()}, sum mu "
              f"{f['sum_mu_bar']:.4f}, sum mu tau {f['sum_mu_tau']:.15f}, "
              f"max rel error {100 * f['max_rel_error']:.2f} % at omega "
              f"tau_M = {f['max_rel_error_omega']:.3g}, 2-norm "
              f"{100 * f['l2_rel_error']:.2f} %, imaginary part "
              f"{100 * f['max_rel_error_imag']:.1f} %"
              f"{', dropped ' + str(f['dropped']) if f['dropped'] else ''}")
    os.makedirs(args.output_dir, exist_ok=True)
    png = os.path.join(args.output_dir, "eb_fit.png")
    plot(fits + free, png)
    pool = free if args.select_free else fits
    selected = next(f for f in pool if f["n"] == args.select)
    record = {
        "model": "Extended Burgers (Lau 2023; Ivins et al. 2022), "
                 "normalised: moduli in units of mu_inf, times in units of "
                 "tau_M = eta_0 / mu_inf",
        "parameters": {"alpha": ALPHA, "Delta": DELTA, "tau_H_over_tau_M":
                       TAU_H, "tau_L_over_tau_M": TAU_L},
        "fit": {"method": "Lau (2024) Eqs 27-29, 33: fixed log-spaced "
                          "element times, linear least squares on the real "
                          "and imaginary parts, absolute misfit, constraint "
                          "sum mu_bar tau_bar = 1 (KKT)",
                "tau_bar_min": TAU_MIN, "tau_bar_max": args.tau_max,
                "n_frequencies": N_FREQ, "omega_tau_M_min": OMEGA_MIN,
                "omega_tau_M_max": OMEGA_MAX, "mu_floor": MU_FLOOR,
                "error_measures": {
                    "max_rel_error": "max over the fit frequencies of "
                                     "|mu_N - mu_EB| / |mu_EB|",
                    "l2_rel_error": "||mu_N - mu_EB|| / ||mu_EB|| over the "
                                    "fit frequencies",
                    "max_rel_error_imag": "max over the fit frequencies of "
                                          "|Im mu_N - Im mu_EB| / "
                                          "|Im mu_EB|"}},
        "dimensionalise": "mu_j = mu_inf * mu_bar_j, eta_j = eta_0 * "
                          "mu_bar_j * tau_bar_j",
        "selected": selected,
        "fits": {str(f["n"]): f for f in fits},
        "free_time_fits": {str(f["n"]): f for f in free},
        "plot": os.path.basename(png),
        "script_sha256": sha256(os.path.abspath(__file__)),
    }
    path = os.path.join(args.output_dir, "eb_fit.json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"wrote {path} (selected N = {args.select}"
          f"{', free times' if args.select_free else ''}) and {png}")
    return record


if __name__ == "__main__":
    main()
