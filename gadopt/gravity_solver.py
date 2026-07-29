r"""Gravitational Poisson solver with Dirichlet-to-Neumann boundary treatment.

This module solves the gravitational Poisson equation

$$
  nabla^2 psi = -4 pi G rho,
$$

whose correct boundary condition lives at infinity: psi must match the decaying
free-space solution. On the finite domains G-ADOPT works with (annuli, spherical
shells and their extensions), that condition is imposed exactly (up to mode
truncation) through Dirichlet-to-Neumann (DtN) maps on circular/spherical
boundaries: each angular eigenmode of the boundary trace extends uniquely into
the source-free exterior (or interior core) as the decaying (regular) harmonic
solution, whose normal derivative is proportional to its own trace. The map
needs the trace of psi only - never the density - so the solver discovers the
mode content by itself. The trace coefficients are global scalar unknowns in
R spaces, defined by scalar constraint rows and fed back as boundary flux;
everything is a plain UFL form, so the system remains differentiable, adjoint-
capable, and embeddable in larger coupled residuals.

The DtN terms use the Robin-shifted formulation: the flux is split as

$$
  (d psi)/(d n) = -(alpha/R) psi + sum_k (alpha - lambda_k R)/R \, c_k e_k,
$$

which is algebraically identical for the treated modes but places a pointwise
Robin term in the psi-psi block, keeping it symmetric positive definite for the
Schur fieldsplit (the naive form leaves that block a singular pure-Neumann
Laplacian). Untreated modes above the truncation see Robin(alpha/R) - a
decaying far field - instead of a reflecting homogeneous Neumann condition.

Mass sheets sitting exactly on a DtN boundary (e.g. dynamic topography) enter
through the flux jump $[(d psi)/(d r)] = -4 pi G sigma$, which reduces to the
standard surface-source term $-4 pi G int sigma v \, ds$ with the DtN machinery
unchanged.

In 2-D the monopole needs one extra ingredient, because the exterior expansion
carries a logarithmic branch $C_0 ln r$ whose coefficient the boundary trace
does not determine. The DtN map is complete for every $m >= 1$ and blind at
$m = 0$, so a 2-D exterior boundary additionally receives the flux datum

$$
  (d psi)/(d n)|_{m = 0} = -2 G M / R,
$$

with $M$ the total enclosed mass. That leaves the additive constant free -
there is no "psi -> 0 at infinity" to fix it - and the Robin shift above fixes
it for us: testing the residual with $v = 1$, the volume source and the datum
cancel exactly and leave $oint psi \, ds = 0$ on the boundary. So the system
stays nonsingular with no nullspace handling, and the potential is returned in
the gauge $oint_{dB} psi \, ds = 0$, i.e. $psi = -2 G M ln(r/R)$ outside the
sources. Only `grad psi` is gauge-independent; anyone comparing against a
closed form that fixes the constant differently will see a constant offset.
3-D has no analogue - the $l = 0$ exterior decays as $1/r$ and the trace map
reproduces it exactly - so nonzero net mass has always been legal there.

Users instantiate the `GravitySolver` class by providing a solution function,
the density, and boundary conditions keyed by boundary marker, then call the
`solve` method to update the solution.
"""

import abc
from collections import namedtuple
from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np
from firedrake import *
from firedrake.ufl_expr import extract_unique_domain

from firedrake.petsc import PETSc
from mpi4py import MPI

from .dtn_adjoint import annotate_lowrank_solve, taped_trace_coefficients
from .dtn_lowrank import (
    LowRankDtNOperator, apply_dirichlet_to_rows, boundary_facet_cellname,
    build_boundary_mode_rows, supports_trace_build, tabulate_boundary_modes,
    validation_keys)
from .solver_options_manager import ConfigType, SolverConfigurationMixin
from .spherical_harmonics import real_spherical_harmonic
from .utility import CombinedSurfaceMeasure, DEBUG, INFO, ensure_constant, log_level

__all__ = ["CylindricalDtN", "SphericalDtN", "GravitySolver"]


#: For the one-by-one systems on a `Real` space that define the enclosed-mass
#: scalars. The operator is a nonzero number, so `preonly` + Jacobi is an exact
#: solve rather than a preconditioning choice, and it must not inherit the
#: gravity solver's own fieldsplit options.
real_scalar_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

direct_gravity_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Direct solve for the psi (potential) block.

MUMPS LU. Exact and robust, but the sparse factors of a 3-D volume operator
suffer fill-in that outgrows the matrix, so this is the default only in 2-D
(and for small 3-D problems selected explicitly). See `iterative_gravity_solver_parameters`
for the scalable alternative.
"""

iterative_gravity_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "pc_type": "python",
    "pc_python_type": "gadopt.SPDAssembledPC",
    "assembled_pc_type": "gamg",
    "assembled_mg_levels_pc_type": "sor",
    "assembled_pc_gamg_threshold": 0.01,
    "assembled_pc_gamg_square_graph": 100,
    "assembled_pc_gamg_coarse_eq_limit": 1000,
    "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
}
"""Iterative solve for the psi (potential) block.

Conjugate gradients preconditioned by algebraic multigrid (GAMG). The Robin
shift makes the psi block strictly SPD, so `gadopt.SPDAssembledPC` sets the
PETSc MAT_SPD flag (CG eigen-estimates in the Chebyshev and GAMG setup) and no
near-nullspace is needed. These are the same parameters G-ADOPT's 3-D spherical
Stokes solver uses for its velocity block, which inverts a strictly harder SPD
operator on the same extruded cubed-sphere meshes at production scale. AMG has
O(N) memory with no fill-in, so it replaces MUMPS as the default psi solver in
3-D.
"""


def _flatten_options(options, prefix=""):
    """PETSc-style flat options from the nested G-ADOPT dictionary."""
    flat = {}
    for key, value in options.items():
        if isinstance(value, Mapping):
            flat.update(_flatten_options(value, prefix + key + "_"))
        else:
            flat[prefix + key] = value
    return flat


lowrank_gravity_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-11,
    "ksp_max_it": 1000,
    "pc_type": "gamg",
    "mg_levels_pc_type": "sor",
    "pc_gamg_threshold": 0.01,
    "pc_gamg_square_graph": 100,
    "pc_gamg_coarse_eq_limit": 1000,
    "pc_gamg_mis_k_minimum_degree_ordering": True,
}
"""CG on `A + B`, algebraic multigrid built from the Robin-shifted `A` alone.

The same GAMG settings the multiplier path gives its `fieldsplit_0`, applied
without the `AssembledPC` indirection because here `A` is already an assembled
`aij` matrix and PETSc can be handed it directly as `Pmat`. `ksp_rtol` matches
the multiplier path's OUTER tolerance, not its inner one: there is no outer
Krylov cleaning up after this solve, so this rtol is the solution accuracy.

Note when comparing iteration counts against the multiplier path: its
`fieldsplit_0` runs at `ksp_rtol` 1e-8, so a comparison against this preset's
1e-11 is not like-for-like. Pinned equal, both paths take 8 CG iterations.

PETSc will report

    Option left: name:-Gravity_lowrank_mg_levels_pc_type value: sor

and it is a **false alarm**: the option is in effect. Interrogating the
preconditioner directly shows the level smoother is `chebyshev/sor` as
requested, and forcing anything else makes it worse (11 CG iterations as
shipped, 15 with `chebyshev/jacobi`, 15 with `richardson/sor`). A genuinely
unused option looks identical to this one, and both natural responses to the
warning - "fixing" it, or discounting a comparison because of it - are wrong.
"""


QuadratureCalibration = namedtuple(
    "QuadratureCalibration",
    ["intercept", "slope", "max_distortion", "max_resolution"])
"""The fitted rule, and the range of meshes it was fitted on.

`intercept` and `slope` give `q = intercept + slope * (L h_max / R)`.
`max_distortion` (`h_max / h_mean`) and `max_resolution` (`L h_max / R`) record
how far the calibration set actually reached, so the solver can say when it is
extrapolating instead of interpolating. Outside those the rule is a guess, and
`GravitySolver.warn_on_quadrature_rule_limits` says so rather than pretending.

Both bounds are the campaign's own maxima, and both must be read with
`boundary_facet_scale`'s definition of `h_mean` - see the note there, because
computing it the other way round understates the distortion and lets meshes the
rule cannot integrate past the bound. The interval bound is 1.65 rather than the
measured 1.621 so that the fitted rows themselves do not trip it; a rule that
declares itself to be extrapolating on its own calibration data is a rule nobody
will keep listening to.

**A scalar bound cannot express the shape of the calibration set.** The 3-D
campaign swept warp and refinement together, and `max_distortion` is the maximum
over both, so a coarse mesh warped beyond anything measured can still report a
distortion below a bound that a much finer, less warped mesh set. Measured, that
leaves a residual: a level-2 shell at warp 0.20 with `L = 8` misses the 1e-8
target by 1.9x while passing both bounds. Closing it properly needs a bound in
the mesh-quality variable itself rather than in a scalar summary of it.
"""

QUADRATURE_RULE_COEFFICIENTS = {
    "interval": QuadratureCalibration(6.0, 2.5, 1.65, 4.9),
    "quadrilateral": QuadratureCalibration(8.0, 2.0, 1.9, 11.1),
}
r"""Calibrated `(intercept, slope)` of the boundary quadrature rule, by facet.

The degree a boundary rule needs is set by how much of one oscillation falls
inside one facet - `L h / R` - and not by `L`, which is what the incumbent
default `2 (max_mode + element_degree)` uses. Keyed by the reference cell of a
boundary facet rather than by dimension, because the requirement is a property
of the rule on that cell: interval facets carry a plain Gauss rule, quadrilateral
ones its tensor product.

Measured on meshes chosen so that no facet-to-facet error cancellation survives
- a composite Gauss rule on a *regular* boundary annihilates every harmonic but
multiples of the facet count, which is worth ten orders on a uniform annulus and
about one quadrature point on a pristine cubed sphere, and calibrating on those
would ship a rule that under-integrates everywhere else. 45 3-D configurations
and 30 2-D ones, target 1e-8 self-convergence, then confirmed against the
integrand the solver actually assembles. Derivation, the intercept decision and
the element-degree measurements are in `NOTES/FINDING-QUADRATURE-DEGREE-FORM.md`;
the mechanism that invalidated the earlier calibration is in
`NOTES/FINDING-QUADRATURE-CANCELLATION.md`.

Simplex facets are deliberately absent: a collapsed Gauss-Jacobi rule on a
triangle is not a tensor Gauss rule, so neither constant transfers, and an
absent entry falls back to the incumbent default rather than guessing.
"""

QUADRATURE_ELEMENT_DEGREE_COST = 2
"""Extra boundary degrees per element degree above CG1.

Measured increments over 21 configurations: CG1 to CG2 gives
`+2, +2, +2, 0, 0, +2, 0` and CG2 to CG3 gives `+2, 0, 0, +2, +2, 0, +2`. The
envelope is +2 at both steps with no sign of growth. It has to be explicit
rather than absorbed into the intercept: the CG1 margin does not cover it, and
CG3 breaks the bare rule by two degrees on two of the seven meshes.
"""


QuadratureRuleReport = namedtuple(
    "QuadratureRuleReport",
    ["degree", "requested", "incumbent", "resolution", "distortion",
     "calibration"])
"""How `GravitySolver` arrived at its boundary quadrature degree.

Fields:
  degree: The degree actually used, i.e. `min(requested, incumbent)`.
  requested: What the calibrated rule asked for, before the cap.
  incumbent: `2 (max_mode + element_degree)`, the rule this replaced.
  resolution: `max(L h_max / R)` over the DtN boundaries - oscillations per
    facet, the variable the rule is written in.
  distortion: `max(h_max / h_mean)`, how uneven the boundary facets are.
  calibration: The `QuadratureCalibration` used, or None where the facet type
    is uncalibrated and the incumbent was taken unchanged.
"""


DtNMode = namedtuple("DtNMode", ["key", "expr", "lam", "norm", "scale"])
"""One angular eigenmode of a DtN boundary.

Fields:
  key: label used in coefficient reporting (e.g. "cos3", "Y2,-1", "mean").
  expr: UFL expression for the boundary eigenfunction e_k.
  lam: DtN eigenvalue, i.e. (d psi)/(d r) = -lam psi for the exterior branch
    of this mode (lam >= 0; the interior branch enters with the same sign in
    the weak form).
  norm: analytic value of the boundary integral of e_k^2.
  scale: norm divided by the analytic boundary measure - the constraint-row
    scaling that makes the multiplier equal the trace coefficient of e_k.
"""


class BaseDtN(abc.ABC):
    """Base class for Dirichlet-to-Neumann boundary condition descriptors.

    A DtN descriptor carries the geometry-specific data of the map: the angular
    eigenfunctions of its boundary type, the per-mode DtN eigenvalues on each
    side (exterior map for a boundary enclosing the sources, interior map for a
    boundary enclosing a source-free core), and the analytic normalisation
    constants. Orientation and radius are not part of the descriptor: the
    solver measures both from the marked boundary itself.
    """

    #: Geometric dimension the descriptor applies to.
    dim: int

    @property
    @abc.abstractmethod
    def max_degree(self) -> int:
        """Highest angular degree treated; drives the quadrature default."""

    @abc.abstractmethod
    def n_multipliers(self, side: str) -> int:
        """Number of R-space multipliers on a boundary of the given side."""

    @abc.abstractmethod
    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        """Mode table on a boundary of radius R with coordinates X.

        Args:
          side: "exterior" (sources inside the boundary) or "interior"
            (source-free core inside the boundary).
          R: Boundary radius, as measured by the solver.
          X: Coordinate vector of the mesh.
        """

    def mode_metadata(self, side: str, R: float) -> list[DtNMode]:
        """The mode table with `expr` left as None.

        Everything the low-rank path needs about a mode except its symbolic
        expression: the key, the DtN eigenvalue, the analytic normalisation and
        the constraint-row scaling. The low-rank build gets its values
        numerically from `gadopt.dtn_tabulate`, so it has no use for the
        expressions and does not build them.

        That used to be a large saving and is now a small one, which is worth
        recording rather than leaving as a stale rationale.
        `real_spherical_harmonic` ran sympy once per `(l, |m|)` to generate
        Horner coefficients; this docstring recorded about 0.015 s per mode,
        i.e. 13 s at `L = 20`, hidden inside the constructor. It no longer runs
        sympy at all - it builds an upward recursion from float constants - and
        the cost has collapsed. Measured here, expression construction alone
        for all 441 modes of `L = 20`: 0.088 s with the recursion (0.20
        ms/mode) against 0.697 s for the pre-fix path with a cold sympy cache
        (1.58 ms/mode) and 0.099 s warm.

        Note that the 1.58 ms/mode I measure for the pre-fix path is an order
        of magnitude below the 0.015 s recorded above, which I could not
        reproduce; it may have been measured under different conditions. Either
        way the conclusion is unchanged in direction and much weaker in size,
        so skipping the expressions is now a tidiness argument rather than a
        performance one.

        The default builds the full table and discards the expressions, which
        is correct but pointless; subclasses override it.
        """
        return [mode._replace(expr=None) for mode in self.modes(side, R, None)]

    def modes_by_key(self, keys, side: str, R: float, X) -> list[DtNMode]:
        """Build only the named modes, with their UFL expressions.

        Lets a caller that needs a fixed handful - the low-rank build's
        self-assertion needs three - pay for three rather than for the `O(L)`
        the sampled table would cost.
        """
        wanted = set(keys)
        return [mode for mode in self.modes(side, R, X) if mode.key in wanted]

    def check_modes(self, side: str, R: float, X) -> list[DtNMode]:
        """The subset of `modes` that `check_boundary_quadrature` samples.

        Subclasses override this to return only the modes whose boundary
        integrals bound the rest, and must *build* only those - filtering a
        fully constructed table would save nothing, since building the table
        is the cost. The default is every mode.
        """
        return self.modes(side, R, X)


class CylindricalDtN(BaseDtN):
    r"""DtN map on a circular boundary of a 2-D (cylindrical) domain.

    Azimuthal modes $e_m in {cos(m phi), sin(m phi)}$ for $m = 1 dots M$ are
    treated exactly: the exterior harmonic branch $r^(-m)$ gives
    $(d psi)/(d r) + (m/R) psi = 0$ and the interior branch $r^(+m)$ gives
    $(d psi)/(d r) - (m/R) psi = 0$. Content above the truncation is
    $O((r_s/R)^(M+1))$ for sources at radius $r_s$ and sees the Robin shift (a
    decaying far field) instead.

    The 2-D monopole is exceptional: the exterior $m = 0$ solution is
    logarithmic, so the trace does not determine the flux. Under the
    zero-net-mass assumption (which the solver verifies) the Robin shift is
    exact for it; on an interior boundary the exact monopole condition is
    homogeneous Neumann, recovered by the $m = 0$ mean multiplier.

    Args:
      M: Mode truncation; modes 1..M are treated exactly on this boundary.
    """

    dim = 2

    def __init__(self, M: int):
        if M < 0:
            raise ValueError(f"Require M >= 0, got M={M}")
        self.M = M

    @property
    def max_degree(self) -> int:
        return max(self.M, 1)

    def n_multipliers(self, side: str) -> int:
        return 2 * self.M + (1 if side == "interior" else 0)

    def _mean_mode(self, R: float) -> DtNMode:
        # The m = 0 mean multiplier undoing the Robin shift: the exact
        # interior monopole condition is homogeneous Neumann (lam = 0).
        return DtNMode("mean", 1.0, 0.0, 2 * np.pi * R, 1.0)

    def _azimuthal_modes(self, m: int, R: float, X) -> list[DtNMode]:
        phi = atan2(X[1], X[0])
        lam = m / R
        return [DtNMode(f"cos{m}", cos(m * phi), lam, np.pi * R, 0.5),
                DtNMode(f"sin{m}", sin(m * phi), lam, np.pi * R, 0.5)]

    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        table = [self._mean_mode(R)] if side == "interior" else []
        for m in range(1, self.M + 1):
            table.extend(self._azimuthal_modes(m, R, X))
        return table

    def mode_metadata(self, side: str, R: float) -> list[DtNMode]:
        table = [self._mean_mode(R)] if side == "interior" else []
        for m in range(1, self.M + 1):
            lam = m / R
            table.append(DtNMode(f"cos{m}", None, lam, np.pi * R, 0.5))
            table.append(DtNMode(f"sin{m}", None, lam, np.pi * R, 0.5))
        return table

    def modes_by_key(self, keys, side: str, R: float, X) -> list[DtNMode]:
        orders = sorted({int(key[3:]) for key in keys
                         if key.startswith(("cos", "sin"))})
        out = [self._mean_mode(R)] if "mean" in keys else []
        for m in orders:
            if 1 <= m <= self.M:
                out.extend(mode for mode in self._azimuthal_modes(m, R, X)
                           if mode.key in keys)
        return out

    def check_modes(self, side: str, R: float, X) -> list[DtNMode]:
        """The constant (where present), the lowest order and the highest.

        `m = M` is the order the boundary mesh and the quadrature rule are
        least able to resolve; `m = 1` and the interior `mean` mode carry
        essentially none of that, so what they report is the mesh's discrete
        boundary measure. Both parities of each order are kept, since `cos` and
        `sin` sample the facets differently.

        On the straight-facet annulus these two are nearly the same number: the
        spread of the deviation across all modes was measured at 1.02 to 1.03,
        i.e. the geometric factor dominates completely. That is *not* true of
        the curved spherical boundary - see `SphericalDtN.check_modes`.

        `M = 0` is the degenerate pure-Robin case with no treated azimuthal
        mode at all, and the sampled set must stay empty (bar the interior
        mean) rather than fabricating an `m = 1` the solver never assembles.
        """
        table = [self._mean_mode(R)] if side == "interior" else []
        for m in sorted({m for m in (1, self.M) if 1 <= m <= self.M}):
            table.extend(self._azimuthal_modes(m, R, X))
        return table


class SphericalDtN(BaseDtN):
    r"""DtN map on a spherical boundary of a 3-D domain.

    Real orthonormal spherical harmonics $Y_(l m)$ for $l = 0 dots L$,
    $-l <= m <= l$ are treated exactly: the exterior branch $r^(-(l+1))$ gives
    $(d psi)/(d r) + ((l+1)/R) psi = 0$ and the interior branch $r^(+l)$ gives
    $(d psi)/(d r) - (l/R) psi = 0$. Unlike 2-D there is no monopole
    exception: the $l = 0$ exterior decays as $1/r$, so the trace-based map is
    complete and no mass restriction applies.

    Args:
      L: Degree truncation; degrees 0..L are treated exactly on this boundary.
        The multiplier count is $(L+1)^2$, so keep L modest; content above the
        truncation decays as $O((r_s/R)^(L+2))$ and sees the Robin shift.
    """

    dim = 3

    def __init__(self, L: int):
        if L < 0:
            raise ValueError(f"Require L >= 0, got L={L}")
        self.L = L

    @property
    def max_degree(self) -> int:
        return max(self.L, 1)

    def n_multipliers(self, side: str) -> int:
        return (self.L + 1) ** 2

    def _mode(self, l: int, m: int, side: str, R: float, X) -> DtNMode:
        lam = (l + 1) / R if side == "exterior" else l / R
        return DtNMode(f"Y{l},{m}", real_spherical_harmonic(l, m, X),
                       lam, R**2, 1.0 / (4.0 * np.pi))

    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        return [self._mode(l, m, side, R, X)
                for l in range(self.L + 1) for m in range(-l, l + 1)]

    def mode_metadata(self, side: str, R: float) -> list[DtNMode]:
        table = []
        for l in range(self.L + 1):
            lam = (l + 1) / R if side == "exterior" else l / R
            for m in range(-l, l + 1):
                table.append(DtNMode(f"Y{l},{m}", None, lam, R**2,
                                     1.0 / (4.0 * np.pi)))
        return table

    def modes_by_key(self, keys, side: str, R: float, X) -> list[DtNMode]:
        parsed = []
        for key in keys:
            l, m = (int(part) for part in key[1:].split(","))
            if 0 <= l <= self.L and abs(m) <= l:
                parsed.append(self._mode(l, m, side, R, X))
        return parsed

    def check_modes(self, side: str, R: float, X) -> list[DtNMode]:
        """`Y_00` and the whole of degree `L`: `2L + 2` modes, not `(L+1)^2`.

        `Y_00` is constant, so its boundary integral reports the mesh's
        discrete boundary measure alone. The degree-`L` shell carries the
        mode-dependent part, and is taken in full rather than at the three
        orders `0, +-L`, because that shorter sample was measured to miss too
        much. On the level-2 cubed-sphere shell the worst deviation over all
        modes sits at `(3, -2)` for `L = 3` and `L = 5` and at `(6, 6)` for
        `L = 8`; against those, `{(0,0), (L,0), (L,+-L)}` reports 0.68, 0.44
        and 0.90 of the true worst, while the full degree-`L` shell reports
        1.00, 0.97 and 0.93.

        So this is a proxy, not a bound - no sample of size `O(L)` can be a
        bound, since the worst mode is not always at the highest degree - and
        `check_boundary_quadrature(sample="all")` remains the guarantee. What
        the proxy is measured to give is the right order of magnitude, which is
        what it is for. (It was sized for a constructor-time warning, which no
        longer exists - `__init__` warns structurally instead and never
        measures. On any mesh whose rule can be recovered, `sample="all"` costs
        two assemblies rather than one per mode, so the proxy now earns its
        keep only on simplex facets.)

        Not summed into a single form, which would look cheaper still: by the
        addition theorem `sum_m Y_lm^2 = (2l+1)/4pi` exactly, so a whole
        degree's sum of squares is a *constant*, integrated exactly by any rule
        that gets the area right. Such a check would pass on a mesh resolving
        none of its modes.
        """
        L = self.L
        sampled = [(0, 0)] + [(L, m) for m in range(-L, L + 1) if L > 0]
        return [self._mode(l, m, side, R, X) for l, m in sampled]


class GravitySolver(SolverConfigurationMixin):
    """Solver for the gravitational Poisson equation with DtN boundaries.

    Solves $nabla^2 psi = -4 pi G rho$ for the perturbation potential psi with
    the infinite-domain behaviour imposed through Dirichlet-to-Neumann maps on
    the boundaries specified in `bcs`. The solution function is updated in
    place by `solve`.

    The density may live on the solver mesh or on a `Submesh` of it (the
    extended-domain configuration, where psi extends beyond the region
    carrying the density); the cross-mesh coupling is set up automatically.

    **The 2-D gauge.** A 2-D problem with an exterior DtN boundary and nonzero
    net mass has a logarithmic far field and therefore no natural zero. The
    solver returns the potential normalised so that its mean over that boundary
    vanishes, i.e. `psi = -2 G M ln(r/R)` outside the sources with `M` the
    total enclosed mass. Only `grad psi` is gauge-independent, so a closed-form
    reference that fixes the constant some other way will disagree by an
    offset that is not an error. Two configurations are refused rather than
    solved in a gauge that cannot exist: a strong `psi` condition alongside a
    nonzero enclosed mass, and more than one exterior DtN boundary on the same
    2-D mesh. 3-D is unaffected throughout.

    Args:
      solution:
        Firedrake function for the potential, on a scalar function space
      rho:
        Density: a Firedrake function (on the solution mesh or a Submesh of
        it), a UFL expression, or a number
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      gravitational_constant:
        Value of G in the chosen non-dimensionalisation
      quad_degree:
        Integer denoting the quadrature degree applied on boundary integrals
        involving the DtN eigenfunctions. UFL's automatic estimate is useless
        for these - it adds a flat two degrees per elementary function, so
        `cos(m phi)` estimates the same at m = 1 and m = 40 - and the degree is
        therefore explicit. The default is mesh-aware: what decides whether a
        rule resolves a mode is how much of one oscillation falls inside one
        facet, `L h / R`, so the default follows that rather than the
        truncation alone, and stops growing once the mesh resolves the mode.
        See `default_quad_degree`. `__init__` warns for free when that rule is
        capped or is extrapolating beyond the meshes it was calibrated on, but
        does **not** measure whether the degree works - call
        `check_boundary_quadrature` for that, and see
        `warn_on_quadrature_rule_limits` for why it is not automatic
      source_quad_degree:
        Optional quadrature degree for the volume source term; useful when
        rho is an analytic UFL expression (e.g. an angular mode restricted to
        a mesh-conforming shell by a conditional) whose exact integration
        removes the source-representation error from convergence studies
      solver_parameters:
        Either a complete dictionary of PETSc solver options, or the string
        "direct"/"iterative" to select a G-ADOPT default. Omitting it (None)
        picks the default by dimension: "direct" in 2-D, "iterative" in 3-D,
        mirroring the Stokes solver. The two presets differ only in how the psi
        (potential) block is inverted - MUMPS versus CG + algebraic multigrid;
        direct does not scale to large 3-D meshes. Passing a full dictionary
        means you own the description of the two Schur blocks that the class
        otherwise sets up.
      solver_parameters_extra:
        Dictionary of PETSc solver options used to update the default G-ADOPT
        options. The safe way to tweak the default (e.g. override only the psi
        block, via a `{"dtn": {"fieldsplit_0": ...}}` entry) while keeping the
        Schur structure.
      dtn_representation:
        How the DtN terms are represented. "multiplier" (default) gives every
        treated mode a scalar Real unknown constrained by a row of a mixed
        system, which keeps every term in plain UFL. "lowrank" eliminates those
        multipliers by hand, leaving one scalar space and a rank-n update to
        the Robin-shifted stiffness applied in factored form; it solves the
        same discrete system - measured agreement 5e-15 on potentials and
        3e-15 on trace coefficients - and is dramatically faster at any
        appreciable truncation, 480x on the first solve and 122x on a repeat
        solve at 162 modes. Its derivative is hand-written in
        `gadopt.dtn_adjoint` rather than taped; the portable adjoint tests pass
        on it at unweakened tolerances. See `NOTES/bench/STEP7-VERDICT.md`.

    ### Valid keys for boundary conditions
    | Condition |  Type  |                    Description                     |
    | :-------- | :----- | :------------------------------------------------: |
    | psi       | Strong | Prescribed potential                               |
    | flux      | Weak   | Prescribed outward normal derivative (d psi)/(d n) |
    | dtn       | Weak   | DtN map (`CylindricalDtN` or `SphericalDtN`)       |
    | sigma     | Weak   | Surface mass density concentrated on the boundary  |

    A boundary with no entry keeps the natural (homogeneous Neumann)
    condition. The orientation of a DtN boundary (exterior map enclosing the
    sources vs interior map enclosing a source-free core) and its radius are
    measured from the marked boundary, not specified by the user. DtN maps
    are only valid on origin-centred coordinate circles/spheres; a boundary
    deviating from constant radius triggers a warning, but the solve
    proceeds with the mean radius and correspondingly degraded accuracy.
    """

    name = "Gravity"
    alpha = 1.0  # Robin shift; see the module docstring

    def __init__(
        self,
        solution: Function,
        rho,
        /,
        *,
        bcs: dict[int | str, dict[str, Any]],
        gravitational_constant: float | Constant = 1.0,
        quad_degree: int | None = None,
        source_quad_degree: int | None = None,
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
        dtn_representation: str = "multiplier",
    ) -> None:
        if dtn_representation not in ("multiplier", "lowrank"):
            raise ValueError(
                f"dtn_representation must be 'multiplier' or 'lowrank', got "
                f"{dtn_representation!r}.")
        self.dtn_representation = dtn_representation
        self.solution = solution
        self.rho = ensure_constant(rho)
        self.bcs = bcs
        self.G = ensure_constant(gravitational_constant)

        self.solution_space = solution.function_space()
        if self.solution_space.value_size != 1:
            raise ValueError("solution must live on a scalar function space")
        self.mesh = self.solution_space.mesh()
        self.X = SpatialCoordinate(self.mesh)

        rho_mesh = extract_unique_domain(self.rho)
        self.cross_mesh = rho_mesh is not None and rho_mesh is not self.mesh
        self.rho_mesh = rho_mesh if self.cross_mesh else self.mesh

        self.set_boundary_conditions()
        self.set_measures(quad_degree, source_quad_degree)
        self.set_boundary_geometry()
        self.set_monopole_datum()
        if self.dtn_representation == "lowrank":
            self.set_lowrank_operator()
        else:
            self.set_function_spaces()
            self.set_form()
        self.warn_on_quadrature_rule_limits()
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.set_solver()

    def set_boundary_conditions(self) -> None:
        """Sorts the `bcs` dictionary into strong and weak conditions."""
        self.dtn_boundaries = []  # [(bc_id, descriptor)]
        self.sigma_bcs = []  # [(bc_id, expression)]
        self.flux_bcs = []  # [(bc_id, expression)]
        self.dirichlet_bcs = []  # [(bc_id, value)]

        for bc_id, bc in self.bcs.items():
            if "dtn" in bc and "psi" in bc:
                raise ValueError(
                    f"Boundary {bc_id}: 'dtn' and 'psi' are mutually exclusive.")
            for bc_type, val in bc.items():
                match bc_type:
                    case "psi":
                        self.dirichlet_bcs.append((bc_id, val))
                    case "flux":
                        self.flux_bcs.append((bc_id, val))
                    case "sigma":
                        self.sigma_bcs.append((bc_id, val))
                    case "dtn":
                        if not isinstance(val, BaseDtN):
                            raise ValueError(
                                f"Boundary {bc_id}: 'dtn' value must be a "
                                "CylindricalDtN or SphericalDtN instance.")
                        if val.dim != self.mesh.geometric_dimension:
                            raise ValueError(
                                f"Boundary {bc_id}: {type(val).__name__} does "
                                f"not apply to a {self.mesh.geometric_dimension}-D mesh.")
                        self.dtn_boundaries.append((bc_id, val))
                    case _:
                        raise ValueError(
                            f"Boundary {bc_id}: unknown condition '{bc_type}'.")

    @staticmethod
    def _scalar_degree(degree) -> int:
        """Tensor-product elements report a (horizontal, vertical) tuple."""
        return degree if isinstance(degree, int) else max(degree)

    @property
    def element_degree(self) -> int:
        """Polynomial degree of the potential's element."""
        return self._scalar_degree(self.solution_space.ufl_element().degree())

    def geometry_probe_measure(self) -> Measure:
        """A cheap boundary measure for sizing the real one.

        The mesh-aware default needs the boundary's facet size and radius
        before it can choose a degree, and those come from boundary integrals,
        which need a measure. This is that measure, and it is not circular: its
        integrands are the facet Jacobian and `|x|`, whose difficulty is set by
        the coordinate element rather than by the truncation, and its results
        only ever reach a `ceil`. `set_boundary_geometry` re-measures the radius
        and the discrete boundary area at the *chosen* degree afterwards, so
        nothing the solver uses numerically comes from this rule.
        """
        return self.surface_measure(2 * self._scalar_degree(
            self.mesh.coordinates.function_space().ufl_element().degree()) + 4)

    def surface_measure(self, degree: int) -> Measure:
        """A boundary measure of the given degree, extruded or not."""
        if self.mesh.extruded:
            return CombinedSurfaceMeasure(self.mesh, degree)
        return ds(domain=self.mesh, degree=degree)

    def boundary_facet_scale(self, measure, bc_id) -> tuple[float, float, float]:
        """`(h_max, h_mean, R)` of one boundary: facet scales and mean radius.

        `h` is the facet's own measure in 2-D and the square root of its area
        in 3-D - a facet *scale*, not its diameter, which for a quadrilateral
        of side `s` would be `s sqrt(2)`. The distinction does not affect the
        rule, because the constants in `QUADRATURE_RULE_COEFFICIENTS` were
        calibrated against this same quantity, but the two are not the same
        number and the difference is absorbed into the intercept.

        `h_max` and not `h_mean`, because a mode is under-resolved on the worst
        facet and one degree has to serve all of them.

        The per-facet measure comes from a DG0 test function, whose entry for a
        cell is the total measure of that cell's facets on this boundary. A cell
        contributing two facets to the same marked boundary therefore reports
        their sum and overstates `h`, which errs towards a richer rule.
        """
        areas = assemble(
            TestFunction(FunctionSpace(self.mesh, "DG", 0)) * measure(bc_id))
        owned = np.asarray(areas.dat.data_ro, dtype=float)
        owned = owned[owned > 0.0]
        # The facet scale is taken per facet and only then averaged -
        # mean(sqrt(area)), never sqrt(mean(area)). The two differ by Jensen's
        # inequality, always in the direction that makes a boundary look more
        # uniform than it is, and `QUADRATURE_RULE_COEFFICIENTS.max_distortion`
        # is read off the calibration campaign, which takes the former
        # (`NOTES/bench/quad_rule.py:206`). Measured, the wrong order
        # understated the distortion of a warped level-4 shell by 4.2% - 1.800
        # against 1.879 - which was enough to hide meshes the rule cannot
        # integrate behind a bound calibrated on meshes it can.
        scales = np.sqrt(owned) if self.mesh.geometric_dimension == 3 else owned
        comm = self.mesh.comm
        h_max = comm.allreduce(float(scales.max()) if scales.size else 0.0,
                               MPI.MAX)
        h_total = comm.allreduce(float(scales.sum()), MPI.SUM)
        facets = comm.allreduce(int(scales.size), MPI.SUM)

        dss = measure(bc_id)
        area = assemble(1 * dss)
        if area <= 0.0 or facets == 0:
            raise ValueError(f"DtN boundary {bc_id} has zero measure.")
        radius = assemble(sqrt(dot(self.X, self.X)) * dss) / area
        return h_max, h_total / facets, radius

    def default_quad_degree(self) -> int:
        r"""The mesh-aware boundary quadrature degree.

            q = odd_ceil(a + b (L h_max / R) + 2 (element_degree - 1))

        capped above by the incumbent `2 (max_mode + element_degree)`, so the
        default never *costs* more than the one it replaces. That is a
        statement about cost and not about accuracy: the cap is a `min`, so on
        a boundary coarse enough for the rule to ask for more than the
        incumbent the chosen degree is the incumbent, and integrates no better
        than the code this replaces did. `check_boundary_quadrature` is what
        makes that visible.

        Odd because a degree-`q` rule uses `(q + 2) // 2` points per direction,
        so `q` even and `q + 1` are the *same* rule and the odd one is exact to
        one degree more; the incumbent is always even and pays for that every
        time. Odd also preserves the low-rank path's `p = q // 2` node
        coincidence, since `(q + 2) // 2 - 1 == q // 2` at either parity.

        Falls back to the incumbent where there is nothing to size the rule
        against - no DtN boundary at all - or where the facet type is
        uncalibrated; see `QUADRATURE_RULE_COEFFICIENTS`.
        """
        return self.quadrature_rule_report().degree

    def quadrature_rule_report(self) -> "QuadratureRuleReport":
        """`default_quad_degree` with its workings, for the warnings to read.

        Separated because the two things worth telling a user - that the cap
        bound the degree, and that the mesh sits outside the range the rule was
        calibrated on - are both by-products of choosing the degree and cost
        nothing extra, whereas measuring whether the degree actually works is
        expensive. See `warn_on_quadrature_rule_limits`.
        """
        element_degree = self.element_degree
        max_mode = max(
            (dtn.max_degree for _, dtn in self.dtn_boundaries), default=1)
        incumbent = 2 * (max_mode + element_degree)

        calibration = QUADRATURE_RULE_COEFFICIENTS.get(
            boundary_facet_cellname(self.mesh))
        if not self.dtn_boundaries or calibration is None:
            return QuadratureRuleReport(
                incumbent, incumbent, incumbent, 0.0, 1.0, calibration)

        probe = self.geometry_probe_measure()
        resolution, distortion = 0.0, 0.0
        for bc_id, dtn in self.dtn_boundaries:
            h_max, h_mean, radius = self.boundary_facet_scale(probe, bc_id)
            resolution = max(resolution, dtn.max_degree * h_max / radius)
            distortion = max(distortion, h_max / h_mean)

        requested = int(np.ceil(
            calibration.intercept + calibration.slope * resolution))
        requested += QUADRATURE_ELEMENT_DEGREE_COST * (element_degree - 1)
        if requested % 2 == 0:
            requested += 1
        return QuadratureRuleReport(
            min(requested, incumbent), requested, incumbent, resolution,
            distortion, calibration)

    def warn_on_quadrature_rule_limits(self) -> None:
        """Say when the default is outside what it was calibrated to do.

        `__init__` deliberately does **not** measure whether the chosen degree
        integrates the modes. It could - `check_boundary_quadrature` does it in
        0.05 s - but only by building `HDiv Trace` spaces over the whole mesh,
        and a trace space spans every facet rather than the boundary alone: at
        degree 6 on a level-4 extruded cubed sphere that is 1.9 million dofs
        against 13842 potential dofs. Firedrake caches the shared data on the
        mesh, so the cost is retained rather than transient - measured, 404 MiB
        held for the lifetime of the mesh, on a problem whose entire remaining
        construction costs 251 MiB. Paying that on every construction, on the
        multiplier path which otherwise builds no trace space at all, is worse
        than the boundary-assembly cost this whole change exists to reduce.

        So the constructor reports only what choosing the degree already told
        it, which is free, and which happens to cover the two known ways the
        rule can be wrong: the cap binding, and a mesh outside the calibration
        set. Anything subtler needs the measurement, which is one call away and
        is what the test suite uses.
        """
        report = getattr(self, "quad_rule_report", None)
        if report is None or report.calibration is None:
            return
        advice = ("Call solver.check_boundary_quadrature() to measure whether "
                  "the rule actually resolves the modes.")
        # Compare rules, not degrees. A degree-q rule uses (q+2)//2 points, so
        # capping an odd request down to the even incumbent below it changes
        # nothing at all - it is the same rule - and warning about it would cry
        # wolf on ordinary configurations. A level-2 shell at L=5 asks for 13
        # and is capped to 12: seven points either way.
        if (report.requested + 2) // 2 > (report.incumbent + 2) // 2:
            warn(
                f"The boundary quadrature rule asked for degree "
                f"{report.requested} at L h_max/R = {report.resolution:.2f}, "
                f"but the default is capped at the {report.incumbent} the "
                f"previous rule would have given. This boundary is too coarse "
                f"for this truncation: the degree is no worse than before, and "
                f"no better. Refine the boundary, lower the truncation, or set "
                f"quad_degree explicitly. " + advice)
        elif (report.distortion > report.calibration.max_distortion
                or report.resolution > report.calibration.max_resolution):
            warn(
                f"The boundary quadrature default is extrapolating: this mesh "
                f"has h_max/h_mean = {report.distortion:.2f} and "
                f"L h_max/R = {report.resolution:.2f}, against "
                f"{report.calibration.max_distortion} and "
                f"{report.calibration.max_resolution} in the set the rule was "
                f"calibrated on. Degree {report.degree} may not be enough. "
                + advice)

    def set_measures(
        self, quad_degree: int | None, source_quad_degree: int | None
    ) -> None:
        """Sets volume and surface measures, with explicit boundary quadrature."""
        #: The rule's workings, or None when the caller named the degree and
        #: there is consequently nothing for the solver to have an opinion on.
        self.quad_rule_report = None
        if quad_degree is None:
            self.quad_rule_report = self.quadrature_rule_report()
            quad_degree = self.quad_rule_report.degree
        self.quad_degree = quad_degree
        self.ds = self.surface_measure(quad_degree)

        if self.cross_mesh:
            self.dx = Measure(
                "dx", domain=self.mesh,
                intersect_measures=(Measure("dx", domain=self.rho_mesh),))
            self.dx_rho = Measure(
                "dx", domain=self.rho_mesh,
                intersect_measures=(Measure("dx", domain=self.mesh),))
        else:
            self.dx = self.dx_rho = dx(domain=self.mesh)
        if source_quad_degree is not None:
            self.dx_rho = self.dx_rho(degree=source_quad_degree)

    def set_boundary_geometry(self) -> None:
        """Measures radius and orientation of every DtN boundary.

        The sign of the boundary integral of n . x decides the side: positive
        means the domain lies inside the boundary (exterior map), negative
        means the boundary encloses a source-free core (interior map).
        """
        n = FacetNormal(self.mesh)
        r = sqrt(dot(self.X, self.X))
        self.boundary_geometry = {}  # bc_id -> (side, R)
        self.boundary_area = {}  # bc_id -> discrete boundary measure A_h

        for bc_id, _ in self.dtn_boundaries:
            dss = self.ds(bc_id)
            area = assemble(1 * dss)
            if area <= 0.0:
                raise ValueError(f"DtN boundary {bc_id} has zero measure.")
            R = assemble(r * dss) / area
            rms = np.sqrt(abs(assemble((r - R) ** 2 * dss)) / area)
            if rms > 1e-3 * R:
                warn(
                    f"DtN boundary {bc_id} deviates from a coordinate "
                    f"circle/sphere (rms radius deviation {rms:.2e} at radius "
                    f"{R:.4g}); the DtN map assumes a boundary of constant radius.")
            side = "exterior" if assemble(dot(n, self.X) * dss) > 0 else "interior"
            self.boundary_geometry[bc_id] = (side, R)
            # The DISCRETE boundary measure. The low-rank elimination divides
            # by scale_k * A_h, and using the analytic measure instead is wrong
            # by the boundary-area error - 2.7e-07 on a cubed-sphere shell,
            # mode-independent and converging, so it reads as a discretisation
            # effect rather than as the wrong formula.
            self.boundary_area[bc_id] = area

    def set_monopole_datum(self) -> None:
        """Prepares the 2-D exterior monopole flux datum; see the module docstring.

        The enclosed mass is held in `Real` `Function`s rather than baked into
        the form as numbers, and rather than promoted to unknowns with
        constraint rows. Both alternatives were measured and both are worse: a
        number severs the tape (§ `update_total_mass`), while an unknown costs
        a field, a constraint row and a Schur dimension, and does not transfer
        to the low-rank path at all, which has no multipliers. A `Real`
        `Function` assigned from an assembled `AdjFloat` is differentiable *and*
        is an ordinary UFL coefficient, so one expression serves both
        representations and `gadopt.dtn_adjoint` picks it up with no new code.

        Two scalars rather than one, because the flux conditions carry mass
        whose contribution to the datum has no `G` in it - see `monopole_flux`.
        """
        #: The 2-D exterior DtN boundaries, which are the only ones whose
        #: monopole the trace fails to determine. Empty in 3-D.
        self.monopole_boundaries = [
            bc_id for bc_id, _ in self.dtn_boundaries
            if self.mesh.geometric_dimension == 2
            and self.boundary_geometry[bc_id][0] == "exterior"]

        self._real_space = None
        self.mesh_volume = None
        self.source_mass = None
        self.cavity_flux = None
        if not self.monopole_boundaries:
            # Nothing is built in 3-D, not even the Real space and the volume
            # assemble: "the 3-D path pays nothing for this" should be true of
            # the code and not only of the per-solve cost.
            return

        if len(self.monopole_boundaries) > 1:
            raise ValueError(
                f"2-D mesh with {len(self.monopole_boundaries)} exterior DtN "
                f"boundaries {self.monopole_boundaries}: the monopole datum "
                "needs the mass enclosed by the exterior boundary, which is "
                "not defined when there is more than one of them.")
        clash = sorted(
            {bc_id for bc_id, _ in self.flux_bcs}.intersection(
                self.monopole_boundaries),
            key=str)
        if clash:
            raise ValueError(
                f"Boundary {clash[0]}: 'flux' and a 2-D exterior 'dtn' both "
                "prescribe the m = 0 normal derivative there - the DtN map "
                "through the enclosed mass, the flux condition directly - so "
                "the two are a double specification of one quantity. Move the "
                "flux condition to an interior boundary or drop it.")

        self._real_space = FunctionSpace(self.mesh, "R", 0)
        #: The measure the one-by-one system's operator assembles to; plain
        #: `dx`, never `self.dx`, which in the cross-mesh case is intersected
        #: with the density's mesh and is therefore a different number.
        self.mesh_volume = assemble(Constant(1.0) * dx(domain=self.mesh))
        #: `int rho dx + sum_sheets int sigma ds`, refreshed by `solve`.
        self.source_mass = Function(self._real_space, name="source_mass")
        #: `sum_flux int (d psi)/(d n) ds` over the prescribed-flux boundaries,
        #: which is `4 pi G` times the mass they imply inside their cavities.
        self.cavity_flux = Function(self._real_space, name="cavity_flux")

    def enclosed_mass_forms(self) -> tuple:
        """`(identity, mass, flux)`: the one-by-one systems on the Real space.

        `mu` and `nu` are globally constant, so dividing the left-hand side by
        `mesh_volume` - the value that same measure assembles to, from the same
        measure - makes the system the identity, and `source_mass` holds the
        mass its name claims rather than a mass per unit volume. See
        `update_total_mass` for why the mass is routed through a solve at all,
        and why the solve is rebuilt every time rather than cached.

        `ensure_constant` is applied to the boundary values because a Python
        `0.0` folds the whole integral away, and a right-hand side that folds
        to nothing is not a linear form at all - which for a lone
        `{"flux": 0.0}` condition meant a `ValueError` from deep inside
        Firedrake rather than the zero it obviously means.
        """
        mu = TestFunction(self._real_space)
        nu = TrialFunction(self._real_space)
        identity = (nu / self.mesh_volume) * mu * dx(domain=self.mesh)

        mass = self.rho * mu * self.dx_rho
        for bc_id, sigma in self.sigma_bcs:
            mass += ensure_constant(sigma) * mu * self.ds(bc_id)

        flux = None
        for bc_id, value in self.flux_bcs:
            term = ensure_constant(value) * mu * self.ds(bc_id)
            flux = term if flux is None else flux + term
        return identity, mass, flux

    def monopole_flux(self, bc_id):
        """`(d psi)/(d n)` of the m = 0 mode on a 2-D exterior DtN boundary.

        `-2 G M / R`, with `M` everything the boundary encloses: the volumetric
        density, the sheets, and the cavities implied by prescribed-flux
        boundaries. By Gauss the last of those is `int g ds / (4 pi G)` for a
        flux datum `g`, whose `G` cancels against the `2 G` in front - so that
        piece is written without `G` rather than as a `G/G`, which keeps it
        exactly differentiable when the gravitational constant is itself a
        control.

        Returned as a flux rather than as a form so that both representations
        can feed it through the same idiom they already use for the `flux`
        boundary conditions, where the two sign conventions are visible side by
        side (`set_form` builds a residual, `set_lowrank_operator` a
        right-hand side).
        """
        _, R = self.boundary_geometry[bc_id]
        flux = -2.0 * self.G * self.source_mass / R
        if self.flux_bcs:
            # Left out entirely rather than added as a term that is
            # structurally zero, so that a configuration with no flux
            # conditions does not carry a tape dependency that can never move.
            flux -= self.cavity_flux / (2.0 * np.pi * R)
        return flux

    def set_function_spaces(self) -> None:
        """Builds the internal mixed space: psi, cross-mesh dummy, multipliers.

        Vector-valued R-space unknowns are unsupported in Firedrake, so each
        trace coefficient is a separate scalar R field.
        """
        spaces = [self.solution_space]
        if self.cross_mesh:
            spaces.append(FunctionSpace(self.rho_mesh, "DG", 0))

        R_scalar = FunctionSpace(self.mesh, "R", 0)
        self.n_multipliers = sum(
            dtn.n_multipliers(self.boundary_geometry[bc_id][0])
            for bc_id, dtn in self.dtn_boundaries)
        if self.n_multipliers > 200:
            warn(
                f"{self.n_multipliers} DtN multipliers requested; the "
                "per-field bookkeeping of scalar R spaces makes large "
                "truncation degrees expensive. Consider a lower truncation "
                "combined with a buffer region.")
        self._multiplier_offset = len(spaces)
        spaces.extend([R_scalar] * self.n_multipliers)

        self.mixed_space = MixedFunctionSpace(spaces)
        self.mixed_solution = Function(self.mixed_space)

    def set_form(self) -> None:
        """Sets the weak form: Poisson, DtN maps, sheets, and flux conditions."""
        trials = split(self.mixed_solution)
        tests = TestFunctions(self.mixed_space)
        psi, v = trials[0], tests[0]
        i_R = self._multiplier_offset
        multipliers = list(zip(trials[i_R:], tests[i_R:]))
        self._multiplier_keys = []

        F = dot(grad(psi), grad(v)) * self.dx
        F -= 4 * np.pi * self.G * self.rho * v * self.dx_rho
        if self.cross_mesh:
            # The dummy field makes Firedrake set up the cross-mesh entity
            # maps; its equation is trivial.
            lam, mu_lam = trials[1], tests[1]
            F += lam * mu_lam * Measure("dx", domain=self.rho_mesh)

        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            dss = self.ds(bc_id)
            F += (self.alpha / R) * psi * v * dss

            for mode in dtn.modes(side, R, self.X):
                (c, mu), multipliers = multipliers[0], multipliers[1:]
                self._multiplier_keys.append((bc_id, mode.key))
                # Constraint row: mu is globally constant, so this is one
                # scalar equation enforcing c = (boundary integral of
                # psi e_k) / norm - the trace coefficient of e_k.
                F += (psi * mode.expr - mode.scale * c) * mu * dss
                # Modal correction to the shifted flux (same sign for
                # exterior and interior maps).
                F += (mode.lam - self.alpha / R) * c * mode.expr * v * dss

        for bc_id, sigma in self.sigma_bcs:
            F -= 4 * np.pi * self.G * sigma * v * self.ds(bc_id)

        for bc_id, flux in self.flux_bcs:
            F -= flux * v * self.ds(bc_id)

        for bc_id in self.monopole_boundaries:
            F -= self.monopole_flux(bc_id) * v * self.ds(bc_id)

        self.F = F
        self.strong_bcs = [
            DirichletBC(self.mixed_space.sub(0), val, bc_id)
            for bc_id, val in self.dirichlet_bcs]

    def set_lowrank_operator(self) -> None:
        """Builds the Robin-shifted stiffness, the right-hand side and `C`.

        The multipliers are eliminated by hand (see `gadopt.dtn_lowrank`), so
        there is no mixed space, no Schur complement and no matrix-free
        operator: one scalar space, one assembled SPD matrix, and a rank-`n`
        update applied in factored form. `mixed_solution` still exists and is
        still the thing to reset before a timed solve, but it is now a plain
        `Function` on the potential space rather than a mixed one.
        """
        V = self.solution_space
        u, v = TrialFunction(V), TestFunction(V)
        self.mixed_space = V
        self.mixed_solution = Function(V)
        self.n_multipliers = 0
        self._multiplier_offset = 0
        self._multiplier_keys = []

        a = dot(grad(u), grad(v)) * self.dx
        for bc_id, _ in self.dtn_boundaries:
            _, R = self.boundary_geometry[bc_id]
            a += (self.alpha / R) * u * v * self.ds(bc_id)

        F = 4 * np.pi * self.G * self.rho * v * self.dx_rho
        for bc_id, sigma in self.sigma_bcs:
            F += 4 * np.pi * self.G * sigma * v * self.ds(bc_id)
        for bc_id, flux in self.flux_bcs:
            F += flux * v * self.ds(bc_id)

        for bc_id in self.monopole_boundaries:
            F += self.monopole_flux(bc_id) * v * self.ds(bc_id)

        self.a_form, self.rhs_form = a, F
        self.strong_bcs = [
            DirichletBC(V, val, bc_id) for bc_id, val in self.dirichlet_bcs]

        self.mode_rows = []
        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            self.mode_rows.append(build_boundary_mode_rows(
                V, self.ds(bc_id), dtn, side, R, self.alpha, self.quad_degree))
            self._multiplier_keys.extend(
                (bc_id, key) for key in self.mode_rows[-1].keys)
        self.build_time = sum(rows.build_time for rows in self.mode_rows)

        constrained = set()
        for bc in self.strong_bcs:
            constrained.update(np.asarray(bc.nodes, dtype=np.int64).tolist())
        apply_dirichlet_to_rows(self.mode_rows, constrained)

    def set_lowrank_solver(self) -> None:
        """Drives a PETSc KSP directly: CG on `A + B`, preconditioned by `A`.

        `NonlinearVariationalSolver` offers no hook to hand PETSc a custom
        `Amat` while taking `Pmat` from a form, so this path owns its KSP. That
        is also why the taped variational solve disappears here.

        `ksp.setOperators(N, A)` with different `Amat` and `Pmat` is exactly the
        paper's "build the multigrid on the stiffness alone"; our `A` is
        Robin-shifted rather than `epsilon`-shifted, so it is SPD by
        construction with nothing to tune.
        """
        self.A = assemble(self.a_form, bcs=self.strong_bcs, mat_type="aij")
        petsc_A = self.A.petscmat
        petsc_A.setOption(petsc_A.Option.SPD, True)

        comm = self.mesh.comm
        self.operator_context = LowRankDtNOperator(petsc_A, self.mode_rows, comm)
        self.N = PETSc.Mat().createPython(
            petsc_A.getSizes(), self.operator_context, comm=comm)
        self.N.setUp()
        self.N.setOption(PETSc.Mat.Option.SYMMETRIC, True)

        self.ksp = PETSc.KSP().create(comm=comm)
        self.ksp.setOperators(self.N, petsc_A)
        self.ksp.setOptionsPrefix(self.name + "_lowrank_")
        options = PETSc.Options()
        for key, value in _flatten_options(self.solver_parameters).items():
            options[self.name + "_lowrank_" + key] = value
        self.ksp.setFromOptions()

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None,
    ) -> None:
        """Sets PETSc solver options.

        The psi (potential) block is the only thing that varies between the two
        presets. Matrices with R-space blocks cannot be assembled
        monolithically, so with DtN multipliers present both presets share a
        full Schur complement eliminating onto the R fields: the psi block is
        assembled and inverted (directly by MUMPS, or iteratively by CG + AMG),
        and the tiny dense multiplier Schur complement is handled by GMRES.
        Without multipliers the psi solver is applied to the whole assembled
        system.

        The split itself is set up by `gadopt.DtNTwoBlockSchurPC`, which
        describes the two blocks to PETSc as index sets rather than as lists of
        field numbers; that is what keeps the solve working past the 128-field
        limit PETSc imposes on field enumeration, which one scalar R space per
        angular mode otherwise crosses at a spherical truncation of L = 7 on a
        two-boundary shell (L = 11 with a single DtN boundary). Its sub-solvers
        are configured under the `dtn` entry.

        A full solver-options dictionary is used verbatim. The strings "direct"
        and "iterative" select a G-ADOPT preset; None picks one by dimension
        (direct in 2-D, iterative in 3-D), mirroring the Stokes solver, so the
        scalable path is the default at production scale.
        """
        if isinstance(solver_preset, Mapping):
            if (self.n_multipliers > 0
                    and solver_preset.get("mat_type") in ("aij", "baij", "sbaij")):
                raise ValueError(
                    "Monolithic matrix assembly is unsupported with DtN "
                    "R-space multipliers. Provide a fieldsplit configuration "
                    "(see the G-ADOPT default) or adjust the default via "
                    "solver_parameters_extra instead.")
            self.add_to_solver_config(solver_preset)
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.set_solver)
            return

        if solver_preset is not None and solver_preset not in ("direct", "iterative"):
            raise ValueError(
                f"solver_parameters must be a dictionary or one of 'direct', "
                f"'iterative', got {solver_preset!r}.")
        if solver_preset is None:
            solver_preset = (
                "direct" if self.mesh.topological_dimension == 2 else "iterative")

        if self.dtn_representation == "lowrank":
            # This path drives a PETSc KSP directly, so the options are plain
            # KSP/PC options with no mat_type or snes_type: there is no
            # Firedrake solver object to interpret those. "direct" inverts the
            # PRECONDITIONER A by LU, not the operator A + B - the low-rank
            # update is never assembled, so no direct method applies to it, and
            # CG is still doing the work.
            if solver_preset == "direct":
                self.add_to_solver_config({
                    "ksp_type": "cg", "ksp_rtol": 1e-11, "ksp_max_it": 1000,
                    **{f"pc_{k}": v for k, v in (
                        ("type", "lu"),
                        ("factor_mat_solver_type", "mumps"))},
                })
            else:
                self.add_to_solver_config(dict(lowrank_gravity_solver_parameters))
            if INFO >= log_level:
                self.add_to_solver_config({"ksp_converged_reason": None})
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.set_solver)
            return

        if self.n_multipliers == 0:
            # No DtN multipliers: the psi solver acts on the whole assembled
            # system. GAMG wants the plain (unprefixed) options here, since
            # there is no fieldsplit_0 AssembledPC to hand them to. This also
            # means no SPDAssembledPC and hence no MAT_SPD flag - the eigen-
            # estimates fall back to GMRES; benign, and not worth the
            # AssembledPC indirection on a monolithic aij solve. Unlike the
            # multiplier path there is no outer Krylov cleaning up the psi
            # solve, so ksp_rtol here IS the final solution accuracy (the same
            # situation as the Stokes preonly-outer path): tighten it if a
            # no-multiplier 3-D config ever needs gradient-quality output.
            base = {"mat_type": "aij", "snes_type": "ksponly"}
            if solver_preset == "direct":
                self.add_to_solver_config({**base, **direct_gravity_solver_parameters})
            else:
                self.add_to_solver_config({
                    **base,
                    "ksp_type": "cg",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                    "pc_type": "gamg",
                    "mg_levels_pc_type": "sor",
                    "pc_gamg_threshold": 0.01,
                    "pc_gamg_square_graph": 100,
                    "pc_gamg_coarse_eq_limit": 1000,
                    "pc_gamg_mis_k_minimum_degree_ordering": True,
                })
                if INFO >= log_level:
                    self.add_to_solver_config({"ksp_converged_reason": None})
        else:
            if solver_preset == "direct":
                fieldsplit_0 = {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled": dict(direct_gravity_solver_parameters),
                }
            else:
                fieldsplit_0 = dict(iterative_gravity_solver_parameters)
            self.add_to_solver_config({
                "mat_type": "matfree",
                "snes_type": "ksponly",
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-11,
                "pc_type": "python",
                "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
                "dtn": {
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0": fieldsplit_0,
                    "fieldsplit_1": {
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-6,
                        "pc_type": "none",
                    },
                },
            })
            if INFO >= log_level:
                self.add_to_solver_config({"ksp_converged_reason": None})
            if DEBUG >= log_level:
                self.add_to_solver_config({
                    "dtn": {
                        "fieldsplit_0": {"ksp_converged_reason": None},
                        "fieldsplit_1": {"ksp_converged_reason": None},
                    },
                })

        self.add_to_solver_config(solver_extras)
        self.warn_on_stale_fieldsplit_options()
        self.register_update_callback(self.set_solver)

    def warn_on_stale_fieldsplit_options(self) -> None:
        """Warns about fieldsplit options left at the top level of the preset.

        The two Schur blocks used to be described to PETSc by field number, so
        their sub-solvers sat directly under the solver options. They now live
        inside `gadopt.DtNTwoBlockSchurPC` under its own `dtn` prefix, and a
        `fieldsplit_0` entry passed at the top level of
        `solver_parameters_extra` silently stops matching anything rather than
        failing - the one way this change can bite an existing script.
        """
        if self.n_multipliers == 0:
            return
        stale = sorted(
            key for key in self.solver_parameters
            if key.startswith(("fieldsplit_", "pc_fieldsplit_")))
        if stale:
            warn(
                f"Solver options {stale} sit at the top level, where the DtN "
                "preset no longer reads them: the two Schur blocks are now "
                "described by index sets inside gadopt.DtNTwoBlockSchurPC and "
                "its sub-solvers live under the 'dtn' entry. Pass them as "
                "solver_parameters_extra={'dtn': {'fieldsplit_0': ...}} "
                "instead.")

    def set_solver(self) -> None:
        """Sets up the Firedrake variational problem and solver."""
        if self.dtn_representation == "lowrank":
            self.set_lowrank_solver()
            return
        self.problem = NonlinearVariationalProblem(
            self.F, self.mixed_solution, bcs=self.strong_bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            options_prefix=self.name,
        )

    def update_total_mass(self) -> None:
        """Refreshes the enclosed-mass scalars the monopole datum reads.

        Called from `solve` and not from `__init__`, so that the assembles and
        the assigns land on the tape. That placement is the whole design: the
        mass has to be differentiable without ever becoming an unknown.

        **Do not freeze these as `float`s.** Doing so severs the channel in a
        way no Taylor test can see: the taped map is then self-consistent, so
        `taylor_test` converges at rate 2.0000 while the gradient of the actual
        forward map is wrong - measured at 325% on both representations, the
        adjoint returning 4.948 where a central difference of fresh solves gives
        1.163. The instrument that catches it is a finite difference of *fresh
        solves*, which is why the tests carry one.

        **And do not reach for the obvious spelling either.** Getting an
        assembled scalar into a form differentiably is narrower than it looks;
        of the five ways to write it, two work. Measured on a stand-alone
        problem whose right-hand side depends on the control only through the
        scalar, against a reference gradient of 6.168:

        | spelling                                   | adjoint | tangent |
        | ------------------------------------------ | ------- | ------- |
        | `Constant.assign(assemble(...))`           | **0.0** | fails   |
        | `assemble(...).riesz_representation("l2")` | **0.0** | fails   |
        | `Function(R).assign(assemble(...))`        | 6.168   | **fails** |
        | `assemble(...).riesz_representation("L2")` | 6.168   | 6.168   |
        | a small variational problem on `R`         | 6.168   | 6.168   |

        The first two sever silently, which is the failure mode this whole item
        exists to avoid. The third is the tempting one and is what the plan note
        proposed: its adjoint is exact, because `assemble` of a 0-form returns
        an `AdjFloat` and the assign is taped, but its *tangent* value arrives
        as a bare `numpy.float64` that `derivative` rejects, so it fails the
        TLM-adjoint dot-product identity while every Taylor test still passes.
        The last two are equivalent - the `L2` Riesz map is a mass-matrix solve
        and emits the same block - so the explicit one is written out, because
        it says what it does without an argument about which volume factor
        cancels which.

        **The solve is rebuilt on every call, and must be.** Hoisting it into a
        cached `LinearVariationalSolver` is the obvious optimisation - it saves
        3.2 ms of the datum's 3.8 ms, since the one-by-one operator is a fixed
        number - and it is wrong in parallel. A cached solver whose right-hand
        side carries a *facet* integral into a `Real` space returns the correct
        value on its first solve and garbage afterwards: measured on a unit
        disc, `1.881574` then `0.0` alternating on two ranks, and
        `1.881574, -3.763149, 13.171020, -37.631486` on four. The KSP still
        converges, so the failure is a silently wrong potential in exactly the
        configuration this item exists to enable - a mass sheet on a 2-D
        exterior boundary, solved more than once, in parallel.

        Every other spelling is stable at 1, 2 and 4 ranks: plain repeated
        1-form assembly of the same form object, of a fresh one, assembly into
        a pre-zeroed tensor, 0-form assembly, and the fresh solve used here.
        Only the cached solver reuses whatever state is at fault. This looks
        like a Firedrake defect rather than a misuse and is worth reporting
        upstream, alongside the `R`-row reproducer `SOLVE-STRATEGIES.md`
        already owes them.

        A no-op outside 2-D, and in 2-D without an exterior DtN boundary, so
        the 3-D path pays nothing for any of this.
        """
        if not self.monopole_boundaries:
            return

        identity, mass, flux = self.enclosed_mass_forms()
        solve(identity == mass, self.source_mass,
              solver_parameters=real_scalar_solver_parameters)
        if flux is not None:
            solve(identity == flux, self.cavity_flux,
                  solver_parameters=real_scalar_solver_parameters)
        self.check_net_mass()

    def total_enclosed_mass(self) -> float:
        """Everything the 2-D exterior DtN boundary encloses, as a number.

        Density, sheets, and the cavity masses implied by prescribed-flux
        boundaries. Valid after `solve`; zero when there is no such boundary,
        which includes every 3-D configuration - there the scalars are never
        built, because the 3-D `l = 0` exterior mode is determined by the trace
        and needs none of this.

        Diagnostic only - the differentiable quantities are `source_mass` and
        `cavity_flux`, and reading a float off them is the sever documented in
        `update_total_mass`.
        """
        if not self.monopole_boundaries:
            return 0.0
        return (self._real_value(self.source_mass)
                + self._real_value(self.cavity_flux)
                / (4.0 * np.pi * self._real_value(self.G)))

    @staticmethod
    def _real_value(value) -> float:
        """A number from a `Constant`, a `Real` `Function` or a plain scalar.

        `Real` data is replicated on every rank, so reading entry 0 is a local
        operation that needs no reduction and agrees everywhere.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(value.dat.data_ro[0])

    def check_net_mass(self) -> None:
        """Warns about monopole leakage, and refuses a doubly-anchored gauge.

        This used to refuse any nonzero net mass in 2-D, because the monopole
        datum did not exist. It does now (see the module docstring), so nonzero
        mass is ordinary, and what is left is the check's second job: a density
        that *should* integrate to zero but does not usually has DG0 leakage
        from an interface that does not conform to cell edges, which is a
        genuine accuracy problem whatever the boundary treatment.

        "Nonzero" can no longer be the trigger for that, so the trigger is
        "nonzero but implausibly small": a leaked monopole is a discretisation
        residue a few orders above round-off, while a mass anyone meant to be
        there sits far above the band. The band errs quiet - it will miss
        leakage on a density whose own mass is genuinely tiny - which is the
        right direction for a warning that would otherwise fire on every
        legitimate nonzero-mass problem and teach users to filter it.

        The refusal that replaces the old one is the gauge argument's caveat.
        With a strong `psi` condition present, `v = 1` is not admissible, so
        the Robin monopole relation and the Dirichlet condition become two
        competing anchors for the same constant and the system is
        over-constrained. Measured on an annulus, that costs 1.1555 relative
        error, identical at two refinements - a real over-constraint, not a
        convergence artefact - so it is refused rather than warned about.
        """
        if not self.monopole_boundaries:
            return

        scale = assemble(abs(self.rho) * self.dx_rho)
        for bc_id, sigma in self.sigma_bcs:
            # `ensure_constant` for the same reason as in `enclosed_mass_forms`:
            # a Python `0.0` folds `abs(sigma) * ds` away entirely, and
            # assembling nothing raises `IndexError: tuple index out of range`
            # from inside UFL rather than returning the zero it means.
            scale += assemble(abs(ensure_constant(sigma)) * self.ds(bc_id))
        mass = abs(self._real_value(self.source_mass))
        # Against the density's own scale and nothing else. An earlier
        # `max(scale, 1.0)` floor turned this into an absolute test below
        # scale 1, so uniformly rescaling rho - a transformation the whole
        # linear problem is invariant under - switched the refusal below off:
        # rho = 1e-9 was refused and rho = 1e-10 solved silently and wrongly.
        relative = mass / scale if scale > 0.0 else float(mass > 0.0)
        # The flux conditions are user data rather than a discretised field, so
        # they cannot leak; they can still anchor a monopole, hence the second
        # term here and nowhere in the leakage band below.
        anchored = relative > 1e-8 or self._real_value(self.cavity_flux) != 0.0

        if anchored and self.dirichlet_bcs:
            raise ValueError(
                f"Net enclosed mass {self.total_enclosed_mass():.3e} is "
                "nonzero on a 2-D mesh that has both an exterior DtN boundary "
                f"and a strong 'psi' condition on boundary "
                f"{self.dirichlet_bcs[0][0]}. The 2-D monopole datum fixes the "
                "potential's additive constant through the Robin term, and the "
                "Dirichlet condition fixes it again: the two anchors disagree "
                "and the system is over-constrained. Drop one of them, or use "
                "a zero-net-mass density.")

        if 1e-8 < relative < 1e-4:
            warn(
                f"Net mass {self._real_value(self.source_mass):.3e} is "
                f"{relative:.1e} of the density's own scale - nonzero, but too "
                "small to look deliberate. This is the signature of monopole "
                "leakage from a density interface that does not conform to "
                "cell edges, which is an accuracy problem in its own right; "
                "align the mesh with the density interfaces. If the mass is "
                "meant to be there, ignore this.")

    def boundary_quadrature_rule(self, degree: int, bc_id):
        """The boundary rule at `degree`, as `(weights * detJ, points)`.

        `HDiv Trace` of degree `p` places its nodes at the `(p + 1)`-point
        Gauss-Legendre points, and a measure of degree `q` uses `(q + 2) // 2`
        of them, so at `p = q // 2` the nodes **are** the quadrature points.
        Assembling a trace test function against the measure then returns
        `w_i |J|_i` at node `i`, and interpolating the coordinates gives the
        points: one assembly and one interpolation recover the entire rule,
        after which every mode integral over that boundary is numpy.

        This is the same undocumented FInAT property `gadopt.dtn_lowrank`
        relies on, so it holds exactly where `supports_trace_build` does, and
        callers must validate it rather than trust it - `check_boundary_
        quadrature` does, against one assembled mode.

        **One scalar trace space, never a vector one.** The coordinates are
        interpolated component by component through a single reused `Function`
        rather than through a `VectorFunctionSpace`. That is not a
        micro-optimisation: a trace space spans every facet of the mesh, not
        just the boundary, and Firedrake caches its shared data on the mesh, so
        anything built here is retained for the mesh's lifetime. Measured on a
        level-4 extruded cubed sphere - 13842 potential dofs - the vector
        variant cost 923 MiB of retained memory against 251 MiB for the whole
        rest of the constructor. Only the boundary entries are kept, so the
        arrays returned are `O(boundary)` rather than `O(mesh)`.

        Returns owned entries only; the caller reduces across ranks.
        """
        measure = self.surface_measure(degree)(bc_id)
        trace = FunctionSpace(self.mesh, "HDiv Trace", degree // 2)
        weights = np.asarray(
            assemble(TestFunction(trace) * measure).dat.data_ro, dtype=float)
        # Strictly positive on every facet the measure runs over - Gauss
        # weights are positive and |detJ| does not vanish - and exactly zero on
        # every other facet, which is what makes this the boundary selector.
        carries = weights != 0.0
        component = Function(trace)
        points = np.empty((int(carries.sum()), self.mesh.geometric_dimension))
        for axis in range(self.mesh.geometric_dimension):
            component.interpolate(self.X[axis])
            points[:, axis] = np.asarray(
                component.dat.data_ro, dtype=float)[carries]
        return weights[carries], points

    #: Points per chunk when evaluating modes on a recovered boundary rule. The
    #: tabulation is dense in (modes, points), so this bounds its footprint at
    #: `n_modes * CHUNK` doubles regardless of boundary size or truncation.
    _tabulation_chunk = 8192

    def _mode_norms(self, bc_id, dtn, side, radius, degree, keys) -> np.ndarray:
        """`integral e_k^2 ds` at `degree`, for the named modes, in `keys` order."""
        if supports_trace_build(self.mesh):
            weights, points = self.boundary_quadrature_rule(degree, bc_id)
            order = {mode.key: i for i, mode
                     in enumerate(dtn.mode_metadata(side, radius))}
            wanted = [order[key] for key in keys]
            local = np.zeros(len(keys))
            for start in range(0, len(weights), self._tabulation_chunk):
                block = slice(start, start + self._tabulation_chunk)
                table = tabulate_boundary_modes(dtn, side, points[block])[wanted]
                local += (table * table) @ weights[block]
            # Owned entries only, so this is the exact global integral with
            # nothing double counted - the same reduction the low-rank
            # operator's coefficients() performs.
            total = np.empty_like(local)
            self.mesh.comm.Allreduce(local, total, MPI.SUM)
            return total

        measure = self.surface_measure(degree)(bc_id)
        modes = {mode.key: mode for mode
                 in dtn.modes_by_key(keys, side, radius, self.X)}
        return np.array([assemble(modes[key].expr**2 * measure) for key in keys])

    def _validate_recovered_rule(self, bc_id, dtn, side, radius, degree,
                                 keys, values) -> None:
        """Confirm the recovered rule reproduces one assembled mode integral.

        The recovery rests on trace nodes coinciding with quadrature points,
        which is a FInAT variant choice rather than a guarantee. If it lapsed,
        the points would be misplaced while the weights still summed to the
        boundary area, so a cheaper invariant would not catch it - and this
        function is the backstop for the quadrature default, which makes
        silently measuring the wrong thing the one failure that matters.

        The mode is taken from `validation_keys` and never from the end of the
        caller's list, which in the degenerate truncations is the constant:
        `SphericalDtN(L=0)` samples only `Y0,0` and `CylindricalDtN(M=0)` on an
        interior boundary only `mean`. A constant is reproduced exactly by any
        nodal interpolation whatever its node placement, so validating on one
        certifies nothing - measured elsewhere in this codebase, `Y0,0` passes
        at 1.2e-16 on a build wrong by eight orders.
        """
        wanted = [key for key in validation_keys(dtn) if key in keys]
        if not wanted:
            return
        key = wanted[0]
        value = values[keys.index(key)]
        sampled = dtn.modes_by_key([key], side, radius, self.X)
        if not sampled:
            return
        reference = assemble(sampled[0].expr**2 * self.surface_measure(degree)(bc_id))
        if abs(value - reference) > 1e-9 * abs(reference):
            raise RuntimeError(
                f"The recovered boundary quadrature rule disagrees with "
                f"assembly by {abs(value - reference) / abs(reference):.3e} on "
                f"mode {key} of boundary {bc_id} at degree {degree}. The "
                f"recovery needs 'HDiv Trace' of degree {degree // 2} to place "
                f"its nodes at the {degree // 2 + 1}-point Gauss-Legendre "
                f"points, which this Firedrake no longer does; the same "
                f"assumption underpins gadopt.dtn_lowrank's fast build.")

    def check_boundary_quadrature(
        self, rtol: float = 1e-8, action: str = "raise",
        sample: str = "all", reference_degree: int | None = None,
    ) -> float:
        """Measures the boundary rule's own quadrature error, by self-convergence.

        Differences the boundary integral of each eigenfunction squared at
        `quad_degree` against the same integral at a richer degree on the same
        mesh, and returns the worst relative difference, raising (or warning,
        with action="warn") beyond rtol.

        **It used to compare against the analytic normalisation, and that could
        not see the variable it was named for.** The discrete boundary is not a
        sphere - a degree-2 coordinate cubed sphere differs from one by an
        amount no rule removes - so that comparison reported boundary
        sphericity. Measured on a level-2 extruded cubed sphere with
        `SphericalDtN(L=8)`, the old quantity moves from 2.328711e-05 at
        `q = 12` to 2.328713e-05 at `q = 40` - flat to seven significant
        figures across a four-fold change in the degree - while one refinement
        level moves it to 4.976520e-07, nearly two orders.
        Differencing two degrees on one mesh cancels the geometry error exactly,
        because it is common to both, and leaves quadrature error alone: on the
        same configuration the new quantity runs 1.5492e-05 at `q = 8` to
        1.6221e-15 at `q = 20`, ten orders down to the round-off floor of the
        assembled integral.
        Sphericity is not thereby unchecked: `set_boundary_geometry` warns on
        the rms radius deviation, which is the direct measurement of it.

        This matters more than a tidy-up because `default_quad_degree` is a
        calibrated formula whose margin on badly distorted meshes is thin, and
        this is the only thing standing behind it.

        Args:
          rtol: Relative difference beyond which the check fails.
          action: "raise" or "warn".
          sample: "all" checks every treated mode; this is the guarantee, and
            it is what the tests use. "extremes" checks only the modes the
            descriptor's `check_modes` returns, `O(L)` of them rather than
            `O(L^2)`. It is a **proxy, not a
            bound** - the worst mode is not always the highest one, and on a
            level-2 cubed sphere the sampled set was measured at 0.93 to 1.00
            of the true worst.
            On meshes where the rule can be recovered (interval and
            quadrilateral facets) the cost of either is two assemblies per
            boundary per degree rather than one per mode, so "all" is nearly
            free there and the distinction only bites on simplex facets.
          reference_degree: Degree to difference against. Defaults to
            `quad_degree + 8`, four more points per direction, which sits about
            four orders below the degree under test - enough to certify a 1e-8
            target. Measured against a far richer reference it under-reports by
            nothing at all down to `q = 3`, and on a mesh whose self-convergence
            genuinely floors it still tracks the truth to within 16%, erring
            conservative.
        """
        if sample not in ("all", "extremes"):
            raise ValueError(
                f"sample must be 'all' or 'extremes', got {sample!r}.")
        if reference_degree is None:
            reference_degree = self.quad_degree + 8

        worst, worst_key = 0.0, None
        for bc_id, dtn in self.dtn_boundaries:
            side, radius = self.boundary_geometry[bc_id]
            keys = [mode.key for mode
                    in (dtn.mode_metadata(side, radius) if sample == "all"
                        else dtn.check_modes(side, radius, self.X))]
            if not keys:
                continue
            values = self._mode_norms(
                bc_id, dtn, side, radius, self.quad_degree, keys)
            exact = self._mode_norms(
                bc_id, dtn, side, radius, reference_degree, keys)
            if supports_trace_build(self.mesh):
                self._validate_recovered_rule(
                    bc_id, dtn, side, radius, reference_degree, keys, exact)
            for key, value, reference in zip(keys, values, exact):
                deviation = abs(value - reference) / abs(reference)
                if deviation > worst:
                    worst, worst_key = deviation, (bc_id, key)

        if worst > rtol:
            message = (
                f"Boundary quadrature does not resolve DtN mode {worst_key}: "
                f"self-convergence against degree {reference_degree} is "
                f"{worst:.3e} > {rtol:.1e} at quad_degree={self.quad_degree}. "
                "Raise quad_degree, or refine the boundary mesh.")
            if action == "warn":
                warn(message)
            else:
                raise ValueError(message)
        return worst

    def solve(self) -> None:
        """Solves the system and updates the solution function in place."""
        self.update_total_mass()
        if self.dtn_representation == "lowrank":
            self._solve_lowrank()
            annotate_lowrank_solve(self)
        else:
            self.solver.solve()
        self.solution.assign(self.mixed_solution.subfunctions[0])

    def _solve_lowrank(self, rhs_form=None) -> None:
        """CG on `A + B` with the right-hand side reassembled each call.

        `rho` and the sheet densities may have changed since construction;
        `A + B` cannot have, since every coefficient in it is geometry.

        Strong conditions are lifted by hand, and **both** halves of that are
        necessary. `assemble(L, bcs=...)` *zeroes* the constrained entries -
        the homogeneous convention - rather than setting them to the
        prescribed values, so the write-back below is not redundant tidying:
        it is what makes the answer right. And it subtracts none of the
        coupling of the prescribed values into the free rows. Measured against
        Firedrake's own solver on a plain Dirichlet Poisson problem with no
        DtN anywhere:

            raw KSP, b = assemble(L, bcs=bcs)   relative error 1.0
            fd.solve(A, w, b), same b           relative error 3.0e-16
            the construction below              relative error 4.4e-16

        Both the lifting and the value-setting live in Firedrake's `solve`
        wrapper rather than in assembly. **Any Firedrake code that drives a
        KSP directly has to do both by hand** - which is why the ordinary
        idiom works everywhere else in this codebase and fails only here.

        `lift` must be the ZERO EXTENSION of the boundary data, not the data
        function itself: `action(a, g)` with `g` nonzero in the interior
        over-subtracts by the interior block and gives a relative error of 1.0
        again. The wrong version looks more natural.

        The lifting needs `A` only: the columns of `C` are zeroed at
        constrained degrees of freedom, so `B` applied to a field supported
        only there is zero.
        """
        rhs_form = self.rhs_form if rhs_form is None else rhs_form
        if self.strong_bcs:
            lift = Function(self.solution_space)
            for bc in self.strong_bcs:
                bc.apply(lift)
            b = assemble(rhs_form - action(self.a_form, lift))
            # `bc.nodes` carries ghost entries too, while `dat.data_wo` is the
            # owned block; the vector handed to the KSP is owned-only, so the
            # ghosts are both unnecessary and out of range.
            owned = self.solution_space.dof_dset.size
            for bc in self.strong_bcs:
                nodes = bc.nodes[bc.nodes < owned]
                b.dat.data_wo[nodes] = lift.dat.data_ro[nodes]
        else:
            b = assemble(rhs_form)
        with b.dat.vec_ro as rhs, self.mixed_solution.dat.vec as x:
            self.ksp.solve(rhs, x)
        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"The low-rank DtN solve did not converge: PETSc reason "
                f"{self.ksp.getConvergedReason()} after "
                f"{self.ksp.getIterationNumber()} iterations.")

    def coefficients(self) -> dict[int | str, dict[str, float]]:
        """Solved trace coefficients of every DtN boundary, keyed by marker.

        For each DtN boundary, maps mode labels (e.g. "cos3", "Y2,-1") to the
        solved trace coefficients - the spectrum of psi on that boundary,
        i.e. the geoid coefficients when evaluated at the surface.
        """
        out = {bc_id: {} for bc_id, _ in self.dtn_boundaries}
        if self.dtn_representation == "lowrank":
            # Taped, not read off .dat.data_ro and float()ed: c_k is a linear
            # functional of psi with a constant vector, so it is trivially
            # differentiable, and the trace spectrum is the geoid rather than a
            # diagnostic. The values are AdjFloat, which is a float subclass,
            # so callers that only want a number are unaffected.
            return taped_trace_coefficients(self)
        i_R = self._multiplier_offset
        for (bc_id, key), f in zip(
                self._multiplier_keys, self.mixed_solution.subfunctions[i_R:]):
            out[bc_id][key] = float(f)
        return out
