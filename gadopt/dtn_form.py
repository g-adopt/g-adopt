r"""The Dirichlet-to-Neumann boundary treatment, as a reusable form builder.

`DtNGravityForm` owns everything the DtN boundary condition needs and nothing
else: it sorts the boundary dictionary, chooses and holds the boundary
measures, measures the radius and orientation of every DtN boundary, counts and
keys the multipliers, and writes the boundary bilinear form and the boundary
source. The physics it implements - the Robin-shifted flux split, the modal
constraint rows, mass sheets on exterior and tagged interior facets, the 2-D
monopole exception - is described in the module docstring of
`gadopt.gravity_solver`, which is the only place it needs describing.

**What it deliberately does not own.** No density, no volume source, no
Laplacian, no function space beyond the potential's, no solver, and none of the
2-D enclosed-mass bookkeeping. Every coefficient in it is geometry or boundary
data, so nothing it builds is taped and nothing it builds depends on the state
of a simulation. That is what makes it safe to instantiate twice, on two
different spaces, inside one coupled residual.

The one quantity that has to cross the seam in the other direction is the 2-D
monopole flux datum, which is a function of the enclosed mass and therefore of
the density. It crosses as a coefficient the caller injects,
`boundary_source(v, extra_flux={bc_id: expr})`, through the same channel the
prescribed-`flux` conditions already use. `gadopt.gravity_solver.GravitySolver`
keeps the `Real` fields, the mass forms and `check_net_mass`; see
`GravitySolver.update_total_mass`, whose docstring explains why that code is
delicate enough to be worth leaving where its tests can reach it.

There are two consumers. `GravitySolver` uses `boundary_bilinear` with a list
of multiplier `(trial, test)` pairs on its multiplier path and without them on
its low-rank path, where `build_mode_rows` eliminates the multipliers by hand
instead; both take their right-hand side from `boundary_source`. A coupled
self-gravitating residual is the second, and uses the same two methods on its
own mixed space.
"""

import abc
from collections import namedtuple
from typing import Any
from warnings import warn

import numpy as np
from firedrake import *
from mpi4py import MPI
from ufl import Form

from .dtn_lowrank import (
    boundary_facet_cellname, build_boundary_mode_rows, supports_trace_build,
    tabulate_boundary_modes, validation_keys)
from .spherical_harmonics import real_spherical_harmonic
from .utility import CombinedSurfaceMeasure, ensure_constant

__all__ = ["BaseDtN", "CylindricalDtN", "DtNGravityForm", "DtNMode",
           "QuadratureCalibration", "QuadratureRuleReport", "SphericalDtN"]


QuadratureCalibration = namedtuple(
    "QuadratureCalibration",
    ["intercept", "slope", "max_distortion", "max_resolution"])
"""The fitted rule, and the range of meshes it was fitted on.

`intercept` and `slope` give `q = intercept + slope * (L h_max / R)`.
`max_distortion` (`h_max / h_mean`) and `max_resolution` (`L h_max / R`) record
how far the calibration set actually reached, so the solver can say when it is
extrapolating instead of interpolating. Outside those the rule is a guess, and
`DtNGravityForm.warn_on_quadrature_rule_limits` says so rather than pretending.

Both bounds are the campaign's own maxima, and both must be read with
`boundary_facet_scale`'s definition of `h_mean` - see the note there, because
computing it the other way round understates the distortion and lets meshes the
rule cannot integrate past the bound. The interval bound is 1.65 rather than the
measured 1.621, and the triangular ones 2.1 and 5.7 rather than the measured
2.07 and 5.654, so that the fitted rows themselves do not trip them; a rule that
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
    "triangle": QuadratureCalibration(9.0, 2.5, 2.1, 5.7),
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
would ship a rule that under-integrates everywhere else. 45 quadrilateral
configurations, 45 triangular ones and 30 interval ones, target 1e-8
self-convergence, then confirmed against the integrand the solver actually
assembles. Derivation, the intercept decision and the element-degree
measurements are in `NOTES/FINDING-QUADRATURE-DEGREE-FORM.md`; the mechanism
that invalidated the earlier calibration is in
`NOTES/FINDING-QUADRATURE-CANCELLATION.md`.

**Triangles carry their own constants rather than the quadrilateral ones**,
measured, not assumed: transferring the quadrilateral constants would have
under-integrated 5 of the 45 measured triangular configurations, worst at
`ms0.5 w0.2 L20` (needs 25, would be given 21), so the fitted rule is both
taller and steeper (9 + 2.5 x against 8 + 2.0 x). **Why is not as settled as
the fact.** All 5 failing rows are warped (`w0.1` or `w0.2`); none of the 15
unwarped rows fails, and the warped rows' distortion (worst 2.07) exceeds
anything in the quadrilateral set (1.88). A collapsed Gauss-Jacobi rule being
weaker than a tensor rule at the same nominal degree is one explanation and
the one this campaign was designed to test, but the calibration variable
`x = L h_max/R` not capturing distortion (the open question in A8) is an
equally consistent reading of the same 5 rows, and the data does not
distinguish them. Do not transfer `(9, 2.5)` to a differently-distorted
triangular family on the strength of the mechanism story alone. The set is
unstructured gmsh tetrahedra at three mesh sizes and three warps; the
campaign is `--simplex` in `NOTES/bench/quad_calibrate.py`, and the instrument
that made it affordable is described in `NOTES/bench/quad_rule.py`.

Read by the facet's reference cell, so this entry also governs an *extruded*
triangular base cell - the icosahedral shell - which the calibration set does
not contain. Neither fast instrument can measure that family (see
`dtn_lowrank.boundary_facet_cellname`), so it is covered by a per-mode
assembled spot check instead, `--icosahedral`.
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
"""How `DtNGravityForm` arrived at its boundary quadrature degree.

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


class DtNGravityForm:
    """The DtN boundary treatment on one potential space.

    Sorts `bcs`, chooses the boundary quadrature rule, measures every DtN
    boundary, and hands back the two pieces of the residual that live on the
    boundary: `boundary_bilinear(psi, v, multipliers)` and
    `boundary_source(v, extra_flux)`. Their difference is the boundary
    residual, and `boundary_residual` writes it out.

    The class computes nothing that depends on the state of a simulation, so
    a form built once serves every timestep; the only things it assembles are
    the geometric measurements of `set_boundary_geometry` and
    `check_sheet_measures`, both of which the shipped `GravitySolver` already
    made in its own constructor.

    Args:
      potential_space:
        Scalar function space the potential lives on.
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and
        value), in the same form `GravitySolver` takes; see its docstring for
        the table of valid keys.
      gravitational_constant:
        Value of G in the chosen non-dimensionalisation. This multiplies the
        mass sheets, which is the only place it appears here. **A coupled
        solver that has absorbed `4 pi G` into a scaling of its own must pass
        that scaling divided by `4 pi`.** With the self-gravitating momentum
        scaling `Lambda = 4 pi G rho_bar D / g_bar`, the volume source is
        written `Lambda int rho_0 u . grad v` with the `4 pi G` already inside
        `Lambda`, while the sheets here carry an explicit `4 pi G sigma`; so
        the constructor argument is `Lambda / (4 pi)` and the two are then
        consistent by construction. Getting it wrong scales the sheet against
        the volume source by a constant, which a symmetry test cannot see -
        only a comparison against a closed form that fixes the sheet's
        absolute size does.
      quad_degree:
        Integer denoting the quadrature degree applied on boundary integrals
        involving the DtN eigenfunctions. `None` takes the mesh-aware default;
        see `default_quad_degree` and `GravitySolver`'s own docstring for why
        UFL's automatic estimate cannot be used here.
      alpha:
        The Robin shift, defaulting to the class attribute. Passed in rather
        than read off the class so that a caller which exposes the shift as
        its own knob - `GravitySolver` does, and a test subclasses it to vary
        the shift - stays in control of the value the boundary form uses.
    """

    alpha = 1.0  # Robin shift; see the `gadopt.gravity_solver` module docstring

    def __init__(
        self,
        potential_space: FunctionSpace,
        bcs: dict[int | str, dict[str, Any]],
        /,
        *,
        gravitational_constant: float | Constant = 1.0,
        quad_degree: int | None = None,
        alpha: float | Constant | None = None,
    ) -> None:
        self.solution_space = potential_space
        if self.solution_space.value_size != 1:
            raise ValueError("potential_space must be a scalar function space")
        self.bcs = bcs
        self.G = ensure_constant(gravitational_constant)
        if alpha is not None:
            self.alpha = alpha

        self.mesh = self.solution_space.mesh()
        self.X = SpatialCoordinate(self.mesh)

        self.set_boundary_conditions()
        self.set_measures(quad_degree)
        self.check_sheet_measures()
        self.set_boundary_geometry()
        self.count_multipliers()

        #: `[(bc_id, mode_key)]` in the order the multipliers are consumed,
        #: filled by whichever of `boundary_bilinear` and `build_mode_rows`
        #: the caller drives.
        self.multiplier_keys = []
        #: The low-rank mode rows, positionally aligned with `dtn_boundaries`;
        #: `None` until `build_mode_rows` runs.
        self.mode_rows = None

    def set_boundary_conditions(self) -> None:
        """Sorts the `bcs` dictionary into strong and weak conditions.

        A `sigma_bcs` entry carries the facet *kind* alongside the tag, as
        UFL's own integral type, because `ds` and `dS` are different measures
        and no property of the tag distinguishes them - see the
        `gadopt.gravity_solver` module docstring. Every consumer of the list
        goes through `sheet_integral`, so the kind is interpreted in one place.
        """
        self.dtn_boundaries = []  # [(bc_id, descriptor)]
        self.sigma_bcs = []  # [(bc_id, expression, integral_type)]
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
                        self.sigma_bcs.append((bc_id, val, "exterior_facet"))
                    case "interior_sigma":
                        if self.mesh.extruded:
                            raise NotImplementedError(
                                f"Boundary {bc_id}: 'interior_sigma' is not "
                                "available on an extruded mesh. Its interior "
                                "facets split into the horizontal and vertical "
                                "families, dS_h carries no subdomain ids at "
                                "all, and the geometry this condition exists "
                                "for - a gmsh parent with the surface tagged as "
                                "an interior curve/surface - is not extruded. "
                                "Use 'sigma' on a boundary facet instead.")
                        self.sigma_bcs.append((bc_id, val, "interior_facet"))
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
        # (`NOTES/bench/quad_rule.geometry` - a name rather than a line
        # number, since the file is gitignored and reshuffles under edits).
        # Measured, the wrong order understated the distortion of a warped
        # level-4 shell by 4.2% - 1.800
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
        integrates the modes, and the first reason is a convention rather than a
        number: **G-ADOPT solvers do not compute in their constructors.** They
        validate structurally and cheaply; anything that measures is an explicit
        call the caller decides to make. That alone settles the placement, and
        it does not become wrong if the measurement gets cheaper.

        The cost happens to agree, and is recorded because it is what the
        alternative would have cost. `check_boundary_quadrature` does it in
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

    def set_measures(self, quad_degree: int | None) -> None:
        """Sets the boundary measures, with explicit boundary quadrature.

        The volume measures are the caller's: the boundary treatment knows
        nothing about the source, and in the cross-mesh configuration the
        volume measure is a property of the density's mesh rather than of this
        one.
        """
        #: The rule's workings, or None when the caller named the degree and
        #: there is consequently nothing for the solver to have an opinion on.
        self.quad_rule_report = None
        if quad_degree is None:
            self.quad_rule_report = self.quadrature_rule_report()
            quad_degree = self.quad_rule_report.degree
        self.quad_degree = quad_degree
        self.ds = self.surface_measure(quad_degree)
        #: For sheets on tagged interior facets, at the same degree as the
        #: boundary measure - a sheet is the same integrand whichever kind of
        #: facet it sits on, so it has no reason to be integrated differently.
        #: `None` on an extruded mesh, where `set_boundary_conditions` has
        #: already refused the only condition that would read it.
        self.dS = (None if self.mesh.extruded
                   else dS(domain=self.mesh, degree=quad_degree))

    def sheet_integral(self, integrand, bc_id, integral_type: str):
        """One sheet's integral, on whichever kind of facet it sits.

        The only place that knows a sheet can be an interior facet, so that the
        four sites which integrate a sheet - the residual, the low-rank
        right-hand side, the enclosed-mass form and `check_net_mass`'s scale -
        cannot drift apart in the measure or in the restriction. The last two
        of those live on `GravitySolver` and reach this through the shim.

        An interior-facet integrand must be restricted, and it is restricted
        with `avg` and never with a hard-coded `'+'`. Measured on a gmsh
        annulus with the circle tagged, the inner side is `'+'` on every facet
        of it, at 1, 2 and 4 ranks - but that is gmsh's cell ordering and
        nothing guarantees it. `avg` is exact for the continuous case that
        matters here (psi is CG, and a sheet density is single-valued on the
        facet), and for a two-valued DG sigma it takes the mean of the two
        sides, which is the only reading of "the mass on this facet" that does
        not depend on which cell the mesh happens to list first.
        """
        if integral_type == "exterior_facet":
            return integrand * self.ds(bc_id)
        return avg(integrand) * self.dS(bc_id)

    def check_sheet_measures(self) -> None:
        """Refuses a sheet whose facets do not exist. Called from `__init__`.

        A measure asked for the wrong kind of tag does not raise: it assembles
        to `0.0` and emits `WARNING Subdomain (2,) is empty`, so a load sheet
        written on `ds` when its facets are interior is *silently absent* - the
        solve converges, and the potential is missing the entire direct
        attraction of the load. No downstream test looks: a sheet is a
        right-hand side with no Jacobian contribution, so a symmetry test
        cannot see it, and any two solvers sharing the form omit it
        identically.

        This is a measurement in a constructor, which the class otherwise
        avoids (see `warn_on_quadrature_rule_limits`). It is the same
        measurement `set_boundary_geometry` already makes on every DtN
        boundary, one assemble of `1` per sheet, and it is here rather than in
        a test because a test only covers the configurations someone thought
        to write.
        """
        for bc_id, _, integral_type in self.sigma_bcs:
            length = assemble(
                self.sheet_integral(Constant(1.0), bc_id, integral_type))
            if length > 0.0:
                continue
            kind = ("an exterior facet" if integral_type == "exterior_facet"
                    else "a tagged interior facet")
            other = ("interior_sigma" if integral_type == "exterior_facet"
                     else "sigma")
            raise ValueError(
                f"Boundary {bc_id}: the sheet measure is empty - no facet on "
                f"this mesh carries tag {bc_id!r} as {kind}, so the sheet "
                f"would contribute nothing and the solve would converge "
                f"without it. Firedrake reports every physical-group label in "
                f"both facet sets, so the tag alone does not tell you which "
                f"kind it is; if these facets are the other kind, use "
                f"'{other}'.")

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

    def count_multipliers(self) -> None:
        """How many scalar `Real` unknowns the multiplier representation needs.

        A property of the boundaries alone, so it is available before any form
        is built and a caller can size its mixed space with it. The low-rank
        representation eliminates every one of them and needs none, which is
        why `GravitySolver.n_multipliers` is zero there while this stays at the
        true count.
        """
        self.n_multipliers = sum(
            dtn.n_multipliers(self.boundary_geometry[bc_id][0])
            for bc_id, dtn in self.dtn_boundaries)

    def boundary_bilinear(self, psi, v, multipliers=None):
        """The boundary terms bilinear in the unknowns: Robin shift and DtN.

        Without `multipliers` this is the Robin shift alone, which is what the
        low-rank representation wants: it eliminates the modal part by hand
        into `build_mode_rows` and keeps only a symmetric positive-definite
        stiffness in the potential.

        With `multipliers` - an iterable of `(trial, test)` pairs on `Real`
        spaces, one per multiplier, in the caller's own space ordering - the
        modal part is written out as well: a scalar constraint row per mode
        defining the trace coefficient, and the modal correction to the shifted
        flux that feeds it back. The pairs are consumed in the order the
        boundaries appear in `dtn_boundaries` and, within a boundary, the order
        the descriptor's `modes` returns, and `multiplier_keys` records the
        pairing.

        Args:
          psi: The potential, as a trial function or as a split component of a
            mixed solution.
          v: The potential's test function.
          multipliers: `None`, or `n_multipliers` pairs of `(trial, test)`.
        """
        a = Form([])
        remaining = None if multipliers is None else list(multipliers)
        if remaining is not None:
            if len(remaining) != self.n_multipliers:
                raise ValueError(
                    f"boundary_bilinear needs {self.n_multipliers} multiplier "
                    f"(trial, test) pairs for these boundaries, got "
                    f"{len(remaining)}.")
            self.multiplier_keys = []

        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            dss = self.ds(bc_id)
            a += (self.alpha / R) * psi * v * dss
            if remaining is None:
                continue

            for mode in dtn.modes(side, R, self.X):
                (c, mu), remaining = remaining[0], remaining[1:]
                self.multiplier_keys.append((bc_id, mode.key))
                # Constraint row: mu is globally constant, so this is one
                # scalar equation enforcing c = (boundary integral of
                # psi e_k) / norm - the trace coefficient of e_k.
                a += (psi * mode.expr - mode.scale * c) * mu * dss
                # Modal correction to the shifted flux (same sign for
                # exterior and interior maps).
                a += (mode.lam - self.alpha / R) * c * mode.expr * v * dss
        return a

    def boundary_source(self, v, extra_flux=None):
        """The boundary terms linear in the test function: sheets and fluxes.

        Returned in the **right-hand side** convention, so a residual is
        `boundary_bilinear(psi, v, ...) - boundary_source(v, ...)` and a
        right-hand side form is this on its own. The two conventions used to
        sit side by side in `GravitySolver.set_form` and
        `GravitySolver.set_lowrank_operator`; keeping the sign in one place is
        the point of the split.

        Args:
          v: The potential's test function.
          extra_flux: `{bc_id: expression}` of additional outward normal
            derivatives, added exactly as the prescribed-`flux` conditions are.
            This is the channel the 2-D monopole datum crosses the seam
            through: the enclosed-mass bookkeeping stays on `GravitySolver`
            (see `GravitySolver.update_total_mass` for why moving it is a bad
            trade) and only the resulting flux value, `-2 G M / R`, reaches the
            boundary form. The expression may be any UFL coefficient, including
            a `Real` `Function` a tape is watching.

        Returns:
          A `Form`, empty when there is no sheet, no prescribed flux and no
          extra flux. An empty form adds to and subtracts from another form as
          the identity, so callers need no special case.
        """
        L = Form([])
        for bc_id, sigma, integral_type in self.sigma_bcs:
            L += 4 * np.pi * self.G * self.sheet_integral(
                sigma * v, bc_id, integral_type)

        for bc_id, flux in self.flux_bcs:
            L += flux * v * self.ds(bc_id)

        for bc_id, flux in (extra_flux or {}).items():
            L += flux * v * self.ds(bc_id)
        return L

    def boundary_residual(self, psi, v, multipliers=None, extra_flux=None):
        """`boundary_bilinear(...) - boundary_source(...)`.

        The whole boundary contribution to a residual, for a caller that wants
        it in one piece. `GravitySolver` does not use it: its multiplier path
        needs the two halves separately anyway, because the volume source sits
        between them in the term order the shipped form has always had.
        """
        return (self.boundary_bilinear(psi, v, multipliers)
                - self.boundary_source(v, extra_flux))

    def build_mode_rows(self):
        """The low-rank representation's eliminated mode rows, per boundary.

        One `BoundaryModeRows` per DtN boundary, **positionally aligned with
        `dtn_boundaries`** - `gadopt.dtn_adjoint.taped_trace_coefficients`
        pairs the two sequences with a `zip`, so a filtered or re-sorted list
        would attribute every trace coefficient to the wrong boundary without
        raising. The alignment is asserted below rather than left to the
        reader, since `build_boundary_mode_rows` handles a boundary with no
        modes at all gracefully and the temptation to skip it is real.
        """
        self.multiplier_keys = []
        self.mode_rows = []
        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            self.mode_rows.append(build_boundary_mode_rows(
                self.solution_space, self.ds(bc_id), dtn, side, R, self.alpha,
                self.quad_degree))
            self.multiplier_keys.extend(
                (bc_id, key) for key in self.mode_rows[-1].keys)
        assert len(self.mode_rows) == len(self.dtn_boundaries)
        return self.mode_rows

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
