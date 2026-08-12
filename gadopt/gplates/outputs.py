"""Output strategies for plate-reconstruction scalar fields.

An OutputStrategy turns interpolated source arrays at target mesh nodes into a
scalar field. The current two flavours are:

  - indicator fields  — ~1 inside the region of interest, ~0 outside, with a
    smooth quintic transition at the region base (QuinticOutput).

  - normalised geotherms — (T - T_surface) / (T_LAB - T_surface) in [0, 1],
    using an erf profile (oceanic, age-dependent) or a linear profile
    (continental). The "outside" value is 1 — i.e. mantle temperature.

Outputs declare what they need from the source via the class-level ``requires``
set; the connector validates this against the source's ``provides`` set at
construction time so, e.g. a polygon source paired with an erf geotherm fails loudly
rather than silently dropping the missing key.

The radial primitive shared by every indicator output is ``radial_quintic_step``,
a ONE-SIDED quintic smoothstep: exactly 1 from the surface down to the region
base, decaying to exactly 0 over one transition width BELOW the base. Because
it is one-sided, the transition never straddles the base, so the surface value
stays faithful — exactly 1 wherever the region has any
thickness — and the field is exactly 0 below base + width (no asymptotic tail).
The 0.5-crossing therefore sits at base + width/2 in depth, not at the base.

One consequence to keep in mind: where thickness is exactly zero the base sits
at the surface and the SURFACE NODE still reads 1 (the transition hangs just
below it). A thickness channel therefore cannot express "the region is absent
here", and that is not a defect of the step — it is the step being handed one
number where the field needs two.

Which is what separates the two indicator outputs. `QuinticOutput` suits a
source that covers the whole sphere, where thickness never vanishes laterally
(or ``default_thickness_km`` fills the gaps, lithosphere-style) and every
column genuinely has a region in it. A BOUNDED region needs a second channel
saying where it is, and `MaskedQuinticOutput` consumes it. There is no
configuration of a single-channel output that covers the bounded case: the
source's thickness channel holds depth multiplied by membership, and no
per-node arithmetic recovers two unknowns from one number.

  ============================ ==================== ==========================
  Configuration                Base depth           Lateral extent
  ============================ ==================== ==========================
  QuinticOutput()              variable (per-node)  none — total coverage
  QuinticOutput(base_depth_km) fixed                none — total coverage
  MaskedQuinticOutput()        variable, deblended  the membership channel
  MaskedQuinticOutput(base_..) fixed                the membership channel
  LateralFractionOutput        none (pure lateral)  the membership channel
  ============================ ==================== ==========================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import erf


# Mesh geometry
@dataclass(frozen=True)
class MeshConfig:
    """Non-dimensional mesh geometry shared by every output that converts
    radius to physical depth.

    ``r_outer`` is the radial coordinate of Earth's surface in mesh units
    (default 2.208 for an Earth-like spherical shell with r_inner=1.208).
    ``depth_scale`` is the physical depth (km) per non-dimensional unit
    (default 2890 — the mantle's depth ratio).
    """

    r_outer: float = 2.208
    depth_scale: float = 2890.0

    def __post_init__(self):
        if self.r_outer <= 0:
            raise ValueError(f"r_outer must be positive, got {self.r_outer}")
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")


# Geotherm functions (used by GeothermERFOutput / GeothermLinearOutput)
def ocean_erf_normalized(depth_m, z_lab_m, age_myr, kappa):
    """Normalised erf geotherm for oceanic lithosphere.

    T_norm = erf(z / a) / erf(z_lab / a), where a = 2 * sqrt(kappa * t).
    T_norm = 0 at the surface, 1 at the LAB depth. Falls back to linear
    when age <= 0 (just-formed crust at a ridge).
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)
    age_myr = np.asarray(age_myr, dtype=float)

    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    a = 2.0 * np.sqrt(kappa * np.maximum(age_sec, 1.0))

    result = np.zeros_like(depth_m)
    valid = z_lab_m > 0
    erf_z = erf(depth_m[valid] / a[valid])
    erf_zlab = erf(z_lab_m[valid] / a[valid])
    safe = erf_zlab > 1e-10
    result[valid] = np.where(
        safe, erf_z / np.maximum(erf_zlab, 1e-10),
        depth_m[valid] / z_lab_m[valid],
    )
    return np.clip(result, 0.0, 1.0)


def continental_linear(depth_m, z_lab_m):
    """Linear geotherm for continental lithosphere. T_norm = z / z_lab."""
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)
    result = np.zeros_like(depth_m)
    valid = z_lab_m > 0
    result[valid] = depth_m[valid] / z_lab_m[valid]
    return np.clip(result, 0.0, 1.0)


# Shared radial primitive (used by every indicator output)
def radial_quintic_step(r_target, base_r, width_nondim):
    """One-sided quintic smoothstep: 1 above the base, 0 one width below it.

    Returns exactly 1 for ``r >= base_r``, exactly 0 for
    ``r <= base_r - width_nondim``, and the quintic polynomial
    ``6t^5 - 15t^4 + 10t^3`` (t the normalised position inside the band) in
    between. The polynomial has zero first AND second derivatives at both
    ends, so the field is C^2 across the junctions.

    The transition sits entirely BELOW the base: a node at the base itself
    reads exactly 1, and the 0.5-crossing is at ``base_r - width_nondim / 2``.
    All radii are in
    non-dimensional mesh units. ``base_r`` may be a scalar (fixed base depth)
    or a per-node array (variable base depth read from the thickness channel).
    """
    # (r - base)/w + 1 rather than (r - (base - w))/w: any r >= base_r then
    # lands at t >= 1 BEFORE rounding, so the clip makes the upper plateau
    # exactly 1 (no 1-ulp shortfall at the base itself).
    t = np.clip(
        (np.asarray(r_target, dtype=float) - base_r) / width_nondim + 1.0,
        0.0, 1.0,
    )
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


# OutputStrategy ABC
class OutputStrategy(ABC):
    """Map interpolated source arrays at target coords to a scalar field.

    Subclasses declare ``requires`` — the set of source-dict keys (excluding
    ``"xyz"``) they read from — as a class constant or, for composed outputs, as
    an instance attribute set in ``__init__``. The connector validates
    ``output.requires <= source.provides`` at construction time.
    """

    requires: frozenset[str]

    @abstractmethod
    def compute(
        self,
        interpolated: dict[str, np.ndarray],
        r_target: np.ndarray,
        too_far: np.ndarray,
        mesh: MeshConfig,
    ) -> np.ndarray:
        """Return the scalar field at the target points.

        Args:
            interpolated: dict of arrays, one per key in ``self.requires``,
                already kNN-interpolated onto target coords.
            r_target: norms of target coords (one value per target node).
            too_far: boolean mask; True where no source seed is within
                ``InterpolationConfig.distance_threshold``.
            mesh: mesh geometry for radius↔depth conversion.
        """


# Below this blended membership a node is treated as outside the region and the
# thickness deblend is skipped. The deblend divides by the membership fraction,
# so it amplifies blend noise as that fraction goes to zero; the trailing
# multiply by the same fraction bounds the result either way, and this floor
# keeps the intermediate finite.
MEMBERSHIP_FLOOR = 1e-3


# ---------------------------------------------------------------------------
# Composable indicator parts (radial x base x lateral)
# ---------------------------------------------------------------------------
#
# An indicator field is a radial profile whose base depth is set one of a few
# ways, multiplied by a lateral amplitude. Rather than a class per combination,
# `IndicatorOutput` takes one part of each kind and its ``requires`` is the
# union of the parts'. `QuinticOutput` and `MaskedQuinticOutput` are thin
# presets over it, kept because they are the two combinations in constant use.
# Each part declares the channels it reads so the composed ``requires`` is
# exact — a fixed base reads nothing, a data base reads ``thickness``, a
# deblended base reads ``masked_thickness`` and ``membership``.

def _clip_membership(interpolated, too_far):
    """The in-region weight fraction m, clipped to [0, 1] and zeroed where a
    node has no nearby source point. Shared by every membership-reading part so
    the base and the lateral amplitude see exactly the same m."""
    m = np.clip(interpolated["membership"], 0.0, 1.0)
    return np.where(too_far, 0.0, m)


class QuinticStep:
    """Radial part: a one-sided quintic smoothstep, 1 above the base and 0 one
    ``width_km`` below it."""

    requires = frozenset()

    def __init__(self, width_km: float = 10.0):
        if width_km <= 0:
            raise ValueError(f"width_km must be positive, got {width_km}")
        self.width_km = width_km

    def step(self, r_target, base_r, mesh):
        return radial_quintic_step(
            r_target, base_r, self.width_km / mesh.depth_scale
        )


class FixedBase:
    """Base part: the region base sits at one fixed depth for every column, so
    no depth channel is read."""

    requires = frozenset()

    def __init__(self, base_depth_km: float):
        if base_depth_km <= 0:
            raise ValueError(
                f"base_depth_km must be positive, got {base_depth_km}"
            )
        self.base_depth_km = base_depth_km

    def base_r(self, interpolated, too_far, mesh):
        return mesh.r_outer - self.base_depth_km / mesh.depth_scale


class DataBase:
    """Base part: per-node base depth read straight from a genuine ``thickness``
    channel (total-coverage sources). ``too_far`` nodes are filled with
    ``default_thickness_km``."""

    requires = frozenset({"thickness"})

    def __init__(self, default_thickness_km: float = 0.0):
        if default_thickness_km < 0:
            raise ValueError(
                f"default_thickness_km must be non-negative, "
                f"got {default_thickness_km}"
            )
        self.default_thickness_km = default_thickness_km

    def base_r(self, interpolated, too_far, mesh):
        thickness_km = interpolated["thickness"].copy()
        thickness_km[too_far] = self.default_thickness_km
        return mesh.r_outer - thickness_km / mesh.depth_scale


class DeblendedBase:
    """Base part: per-node base depth deblended from a bounded source's
    ``masked_thickness = m * h`` and ``membership = m``, giving the data's own
    depth h up to the boundary rather than the coverage-shallowed product."""

    requires = frozenset({"masked_thickness", "membership"})

    def base_r(self, interpolated, too_far, mesh):
        membership = _clip_membership(interpolated, too_far)
        covered = membership > MEMBERSHIP_FLOOR
        thickness_km = np.where(
            covered,
            interpolated["masked_thickness"] / np.maximum(
                membership, MEMBERSHIP_FLOOR
            ),
            0.0,
        )
        return mesh.r_outer - thickness_km / mesh.depth_scale


class NoAmplitude:
    """Lateral part: none. The region covers the whole sphere, so the amplitude
    is 1 everywhere and only the base depth varies."""

    requires = frozenset()

    def amplitude(self, interpolated, too_far):
        return 1.0


class MembershipAmplitude:
    """Lateral part: the in-region weight fraction m, so the field falls to 0
    across the region's edge (and at ``too_far`` nodes)."""

    requires = frozenset({"membership"})

    def amplitude(self, interpolated, too_far):
        return _clip_membership(interpolated, too_far)


class TaperedAmplitude:
    """Lateral part: an arbitrary taper g(m) of the membership fraction, clipped
    to [0, 1]. Makes the margin taper a parameter — g the identity recovers
    `MembershipAmplitude`, a steeper g sharpens the edge — rather than a class
    of its own."""

    requires = frozenset({"membership"})

    def __init__(self, g):
        self.g = g

    def amplitude(self, interpolated, too_far):
        m = _clip_membership(interpolated, too_far)
        return np.clip(self.g(m), 0.0, 1.0)


class SurfaceAmplitude:
    """Lateral part: a per-node amplitude published by the source, read straight
    from its own channel rather than derived here. The source owns the physics
    (e.g. an oceanic-only fade that weakens thin/young seafloor while continents
    stay at 1.0); this class only interpolates and clips it. ``too_far`` nodes,
    with no seed nearby, read 1.0 (full strength)."""

    requires = frozenset({"surface_amplitude"})

    def amplitude(self, interpolated, too_far):
        a = interpolated["surface_amplitude"].copy()
        a[too_far] = 1.0
        return np.clip(a, 0.0, 1.0)


class IndicatorOutput(OutputStrategy):
    """A composable indicator: ``radial`` profile x ``base`` depth x ``lateral``
    amplitude. ``requires`` is the union of the three parts', so a source need
    only provide what the chosen parts actually read.

    `QuinticOutput` and `MaskedQuinticOutput` below are thin presets for the two
    combinations in constant use; reach for `IndicatorOutput` directly for the
    others (e.g. a fixed base with a membership amplitude, or a tapered margin).
    """

    def __init__(self, radial, base, lateral):
        self.radial = radial
        self.base = base
        self.lateral = lateral
        # The union of what the chosen parts read. Presets override this with
        # their fixed historical contract in __init__.
        self.requires = radial.requires | base.requires | lateral.requires

    def compute(self, interpolated, r_target, too_far, mesh):
        base_r = self.base.base_r(interpolated, too_far, mesh)
        step = self.radial.step(r_target, base_r, mesh)
        return step * self.lateral.amplitude(interpolated, too_far)


# ---------------------------------------------------------------------------
# Concrete outputs
# ---------------------------------------------------------------------------

class QuinticOutput(IndicatorOutput):
    """Smooth indicator field for a region that covers the whole sphere.

    The field is exactly 1 from the surface down to the region base and decays
    to exactly 0 over ``width_km`` below it. There is no lateral term: every
    column is inside the region, and only its depth varies.

    That restriction is the class's whole contract. Pairing it with a source
    that is zero outside a bounded region paints the entire exterior surface —
    zero thickness puts the base at the surface, where the step reads 1 — and
    no lateral amplitude derived from the same thickness channel repairs it,
    because that channel holds depth multiplied by membership. Use
    `MaskedQuinticOutput` for bounded regions; `ScalarFieldConnector` rejects
    the wrong pairing at construction.

    ``base_depth_km``
        ``None`` (default): the base depth is per-node, read from the
        interpolated ``thickness`` channel — deep cratonic roots and shallow
        margins are both captured, the radial transition moves with h(x).
        A number: the base is FIXED at that depth and the thickness channel is
        NOT READ AT ALL. The field is then the same step everywhere and the
        interpolation of ``thickness`` is wasted work; ``requires`` still names
        it, so a source must still provide it, but nothing consumes it.

    ``default_thickness_km`` fills the thickness channel at ``too_far``
    nodes — 100 for lithosphere-style sources, meaning "no seed nearby, assume
    a 100 km plate". It has no effect when ``base_depth_km`` is set.

    ``faded`` (default ``False``): keep the uniform ``NoAmplitude`` lateral
    term. ``True`` swaps in ``SurfaceAmplitude``, which reads a source-published
    ``surface_amplitude`` channel (e.g. an oceanic-only fade) and adds it to
    ``requires``. The base/depth is untouched; only the lateral strength varies.

    A thin preset: ``QuinticStep`` radial x (``FixedBase`` or ``DataBase``) x
    (``NoAmplitude`` or, when ``faded``, ``SurfaceAmplitude``). ``requires``
    stays ``{"thickness"}`` even with a fixed base (the contract is unchanged),
    so a source must still provide it.
    """

    def __init__(
        self,
        width_km: float = 10.0,
        *,
        base_depth_km: float | None = None,
        default_thickness_km: float = 0.0,
        faded: bool = False,
    ):
        # Validate here so the messages and their order match the historical
        # contract, including validating default_thickness_km even when a fixed
        # base means it is never read.
        if base_depth_km is not None and base_depth_km <= 0:
            raise ValueError(
                f"base_depth_km must be positive, got {base_depth_km}"
            )
        if default_thickness_km < 0:
            raise ValueError(
                f"default_thickness_km must be non-negative, "
                f"got {default_thickness_km}"
            )
        base = (FixedBase(base_depth_km) if base_depth_km is not None
                else DataBase(default_thickness_km))
        # ``faded`` reads a source-published ``surface_amplitude`` channel (e.g.
        # an oceanic-only fade); None/False keeps the uniform-amplitude step.
        lateral = SurfaceAmplitude() if faded else NoAmplitude()
        super().__init__(QuinticStep(width_km), base, lateral)
        # Fixed contract: thickness is always required (even with a fixed base,
        # where the channel is not consumed); the fade adds surface_amplitude.
        self.requires = frozenset({"thickness"}) | (
            {"surface_amplitude"} if faded else set()
        )
        self.width_km = width_km
        self.base_depth_km = base_depth_km
        self.default_thickness_km = default_thickness_km
        self.faded = faded


class MaskedQuinticOutput(IndicatorOutput):
    """Indicator for a bounded region whose depth comes from data.

    Consumes two channels rather than one, which is the whole point. A
    polygon source publishes ``masked_thickness`` (the region's depth where it
    exists, zero on the background) and ``membership`` (1 on seeds, 0 on the
    background). The kNN blend smooths both, giving

        masked_thickness_blend = m * h     membership_blend = m

    where ``m`` is the local in-region weight fraction and ``h`` the
    seed-weighted depth. `QuinticOutput` sees only the first of those and
    cannot separate them: it reads the product as a depth, so the base shallows
    toward the surface across the region's edge even though the data says
    nothing of the sort. That mistake is now unspellable rather than merely
    discouraged — the channel is named ``masked_thickness``, not ``thickness``,
    so a `QuinticOutput` (which requires ``thickness``) fails the connector's
    ``requires <= provides`` check against a polygon source outright. Dividing
    recovers both factors exactly — the lateral extent is ``m``, and the base
    depth is the data's own depth right up to the boundary.

    Note what this buys beyond correctness at the edge. Because the two facts
    are separated before either is used, the interpolation bandwidth stops
    changing them: a wider kernel spreads the transition over more nodes but
    leaves the recovered depth and the interior amplitude alone. The kernel
    width becomes a numerical smoothness choice rather than one that moves
    the physics.

    ``base_depth_km`` overrides the per-node depth with a fixed one, for
    regions with no meaningful depth data. ``width_km`` sets the radial decay
    below the base, as in `QuinticOutput`.

    A thin preset: ``QuinticStep`` radial x (``DeblendedBase`` or ``FixedBase``)
    x ``MembershipAmplitude``. The deblended base and the membership amplitude
    read the SAME clipped, too_far-zeroed m (via ``_clip_membership``), so the
    recovered depth and the lateral extent stay consistent.
    """

    def __init__(
        self,
        width_km: float = 10.0,
        *,
        base_depth_km: float | None = None,
    ):
        if base_depth_km is not None and base_depth_km <= 0:
            raise ValueError(
                f"base_depth_km must be positive, got {base_depth_km}"
            )
        base = (FixedBase(base_depth_km) if base_depth_km is not None
                else DeblendedBase())
        super().__init__(QuinticStep(width_km), base, MembershipAmplitude())
        # Fixed contract: both channels are required even with a fixed base.
        self.requires = frozenset({"masked_thickness", "membership"})
        self.width_km = width_km
        self.base_depth_km = base_depth_km


class GeothermERFOutput(OutputStrategy):
    """Erf-based normalised oceanic geotherm.

    Needs both per-point thickness (z_LAB) and seafloor age. Target nodes
    with no nearby seed (``too_far``) are assigned ``too_far_age_myr`` as a
    fallback so the erf profile flattens to ~linear over old, conduction-
    dominated lithosphere. This is distinct from the producer's
    ``default_continental_age_myr``, which sets the age column carried by
    *tracked continental seeds* — that's a property of the source data, not of
    the geotherm's fallback.
    """

    requires = frozenset({"thickness", "age"})

    def __init__(
        self,
        kappa: float = 1e-6,
        default_thickness_km: float = 100.0,
        too_far_age_myr: float = 500.0,
        geotherm: Callable | None = None,
    ):
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if default_thickness_km <= 0:
            raise ValueError(
                f"default_thickness_km must be positive, "
                f"got {default_thickness_km}"
            )
        if too_far_age_myr <= 0:
            raise ValueError(
                f"too_far_age_myr must be positive, got {too_far_age_myr}"
            )
        self.kappa = kappa
        self.default_thickness_km = default_thickness_km
        self.too_far_age_myr = too_far_age_myr
        self._geotherm = geotherm or ocean_erf_normalized

    def compute(self, interpolated, r_target, too_far, mesh):
        thickness_km = interpolated["thickness"].copy()
        age_myr = interpolated["age"].copy()
        thickness_km[too_far] = self.default_thickness_km
        age_myr[too_far] = self.too_far_age_myr

        depth_m = (mesh.r_outer - r_target) * mesh.depth_scale * 1e3
        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m, age_myr, self.kappa)
        return np.clip(T_norm, 0.0, 1.0)


class GeothermLinearOutput(OutputStrategy):
    """Linear normalised geotherm for continental / polygon-bounded regions.

    Outside the region (``too_far``) the output is 1 — mantle temperature.
    Inside, T_norm = z / z_LAB. Matches the current PolygonGeotherm semantics
    where the geotherm is only defined where a polygon claims it; everywhere
    else the surrounding mantle is presumed adiabatic at T_norm=1.
    """

    requires = frozenset({"thickness"})

    def __init__(
        self,
        geotherm: Callable | None = None,
    ):
        self._geotherm = geotherm or continental_linear

    def compute(self, interpolated, r_target, too_far, mesh):
        thickness_km = interpolated["thickness"]
        depth_m = (mesh.r_outer - r_target) * mesh.depth_scale * 1e3
        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m)
        T_norm = np.clip(T_norm, 0.0, 1.0)
        T_norm[too_far] = 1.0
        return T_norm


class MaskedGeothermLinearOutput(OutputStrategy):
    """Linear normalised geotherm for a bounded (polygon) region.

    The geotherm counterpart of `MaskedQuinticOutput`. A polygon source's depth
    channel is ``masked_thickness`` — depth ``h`` multiplied by membership ``m``
    once the kNN blend has run — so a plain `GeothermLinearOutput` reading it as
    the LAB depth would shallow the LAB toward the surface across the region's
    edge, the same two-facts error the indicator avoids. This deblends
    ``z_LAB = masked_thickness / max(m, MEMBERSHIP_FLOOR)`` so the profile uses
    the data's own depth right up to the boundary.

    Inside the region T_norm = z / z_LAB (`continental_linear`). Outside — a
    ``too_far`` node, or one whose blended membership has fallen to/below
    ``MEMBERSHIP_FLOOR`` — the surrounding mantle is presumed adiabatic at
    T_norm = 1.

    Setting T_norm = 1 there **explicitly** is load-bearing, not decorative. If
    ``z_LAB`` were instead allowed to fall to 0 outside and the result left to
    `continental_linear`, its guarded branch (``z_lab <= 0``) returns 0 —
    SURFACE temperature — which is the opposite of the mantle value wanted
    outside a bounded region. `MaskedQuinticOutput`'s identical-looking deblend
    is safe only because that class multiplies by ``m`` afterwards, damping the
    exterior to zero; a geotherm has no such trailing multiply, so the floor
    must be handled here. Easy to get backwards.

    Plain `GeothermLinearOutput` is kept for lithosphere-style sources, whose
    ``thickness`` channel is a genuine depth everywhere and needs no deblend.
    """

    requires = frozenset({"masked_thickness", "membership"})

    def __init__(
        self,
        geotherm: Callable | None = None,
    ):
        self._geotherm = geotherm or continental_linear

    def compute(self, interpolated, r_target, too_far, mesh):
        membership = np.clip(interpolated["membership"], 0.0, 1.0)
        # too_far is outside the region regardless of the blended weight.
        membership = np.where(too_far, 0.0, membership)
        covered = membership > MEMBERSHIP_FLOOR

        # Deblend the depth: masked_thickness = m * h, so h = masked_thickness/m
        # up to the boundary. Floor the divisor so the intermediate stays finite
        # where m -> 0; the ~covered nodes are overwritten to mantle below, so
        # the value there does not matter.
        z_lab_km = np.where(
            covered,
            interpolated["masked_thickness"] / np.maximum(
                membership, MEMBERSHIP_FLOOR
            ),
            0.0,
        )
        depth_m = (mesh.r_outer - r_target) * mesh.depth_scale * 1e3
        z_lab_m = z_lab_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m)
        T_norm = np.clip(T_norm, 0.0, 1.0)
        # Outside the region -> mantle. Set explicitly: letting z_lab=0 flow
        # through continental_linear would return 0 (surface), the opposite.
        T_norm[~covered] = 1.0
        return T_norm


class LateralFractionOutput(OutputStrategy):
    """The region's lateral coverage on its own — no radial dependence.

    Returns the blended ``membership`` channel clamped to [0, 1]: 1 well inside
    the region, 0 well outside, with a halo whose width is set by the
    interpolation kernel. This is the same quantity `MaskedQuinticOutput` uses
    for the lateral amplitude, exposed by itself.

    Useful as a diagnostic — mapping it shows exactly where the field thinks
    the region is, separately from how deep it goes — and as the standalone
    f_lat(x) for a field composed elsewhere. Because it shares the source's
    membership channel rather than duplicating the source with a constant
    property, it is guaranteed consistent with the indicator built from the
    same source.

    Requiring ``membership`` also narrows what it can be paired with: a
    total-coverage lithosphere source publishes no such channel
    and is now refused by the connector's ``requires <= provides`` check. That
    is deliberate — "the region's lateral extent" is not a question a source
    covering the whole sphere has an answer to.
    """

    requires = frozenset({"membership"})

    def compute(self, interpolated, r_target, too_far, mesh):
        frac = np.clip(interpolated["membership"].copy(), 0.0, 1.0)
        frac[too_far] = 0.0
        return frac
