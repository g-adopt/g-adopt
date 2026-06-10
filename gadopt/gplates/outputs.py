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
base, decaying to exactly 0 over one transition width BELOW the base. Unlike
the symmetric tanh kernel it replaced, the transition never straddles the base,
so the surface value stays faithful — exactly 1 wherever the region has any
thickness — and the field is exactly 0 below base + width (no asymptotic tail).
The 0.5-crossing therefore sits at base + width/2 in depth, not at the base.

One consequence to keep in mind: where thickness is exactly zero the base sits
at the surface and the SURFACE NODE still reads 1 (the transition hangs just
below it). Zero-outside sources (polygon mask-and-relabel) therefore MUST pair
the step with a lateral fade (``fade_ref_km``) so the exterior column is
multiplied to 0 — an unfaded variable-depth step is only safe for sources whose
thickness never vanishes laterally without a fade, or whose fallback thickness
fills the gap (lithosphere-style ``default_thickness_km=100``).

``QuinticOutput`` covers the whole indicator family through two switches:

  ============================ ==================== ===================
  Configuration                Base depth           Lateral fade
  ============================ ==================== ===================
  QuinticOutput()              variable (per-node)  none
  QuinticOutput(base_depth_km) fixed                optional
  QuinticOutput(fade_ref_km)   variable             clip(h/ref, 0, 1)
  LateralFractionOutput        none (pure lateral)  the fraction itself
  ============================ ==================== ===================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

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


# Importable default used by the ScalarFieldConnector when the caller doesn't
# pass a MeshConfig of their own.
_DEFAULT_MESH_PARAMETERS = MeshConfig()


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
    reads exactly 1 (the tanh kernel this replaced read 0.5 there), and the
    0.5-crossing is at ``base_r - width_nondim / 2``. All radii are in
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
    ``"xyz"``) they read from. The connector validates
    ``output.requires <= source.provides`` at construction time.
    """

    requires: ClassVar[frozenset[str]]

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


# ---------------------------------------------------------------------------
# Concrete outputs
# ---------------------------------------------------------------------------

class QuinticOutput(OutputStrategy):
    """Smooth indicator field built on the one-sided quintic radial step.

    The field is exactly 1 from the surface down to the region base, decays
    to exactly 0 over ``width_km`` below it, and is optionally multiplied by
    a lateral amplitude fade. Two switches select the configuration:

    ``base_depth_km``
        ``None`` (default): the base depth is per-node, read from the
        interpolated ``thickness`` channel — deep cratonic roots and shallow
        margins are both captured, the radial transition moves with h(x).
        A number: the base is FIXED at that depth; the thickness channel
        then only feeds the fade (the separable lateral-fade x radial-step
        decomposition previously provided by FadedRadialStepOutput).

    ``fade_ref_km``
        ``None`` (default): no fade. The surface reads exactly 1 wherever
        thickness is nonzero — including where it is barely nonzero, so this
        configuration is only safe when thickness never vanishes laterally
        (or ``default_thickness_km`` fills the gaps, lithosphere-style).
        A number: the step is multiplied by ``clip(h / fade_ref_km, 0, 1)``.
        REQUIRED (in practice) for zero-outside polygon sources: without it
        the exterior surface — where the base coincides with the surface —
        reads 1 instead of 0.

    ``default_thickness_km`` fills the thickness channel at ``too_far``
    nodes: 100 for lithosphere-style sources (no seed nearby => assume a
    100 km plate), 0 for polygon-style sources where missing seeds must read
    as "outside the region".
    """

    requires = frozenset({"thickness"})

    def __init__(
        self,
        width_km: float = 10.0,
        *,
        base_depth_km: float | None = None,
        fade_ref_km: float | None = None,
        default_thickness_km: float = 0.0,
    ):
        if width_km <= 0:
            raise ValueError(f"width_km must be positive, got {width_km}")
        if base_depth_km is not None and base_depth_km <= 0:
            raise ValueError(
                f"base_depth_km must be positive, got {base_depth_km}"
            )
        if fade_ref_km is not None and fade_ref_km <= 0:
            raise ValueError(
                f"fade_ref_km must be positive, got {fade_ref_km}"
            )
        if default_thickness_km < 0:
            raise ValueError(
                f"default_thickness_km must be non-negative, "
                f"got {default_thickness_km}"
            )
        self.width_km = width_km
        self.base_depth_km = base_depth_km
        self.fade_ref_km = fade_ref_km
        self.default_thickness_km = default_thickness_km

    def compute(self, interpolated, r_target, too_far, mesh):
        thickness_km = interpolated["thickness"].copy()
        thickness_km[too_far] = self.default_thickness_km

        if self.base_depth_km is None:
            base_r = mesh.r_outer - thickness_km / mesh.depth_scale
        else:
            base_r = mesh.r_outer - self.base_depth_km / mesh.depth_scale

        result = radial_quintic_step(
            r_target, base_r, self.width_km / mesh.depth_scale
        )
        if self.fade_ref_km is not None:
            result = result * np.clip(
                thickness_km / self.fade_ref_km, 0.0, 1.0
            )
        return result


class GeothermERFOutput(OutputStrategy):
    """Erf-based normalised oceanic geotherm.

    Needs both per-point thickness (z_LAB) and seafloor age. Target nodes
    with no nearby seed (``too_far``) are assigned ``too_far_age_myr`` as a
    fallback so the erf profile flattens to ~linear over old, conduction-
    dominated lithosphere. This is distinct from
    ``LithosphereSource.default_continental_age_myr``, which sets the age
    column carried by *tracked continental seeds* — that's a property of
    the source data, not of the geotherm's fallback.
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


class LateralFractionOutput(OutputStrategy):
    """Pure lateral membership fraction — no radial dependence.

    Returns the kNN-interpolated property clamped to [0, 1]. Paired with a
    PolygonSource carrying a uniform value of 1.0 inside the polygon (0 outside),
    the interpolated value *is* the smoothed inside-fraction: 1 well inside the
    region, 0 well outside, with a smooth halo whose width is set by
    InterpolationConfig.gaussian_sigma. Used as the standalone membership channel
    f_lat(x) that the decoupled craton field multiplies against.

    NB ``requires`` names the ``"thickness"`` key only because that is the key a
    PolygonSource publishes; here it carries a membership value (1.0 inside), not
    a thickness. The name is kept to satisfy the connector's
    ``requires <= source.provides`` check.
    """

    requires = frozenset({"thickness"})

    def compute(self, interpolated, r_target, too_far, mesh):
        frac = np.clip(interpolated["thickness"].copy(), 0.0, 1.0)
        frac[too_far] = 0.0
        return frac
