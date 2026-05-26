"""Output strategies for plate-reconstruction scalar fields.

An OutputStrategy turns interpolated source arrays at target mesh nodes into a
scalar field. The current two flavours are:

  - indicator fields  — ~1 inside the region of interest, ~0 outside, with a
    smooth tanh transition at the region base (TanhOutput).

  - normalised geotherms — (T - T_surface) / (T_LAB - T_surface) in [0, 1],
    using an erf profile (oceanic, age-dependent) or a linear profile
    (continental). The "outside" value is 1 — i.e. mantle temperature.

Outputs declare what they need from the source via the class-level ``requires``
set; the connector validates this against the source's ``provides`` set at
construction time so, e.g. a polygon source paired with an erf geotherm fails loudly
rather than silently dropping the missing key.
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


# Shared radial primitive (used by every tanh-step indicator output)
def radial_tanh_step(r_target, base_r, width_nondim):
    """Smooth radial 1->0 step ``0.5 * (1 + tanh((r - base_r) / width))``.

    The single radial kernel shared by every indicator output. All radii are in
    non-dimensional mesh units. ``base_r`` may be a scalar (a fixed base depth,
    as in FadedRadialStepOutput) or a per-node array (a variable base depth read
    from the thickness channel, as in TanhOutput) -- the only thing that
    distinguishes those two outputs at the radial level.
    """
    return 0.5 * (1.0 + np.tanh((r_target - base_r) / width_nondim))


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

class TanhOutput(OutputStrategy):
    """Smooth indicator field: 1 below the lithosphere base, 0 above.

    Apply ``0.5 * (1 + tanh((r - (r_outer - thickness)) / transition))``.
    ``too_far`` target nodes get ``default_thickness_km`` as a fallback —
    100 km for lithosphere-style sources (no seed nearby ⇒ assume mantle is
    100 km below the surface), 0 km for polygon-style sources where exterior
    seeds carry zero thickness already (so missing-seed targets read as
    "outside the region" rather than as a continent).
    """

    requires = frozenset({"thickness"})

    def __init__(
        self,
        transition_width_km: float = 10.0,
        default_thickness_km: float = 100.0,
    ):
        if transition_width_km <= 0:
            raise ValueError(
                f"transition_width_km must be positive, got {transition_width_km}"
            )
        if default_thickness_km < 0:
            raise ValueError(
                f"default_thickness_km must be non-negative, "
                f"got {default_thickness_km}"
            )
        self.transition_width_km = transition_width_km
        self.default_thickness_km = default_thickness_km

    def compute(self, interpolated, r_target, too_far, mesh):
        thickness_km = interpolated["thickness"].copy()
        thickness_km[too_far] = self.default_thickness_km
        # Variable base depth: the radial step inflection follows h(x), which is
        # what makes this output non-separable (it cannot be written as a lateral
        # fraction times a fixed-depth step -- see FadedRadialStepOutput).
        base_r = mesh.r_outer - thickness_km / mesh.depth_scale
        return radial_tanh_step(
            r_target, base_r, self.transition_width_km / mesh.depth_scale
        )


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
    """Pure lateral membership fraction -- no radial dependence.

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


class FadedRadialStepOutput(OutputStrategy):
    """Decoupled lateral fade * radial lithosphere step.

    I(r, x) = f_lat(x) * S(r), where

      f_lat(x) = clip(thickness / crust_thickness_km, 0, 1)

    is the smoothed inside-fraction from the kNN interpolation -- a purely
    lateral, smooth 1 (continent) -> 0 (ocean) band whose width is set by
    InterpolationConfig.gaussian_sigma -- and

      S(r) = radial_tanh_step(r, r_outer - crust/depth, w_r)

    is the radial lithosphere step at a FIXED base depth, with sharpness w_r.

    This keeps the radial 1->0 transition at the lithosphere base intact while
    fixing the two artefacts of TanhOutput-on-uniform-thickness: the ocean no
    longer floors at 0.5 (f_lat = 0 zeroes the whole column, surface included),
    and the lateral transition is the full smooth fraction rather than a
    tanh-saturated fringe. The lateral fade scales the *amplitude* of the
    continental signal rather than moving the base depth around.

    Because the base depth here is fixed, this output is separable into a lateral
    factor times a single radial profile. TanhOutput is the non-separable
    generalisation: it reuses the same radial_tanh_step primitive but with a
    per-node base depth read from the thickness channel, so its transition depth
    moves with x and it carries no lateral factor. The two cannot be reduced to
    one another (a separable product cannot slide the transition depth around);
    they only share the radial_tanh_step kernel.
    """

    requires = frozenset({"thickness"})

    def __init__(self, crust_thickness_km: float = 50.0, radial_width_km: float = 10.0):
        if crust_thickness_km <= 0:
            raise ValueError(
                f"crust_thickness_km must be positive, got {crust_thickness_km}"
            )
        if radial_width_km <= 0:
            raise ValueError(
                f"radial_width_km must be positive, got {radial_width_km}"
            )
        self.crust_thickness_km = crust_thickness_km
        self.radial_width_km = radial_width_km

    def compute(self, interpolated, r_target, too_far, mesh):
        thickness_km = interpolated["thickness"].copy()
        thickness_km[too_far] = 0.0
        f_lat = np.clip(thickness_km / self.crust_thickness_km, 0.0, 1.0)

        base_r = mesh.r_outer - self.crust_thickness_km / mesh.depth_scale
        return f_lat * radial_tanh_step(
            r_target, base_r, self.radial_width_km / mesh.depth_scale
        )
