"""Create scalar fields from interpolated source channels.

`GlobalLayerIndicator` creates an indicator for a layer that covers the sphere.
`BoundedLayerIndicator` creates an indicator for a bounded layer.
`LayerIndicator` composes a radial transition, a base depth, and a lateral weight.

Each output lists its required source channels in `requires`.
The connector rejects a source that does not provide these channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.special import erf


# Mesh geometry
@dataclass(frozen=True)
class MeshConfig:
    """Define mesh geometry for conversions from radius to physical depth.

    `r_outer` is the surface radius in non-dimensional mesh units.
    `depth_scale` is the depth in kilometres per non-dimensional radial unit.
    """

    r_outer: float = 2.208
    depth_scale: float = 2890.0

    def __post_init__(self):
        if self.r_outer <= 0:
            raise ValueError(f"r_outer must be positive, got {self.r_outer}")
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")


# Geotherm functions (used by HalfSpaceCoolingGeotherm / LinearGeotherm)
def ocean_erf_normalized(depth_m, z_lab_m, age_myr, thermal_diffusivity_m2_per_s):
    """Return a normalised half-space cooling geotherm.

    The profile is `erf(z / a) / erf(z_lab / a)`.
    Here, `a = 2 * sqrt(thermal_diffusivity_m2_per_s * t)`.
    The result is zero at the surface and one at the lithosphere base.
    A linear profile avoids unstable division when the denominator approaches zero.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)
    age_myr = np.asarray(age_myr, dtype=float)

    age_sec = np.maximum(age_myr, 0.0) * 3.15576e13
    a = 2.0 * np.sqrt(thermal_diffusivity_m2_per_s * np.maximum(age_sec, 1.0))

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
    """Return the normalised linear profile `z / z_lab`."""
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)
    result = np.zeros_like(depth_m)
    valid = z_lab_m > 0
    result[valid] = depth_m[valid] / z_lab_m[valid]
    return np.clip(result, 0.0, 1.0)


# Shared radial primitive (used by every indicator output)
def radial_quintic_step(r_target, base_r, width_nondim):
    """Return a one-sided quintic radial transition.

    The result is one at or above `base_r`.
    The result is zero at or below `base_r - width_nondim`.
    The first and second derivatives are continuous at both limits.
    The midpoint lies halfway through the transition width.
    All inputs use nondimensional mesh units.
    """
    t = np.clip(
        (np.asarray(r_target, dtype=float) - base_r) / width_nondim + 1.0,
        0.0, 1.0,
    )
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


# OutputStrategy ABC
class OutputStrategy(ABC):
    """Map interpolated source channels to a target field.

    `requires` lists each required channel.
    It does not include the `xyz` coordinate array.
    """

    requires: frozenset[str]

    @abstractmethod
    def compute(
        self,
        interpolated: dict[str, np.ndarray],
        r_target: np.ndarray,
        outside_source_range: np.ndarray,
        mesh: MeshConfig,
    ) -> np.ndarray:
        """Return the scalar field at the target points.

        Args:
            interpolated: dict of arrays, one per key in ``self.requires``,
                already kNN-interpolated onto target coords.
            r_target: norms of target coords (one value per target node).
            outside_source_range: boolean mask. True where no source point is within
                ``InterpolationConfig.max_source_separation_rad``.
            mesh: mesh geometry for radius↔depth conversion.
        """


# This floor keeps membership correction finite near a region boundary.
MEMBERSHIP_FLOOR = 1e-3


class LateralWeight(Protocol):
    """Define a strategy that supplies the lateral part of an indicator."""

    requires: frozenset[str]

    def weight(
        self,
        interpolated: dict[str, np.ndarray],
        outside_source_range: np.ndarray,
    ) -> np.ndarray | float:
        """Return a lateral weight between zero and one."""
        ...


def _clip_membership(interpolated, outside_source_range):
    """Return membership between zero and one within the source range."""
    m = np.clip(interpolated["membership"], 0.0, 1.0)
    return np.where(outside_source_range, 0.0, m)


class RadialQuinticTransition:
    """Apply a quintic transition below the base depth."""

    requires = frozenset()

    def __init__(self, base_transition_width_km: float = 10.0):
        if base_transition_width_km <= 0:
            raise ValueError(
                "base_transition_width_km must be positive, "
                f"got {base_transition_width_km}"
            )
        self.base_transition_width_km = base_transition_width_km

    def step(self, r_target, base_r, mesh):
        return radial_quintic_step(
            r_target, base_r, self.base_transition_width_km / mesh.depth_scale
        )


class FixedBaseDepth:
    """Use one base depth for all target nodes."""

    requires = frozenset()

    def __init__(self, fixed_base_depth_km: float):
        if fixed_base_depth_km <= 0:
            raise ValueError(
                f"fixed_base_depth_km must be positive, got {fixed_base_depth_km}"
            )
        self.fixed_base_depth_km = fixed_base_depth_km

    def base_r(self, interpolated, outside_source_range, mesh):
        return mesh.r_outer - self.fixed_base_depth_km / mesh.depth_scale


class InterpolatedBaseDepth:
    """Read each base depth from the `thickness` channel.

    `fallback_thickness_km` replaces values outside the source range.
    """

    requires = frozenset({"thickness"})

    def __init__(self, fallback_thickness_km: float = 0.0):
        if fallback_thickness_km < 0:
            raise ValueError(
                f"fallback_thickness_km must be non-negative, "
                f"got {fallback_thickness_km}"
            )
        self.fallback_thickness_km = fallback_thickness_km

    def base_r(self, interpolated, outside_source_range, mesh):
        thickness_km = interpolated["thickness"].copy()
        thickness_km[outside_source_range] = self.fallback_thickness_km
        return mesh.r_outer - thickness_km / mesh.depth_scale


class MembershipCorrectedBaseDepth:
    """Recover base depth from membership-weighted thickness.

    The source channel contains `masked_thickness = membership * thickness`.
    Division by membership recovers thickness within the bounded region.
    Nodes below `MEMBERSHIP_FLOOR` receive a base depth of zero.
    """

    requires = frozenset({"masked_thickness", "membership"})

    def base_r(self, interpolated, outside_source_range, mesh):
        membership = _clip_membership(interpolated, outside_source_range)
        covered = membership > MEMBERSHIP_FLOOR
        thickness_km = np.where(
            covered,
            interpolated["masked_thickness"] / np.maximum(
                membership, MEMBERSHIP_FLOOR
            ),
            0.0,
        )
        return mesh.r_outer - thickness_km / mesh.depth_scale


class UniformLateralWeight:
    """Return a lateral weight of one at all target nodes."""

    requires = frozenset()

    def weight(self, interpolated, outside_source_range):
        return 1.0


class MembershipLateralWeight:
    """Use the interpolated membership channel as the lateral weight."""

    requires = frozenset({"membership"})

    def weight(self, interpolated, outside_source_range):
        return _clip_membership(interpolated, outside_source_range)


class MappedMembershipWeight:
    """Map membership to a clipped lateral weight.

    `mapping` accepts membership between zero and one.
    The returned weight is clipped to the same interval.
    """

    requires = frozenset({"membership"})

    def __init__(self, mapping):
        self.mapping = mapping

    def weight(self, interpolated, outside_source_range):
        m = _clip_membership(interpolated, outside_source_range)
        return np.clip(self.mapping(m), 0.0, 1.0)


class SourceLateralWeight:
    """Read the lateral weight from the source channel.

    The output clips each source value to the interval from zero to one.
    A target node outside the source range receives a weight of one.
    This fallback prevents missing source points from suppressing the layer.
    """

    requires = frozenset({"lateral_weight"})

    def weight(self, interpolated, outside_source_range):
        lateral_weight = interpolated["lateral_weight"].copy()
        lateral_weight[outside_source_range] = 1.0
        return np.clip(lateral_weight, 0.0, 1.0)


class LayerIndicator(OutputStrategy):
    """Compose a radial transition, a base depth, and a lateral weight.

    The radial transition defines the vertical profile at each target node.
    The base-depth strategy locates that profile in radius.
    The lateral-weight strategy scales its value across the sphere.
    `requires` combines the source channels that these strategies need.
    """

    def __init__(
        self,
        radial_transition,
        base_depth,
        lateral_weight: LateralWeight,
    ):
        self.radial_transition = radial_transition
        self.base_depth = base_depth
        self.lateral_weight = lateral_weight
        self.requires = (
            radial_transition.requires
            | base_depth.requires
            | lateral_weight.requires
        )

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        base_r = self.base_depth.base_r(
            interpolated, outside_source_range, mesh
        )
        step = self.radial_transition.step(r_target, base_r, mesh)
        return step * self.lateral_weight.weight(
            interpolated, outside_source_range
        )


# ---------------------------------------------------------------------------
# Concrete outputs
# ---------------------------------------------------------------------------

class GlobalLayerIndicator(LayerIndicator):
    """Create an indicator field for a layer that covers the sphere.

    `base_transition_width_km` sets the radial transition width in kilometers.
    `fixed_base_depth_km` sets one base depth for all target nodes.
    If this value is `None`, the `thickness` channel sets each base depth.

    `fallback_thickness_km` replaces thickness outside the source range.
    This value has no effect when `fixed_base_depth_km` is set.

    `lateral_weight` selects a lateral-weight strategy.
    The default strategy returns one at all target nodes.
    """

    def __init__(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
        fallback_thickness_km: float = 0.0,
        lateral_weight: LateralWeight | None = None,
    ):
        # Validate here so the messages and their order match the historical
        # contract, including validating fallback_thickness_km even when a fixed
        # base means it is never read.
        if fixed_base_depth_km is not None and fixed_base_depth_km <= 0:
            raise ValueError(
                f"fixed_base_depth_km must be positive, got {fixed_base_depth_km}"
            )
        if fallback_thickness_km < 0:
            raise ValueError(
                f"fallback_thickness_km must be non-negative, "
                f"got {fallback_thickness_km}"
            )
        base = (FixedBaseDepth(fixed_base_depth_km) if fixed_base_depth_km is not None
                else InterpolatedBaseDepth(fallback_thickness_km))
        if lateral_weight is None:
            lateral_weight = UniformLateralWeight()
        super().__init__(
            RadialQuinticTransition(base_transition_width_km),
            base,
            lateral_weight,
        )
        # Fixed contract: thickness is always required (even with a fixed base,
        # where the channel is not consumed). The strategy adds its channels.
        self.requires = frozenset({"thickness"}) | lateral_weight.requires
        self.base_transition_width_km = base_transition_width_km
        self.fixed_base_depth_km = fixed_base_depth_km
        self.fallback_thickness_km = fallback_thickness_km
        self.lateral_weight = lateral_weight


class BoundedLayerIndicator(LayerIndicator):
    """Create an indicator field for a bounded layer.

    The `membership` channel sets the lateral weight.
    The `masked_thickness` channel contains thickness multiplied by membership.
    The output removes membership before it calculates the base depth.

    `base_transition_width_km` sets the radial transition width in kilometers.
    `fixed_base_depth_km` sets one base depth for all target nodes.
    If this value is `None`, the corrected thickness sets each base depth.
    """

    def __init__(
        self,
        base_transition_width_km: float = 10.0,
        *,
        fixed_base_depth_km: float | None = None,
    ):
        if fixed_base_depth_km is not None and fixed_base_depth_km <= 0:
            raise ValueError(
                f"fixed_base_depth_km must be positive, got {fixed_base_depth_km}"
            )
        base = (FixedBaseDepth(fixed_base_depth_km) if fixed_base_depth_km is not None
                else MembershipCorrectedBaseDepth())
        super().__init__(
            RadialQuinticTransition(base_transition_width_km),
            base,
            MembershipLateralWeight(),
        )
        # Fixed contract: both channels are required even with a fixed base.
        self.requires = frozenset({"masked_thickness", "membership"})
        self.base_transition_width_km = base_transition_width_km
        self.fixed_base_depth_km = fixed_base_depth_km


class HalfSpaceCoolingGeotherm(OutputStrategy):
    """Create a normalised half-space cooling geotherm.

    `thermal_diffusivity_m2_per_s` sets thermal diffusivity in square metres per second.
    `fallback_thickness_km` replaces thickness outside the source range.
    `fallback_age_myr` replaces material age outside the source range.
    """

    requires = frozenset({"thickness", "age"})

    def __init__(
        self,
        thermal_diffusivity_m2_per_s: float = 1e-6,
        fallback_thickness_km: float = 100.0,
        fallback_age_myr: float = 500.0,
        geotherm: Callable | None = None,
    ):
        if thermal_diffusivity_m2_per_s <= 0:
            raise ValueError(
                "thermal_diffusivity_m2_per_s must be positive, "
                f"got {thermal_diffusivity_m2_per_s}"
            )
        if fallback_thickness_km <= 0:
            raise ValueError(
                f"fallback_thickness_km must be positive, "
                f"got {fallback_thickness_km}"
            )
        if fallback_age_myr <= 0:
            raise ValueError(
                f"fallback_age_myr must be positive, got {fallback_age_myr}"
            )
        self.thermal_diffusivity_m2_per_s = thermal_diffusivity_m2_per_s
        self.fallback_thickness_km = fallback_thickness_km
        self.fallback_age_myr = fallback_age_myr
        self._geotherm = geotherm or ocean_erf_normalized

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        thickness_km = interpolated["thickness"].copy()
        age_myr = interpolated["age"].copy()
        thickness_km[outside_source_range] = self.fallback_thickness_km
        age_myr[outside_source_range] = self.fallback_age_myr

        depth_m = (mesh.r_outer - r_target) * mesh.depth_scale * 1e3
        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(
            depth_m,
            z_lab_m,
            age_myr,
            self.thermal_diffusivity_m2_per_s,
        )
        return np.clip(T_norm, 0.0, 1.0)


class LinearGeotherm(OutputStrategy):
    """Create a normalised linear geotherm from unmasked thickness.

    Within the source range, the result is `z / z_lab`.
    Outside the source range, the result is one for mantle temperature.
    """

    requires = frozenset({"thickness"})

    def __init__(
        self,
        geotherm: Callable | None = None,
    ):
        self._geotherm = geotherm or continental_linear

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        thickness_km = interpolated["thickness"]
        depth_m = (mesh.r_outer - r_target) * mesh.depth_scale * 1e3
        z_lab_m = thickness_km * 1e3
        T_norm = self._geotherm(depth_m, z_lab_m)
        T_norm = np.clip(T_norm, 0.0, 1.0)
        T_norm[outside_source_range] = 1.0
        return T_norm


class BoundedLinearGeotherm(OutputStrategy):
    """Create a linear geotherm for a bounded region.

    The source channel contains `masked_thickness = membership * thickness`.
    The output removes membership before it evaluates the geotherm.
    It returns mantle temperature outside the bounded region.
    """

    requires = frozenset({"masked_thickness", "membership"})

    def __init__(
        self,
        geotherm: Callable | None = None,
    ):
        self._geotherm = geotherm or continental_linear

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        membership = np.clip(interpolated["membership"], 0.0, 1.0)
        # A node outside the source range cannot belong to the bounded region.
        membership = np.where(outside_source_range, 0.0, membership)
        covered = membership > MEMBERSHIP_FLOOR

        # Divide the weighted thickness by membership to recover the physical
        # thickness. The floor keeps this calculation finite near the boundary.
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
        # `continental_linear` returns surface temperature when its base depth is
        # zero. Set uncovered nodes explicitly to mantle temperature instead.
        T_norm[~covered] = 1.0
        return T_norm


class MembershipField(OutputStrategy):
    """Return the membership channel as a field between zero and one.

    Nodes outside the source range receive zero.
    This output has no radial dependence.
    """

    requires = frozenset({"membership"})

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        frac = np.clip(interpolated["membership"].copy(), 0.0, 1.0)
        frac[outside_source_range] = 0.0
        return frac
