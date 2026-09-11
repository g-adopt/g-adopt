"""Create scalar fields from interpolated source channels.

An output answers the second half of the question a Source cannot: given
interpolated arrays at the target nodes, what scalar field do we actually want
there? Sources say where the points are and what they carry; outputs turn that
into a lithosphere indicator, a geotherm, or a membership field. See
``gadopt.gplates.sources`` for the other half of the split.

The indicators are deliberately not written as one class per layer type. Every
one of them is the same product of three independent choices -- how the field
falls off in radius, where its base sits, and how strongly it acts laterally --
so ``LayerIndicator`` composes those three strategies and the concrete classes
(``GlobalLayerIndicator``, ``BoundedLayerIndicator``) are thin presets over it.
A new layer type is usually a new combination, not new code.

Each output declares the source channels it reads in ``requires``, which the
connector checks against the source's ``provides`` when the two are wired
together. That turns a mismatched pairing into an error at construction time
rather than a ``KeyError`` deep inside ``compute`` on the first timestep.
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

    Outputs receive target radii in the mesh's own non-dimensional units, but
    every physical quantity they work with -- lithosphere thickness, a cooling
    length, a transition width -- is in kilometres. This pair of numbers is the
    only thing needed to move between the two, and it is kept in one frozen
    object so a connector cannot pick up a mesh scale that disagrees with the
    one its outputs assume.

    Args:
        r_outer: Surface radius in non-dimensional mesh units.
        depth_scale: Depth in kilometres per non-dimensional radial unit.

    Raises:
        ValueError: If either value is not positive.
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

    The profile is ``erf(z / a) / erf(z_lab / a)`` with the cooling length
    ``a = 2 * sqrt(thermal_diffusivity_m2_per_s * t)``. Dividing by the value at
    the lithosphere base is what normalises it: the result is zero at the
    surface and one at the base, whatever the plate age, so the caller can
    scale it by a temperature contrast.

    Very young lithosphere makes that denominator approach zero and the ratio
    numerically useless, so below a small threshold the function falls back to
    the linear profile the erf ratio tends towards anyway.

    Args:
        depth_m: Depth below the surface, in metres.
        z_lab_m: Depth of the lithosphere base, in metres.
        age_myr: Material age, in millions of years. Negative values are
            treated as zero.
        thermal_diffusivity_m2_per_s: Thermal diffusivity, in square metres per
            second.

    Returns:
        The normalised temperature, clipped to the interval from zero to one.
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
    """Return the normalised linear profile ``z / z_lab``.

    This is the continental counterpart to ``ocean_erf_normalized``: a plate
    old enough to have equilibrated has a geotherm close to linear, and there
    is no age to carry.

    Args:
        depth_m: Depth below the surface, in metres.
        z_lab_m: Depth of the lithosphere base, in metres.

    Returns:
        The normalised temperature, clipped to the interval from zero to one.
        Nodes with a non-positive base depth return zero.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    z_lab_m = np.asarray(z_lab_m, dtype=float)
    result = np.zeros_like(depth_m)
    valid = z_lab_m > 0
    result[valid] = depth_m[valid] / z_lab_m[valid]
    return np.clip(result, 0.0, 1.0)


# Shared radial primitive (used by every indicator output)
def radial_quintic_step(r_target, base_r, width_nondim):
    """Return a one-sided quintic radial transition.

    The result is one at or above ``base_r``, zero at or below
    ``base_r - width_nondim``, and one half at the midpoint of the transition.
    The quintic is used rather than a linear ramp or a step because its first
    and second derivatives vanish at both ends: the field it produces is
    C2-continuous, so interpolating it onto a finite element space does not
    leave a kink at the top and bottom of the transition.

    Args:
        r_target: Target radii, in non-dimensional mesh units.
        base_r: Radius of the base of the layer, in the same units.
        width_nondim: Width of the transition, in the same units.

    Returns:
        The transition value at each target radius, between zero and one.
    """
    t = np.clip(
        (np.asarray(r_target, dtype=float) - base_r) / width_nondim + 1.0,
        0.0, 1.0,
    )
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


# OutputStrategy ABC
class OutputStrategy(ABC):
    """Map interpolated source channels to a target field.

    A subclass declares in ``requires`` the source channels its ``compute``
    reads, excluding the ``xyz`` coordinates, which every source provides. The
    connector matches that set against the source's ``provides`` when the pair
    is wired together, so an output can index ``interpolated`` without
    defensive checks.
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
            interpolated: Arrays keyed by channel name, one per key in
                ``self.requires``, already kNN-interpolated onto the target
                coordinates.
            r_target: Norms of the target coordinates, one value per target
                node.
            outside_source_range: Boolean mask, True where no source point lies
                within ``InterpolationConfig.max_source_separation_rad``.
            mesh: Mesh geometry for the conversion between radius and depth.

        Returns:
            The scalar field, one value per target node.
        """


# This floor keeps membership correction finite near a region boundary.
MEMBERSHIP_FLOOR = 1e-3


class LateralWeight(Protocol):
    """Supply the lateral part of an indicator.

    This is the factor that decides how strongly a layer acts at each point on
    the sphere, independent of depth: one everywhere for a global layer, a
    membership fraction for a layer bounded by polygons. It is a Protocol
    rather than a base class so that a caller can pass any object with the
    right two members, including a test double.

    Attributes:
        requires: Source channels the strategy reads.
    """

    requires: frozenset[str]

    def weight(
        self,
        interpolated: dict[str, np.ndarray],
        outside_source_range: np.ndarray,
    ) -> np.ndarray | float:
        """Return a lateral weight between zero and one.

        Args:
            interpolated: Arrays keyed by channel name.
            outside_source_range: Boolean mask, True where no source point is
                in range.

        Returns:
            The weight at each target node, or a single float that applies
            everywhere.
        """
        ...


def _clip_membership(interpolated, outside_source_range):
    """Clip the membership channel and zero it outside the source range.

    A node with no source point in range cannot be shown to belong to the
    bounded region, so it is treated as outside it rather than inheriting the
    membership of a distant point.

    Args:
        interpolated: Arrays keyed by channel name, including ``membership``.
        outside_source_range: Boolean mask, True where no source point is in
            range.

    Returns:
        Membership between zero and one, zero outside the source range.
    """
    m = np.clip(interpolated["membership"], 0.0, 1.0)
    return np.where(outside_source_range, 0.0, m)


class RadialQuinticTransition:
    """Apply a quintic transition below the base depth.

    Args:
        base_transition_width_km: Width of the radial transition, in
            kilometres.

    Raises:
        ValueError: If the width is not positive.
    """

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
    """Use one base depth for all target nodes.

    This is the choice for a layer of prescribed thickness, where the source
    supplies properties but not geometry.

    Args:
        fixed_base_depth_km: Base depth, in kilometres.

    Raises:
        ValueError: If the depth is not positive.
    """

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
    """Read each base depth from the ``thickness`` channel.

    Args:
        fallback_thickness_km: Thickness used where no source point is in
            range, in kilometres. The default of zero puts the base at the
            surface, which switches the layer off there.

    Raises:
        ValueError: If the fallback thickness is negative.
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

    A bounded source cannot interpolate raw thickness, because averaging a
    continental thickness with the zeros outside the polygon would thin the
    margins. It carries ``masked_thickness = membership * thickness`` instead,
    and dividing the interpolated product by the interpolated membership
    recovers the physical thickness inside the region. Below
    ``MEMBERSHIP_FLOOR`` the division is meaningless, so those nodes get a base
    depth of zero and the layer vanishes there.
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
    """Return a lateral weight of one at all target nodes.

    This is the strategy for a layer that covers the whole sphere and varies
    only in thickness.
    """

    requires = frozenset()

    def weight(self, interpolated, outside_source_range):
        return 1.0


class MembershipLateralWeight:
    """Use the interpolated membership channel as the lateral weight.

    The layer fades out across the edge of the bounded region in step with the
    interpolated membership, which avoids a hard polygon boundary in the field.
    """

    requires = frozenset({"membership"})

    def weight(self, interpolated, outside_source_range):
        return _clip_membership(interpolated, outside_source_range)


class MappedMembershipWeight:
    """Map membership through a callable to get the lateral weight.

    Use this when the layer must not follow membership linearly, for example to
    sharpen a margin or to hold full strength over most of a region.

    Args:
        mapping: Callable applied to membership between zero and one. Its
            result is clipped back to the same interval.
    """

    requires = frozenset({"membership"})

    def __init__(self, mapping):
        self.mapping = mapping

    def weight(self, interpolated, outside_source_range):
        m = _clip_membership(interpolated, outside_source_range)
        return np.clip(self.mapping(m), 0.0, 1.0)


class SourceLateralWeight:
    """Read the lateral weight from the ``lateral_weight`` source channel.

    A node outside the source range falls back to a weight of one rather than
    zero, because here a missing source point means the source has nothing to
    say about that node, not that the layer is absent. Zeroing it would let
    gaps in the source cloud punch holes in the field.
    """

    requires = frozenset({"lateral_weight"})

    def weight(self, interpolated, outside_source_range):
        lateral_weight = interpolated["lateral_weight"].copy()
        lateral_weight[outside_source_range] = 1.0
        return np.clip(lateral_weight, 0.0, 1.0)


class LayerIndicator(OutputStrategy):
    """Compose a radial transition, a base depth, and a lateral weight.

    These three are independent: the transition sets the shape of the profile
    in radius, the base depth places that profile, and the lateral weight
    scales it across the sphere. The field is the product of the placed profile
    and the weight. ``requires`` is the union of what the three strategies
    read, so the composed output asks the source for exactly the channels its
    parts need.

    Args:
        radial_transition: Strategy with a ``step(r_target, base_r, mesh)``
            method.
        base_depth: Strategy with a
            ``base_r(interpolated, outside_source_range, mesh)`` method.
        lateral_weight: Strategy following the ``LateralWeight`` protocol.
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

    A preset over ``LayerIndicator`` for the unbounded case: oceanic
    lithosphere, or any layer present everywhere and varying only in thickness.
    The ``thickness`` channel is required whether or not the base depth is
    actually read from it, so that swapping ``fixed_base_depth_km`` on and off
    does not change what the paired source must provide.

    Args:
        base_transition_width_km: Width of the radial transition, in
            kilometres.
        fixed_base_depth_km: One base depth for all target nodes, in
            kilometres. If None, the ``thickness`` channel sets each base
            depth.
        fallback_thickness_km: Thickness used outside the source range, in
            kilometres. Validated but never read when ``fixed_base_depth_km``
            is set.
        lateral_weight: Lateral-weight strategy. Defaults to
            ``UniformLateralWeight``, which returns one everywhere.

    Raises:
        ValueError: If the fixed base depth is not positive, or the fallback
            thickness is negative.
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

    A preset over ``LayerIndicator`` for a layer confined to polygons, such as
    continental lithosphere. The ``membership`` channel supplies the lateral
    weight, and ``masked_thickness`` carries thickness already multiplied by
    membership; the base-depth strategy divides that product back out. See
    ``MembershipCorrectedBaseDepth`` for why the source masks the thickness in
    the first place.

    Both channels are required whether or not the base depth is read from
    them, so that swapping ``fixed_base_depth_km`` on and off does not change
    what the paired source must provide.

    Args:
        base_transition_width_km: Width of the radial transition, in
            kilometres.
        fixed_base_depth_km: One base depth for all target nodes, in
            kilometres. If None, the corrected thickness sets each base depth.

    Raises:
        ValueError: If the fixed base depth is not positive.
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

    The output for oceanic lithosphere, where plate age controls the thermal
    structure. The fallbacks stand in where the source cloud has no point in
    range: an old, thick plate, which is the closest thing to ambient mantle
    that a cooling profile can express.

    Args:
        thermal_diffusivity_m2_per_s: Thermal diffusivity, in square metres per
            second.
        fallback_thickness_km: Thickness used outside the source range, in
            kilometres.
        fallback_age_myr: Material age used outside the source range, in
            millions of years.
        geotherm: Profile function. Defaults to ``ocean_erf_normalized``.

    Raises:
        ValueError: If the diffusivity, the fallback thickness, or the fallback
            age is not positive.
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

    Inside the source range the result is ``z / z_lab``. Outside it the result
    is one, that is, mantle temperature, since a node with no source point in
    range sits outside the lithosphere this geotherm describes.

    Args:
        geotherm: Profile function. Defaults to ``continental_linear``.
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

    The geotherm counterpart to ``BoundedLayerIndicator``: it divides
    ``masked_thickness`` by membership to recover the physical thickness before
    evaluating the profile, and returns mantle temperature outside the region.

    Args:
        geotherm: Profile function. Defaults to ``continental_linear``.
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
        # ``continental_linear`` returns surface temperature when its base depth is
        # zero. Set uncovered nodes explicitly to mantle temperature instead.
        T_norm[~covered] = 1.0
        return T_norm


class MembershipField(OutputStrategy):
    """Return the membership channel as a field between zero and one.

    Useful on its own for diagnostics and for masking other fields: it shows
    where a bounded source considers itself present, with no radial dependence
    at all. Nodes outside the source range are zero.
    """

    requires = frozenset({"membership"})

    def compute(self, interpolated, r_target, outside_source_range, mesh):
        frac = np.clip(interpolated["membership"].copy(), 0.0, 1.0)
        frac[outside_source_range] = 0.0
        return frac
