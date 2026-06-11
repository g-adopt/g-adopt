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
from typing import ClassVar

import numpy as np


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
