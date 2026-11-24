"""Utilities for constructing regularised earthquake stress-glut sources.

Moment-tensor formulae commonly use a local north-east-down (NED) coordinate
system, whereas a spherical G-ADOPT mesh uses Earth-centred Cartesian (ECEF)
coordinates.  This module keeps that rotation and the source
nondimensionalisation explicit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from firedrake import Function, FunctionSpace, SpatialCoordinate, assemble, dx, exp, pi


def ned_basis(latitude_degrees: float, longitude_degrees: float) -> np.ndarray:
    """Return the local NED basis as columns of an ECEF rotation matrix."""
    latitude = np.deg2rad(latitude_degrees)
    longitude = np.deg2rad(longitude_degrees)

    north = np.array(
        [
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ]
    )
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    down = -np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    return np.column_stack((north, east, down))


def moment_tensor_ned(
    scalar_moment: float,
    strike_degrees: float,
    dip_degrees: float,
    rake_degrees: float,
) -> np.ndarray:
    """Construct a double-couple moment tensor in north-east-down coordinates.

    Strike is clockwise from north, dip is measured down from horizontal, and
    rake is measured within the fault plane.  The component formulae follow
    Bock (2012), NMSOP-2 Information Sheet IS 3.8, equation (5).
    """
    strike = np.deg2rad(strike_degrees)
    dip = np.deg2rad(dip_degrees)
    rake = np.deg2rad(rake_degrees)

    m_nn = -scalar_moment * (
        np.sin(dip) * np.cos(rake) * np.sin(2 * strike)
        + np.sin(2 * dip) * np.sin(rake) * np.sin(strike) ** 2
    )
    m_ne = scalar_moment * (
        np.sin(dip) * np.cos(rake) * np.cos(2 * strike)
        + 0.5 * np.sin(2 * dip) * np.sin(rake) * np.sin(2 * strike)
    )
    m_nd = -scalar_moment * (
        np.cos(dip) * np.cos(rake) * np.cos(strike)
        + np.cos(2 * dip) * np.sin(rake) * np.sin(strike)
    )
    m_ee = scalar_moment * (
        np.sin(dip) * np.cos(rake) * np.sin(2 * strike)
        - np.sin(2 * dip) * np.sin(rake) * np.cos(strike) ** 2
    )
    m_ed = -scalar_moment * (
        np.cos(dip) * np.cos(rake) * np.sin(strike)
        - np.cos(2 * dip) * np.sin(rake) * np.cos(strike)
    )
    m_dd = scalar_moment * np.sin(2 * dip) * np.sin(rake)

    return np.array(
        [
            [m_nn, m_ne, m_nd],
            [m_ne, m_ee, m_ed],
            [m_nd, m_ed, m_dd],
        ]
    )


def moment_tensor_ecef(
    scalar_moment: float,
    strike_degrees: float,
    dip_degrees: float,
    rake_degrees: float,
    latitude_degrees: float,
    longitude_degrees: float,
) -> np.ndarray:
    """Construct a double-couple moment tensor in ECEF coordinates."""
    moment_ned = moment_tensor_ned(
        scalar_moment,
        strike_degrees,
        dip_degrees,
        rake_degrees,
    )
    rotation = ned_basis(latitude_degrees, longitude_degrees)
    return rotation @ moment_ned @ rotation.T


def spherical_point_ecef(
    latitude_degrees: float,
    longitude_degrees: float,
    radius: float,
) -> np.ndarray:
    """Return ECEF coordinates on a sphere with the supplied radius."""
    latitude = np.deg2rad(latitude_degrees)
    longitude = np.deg2rad(longitude_degrees)
    return radius * np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def build_rectangular_fault_patches(
    *,
    centre_latitude_deg: float,
    centre_longitude_deg: float,
    strike_deg: float,
    dip_deg: float,
    vertical_extent_m: float,
    sections: Sequence[tuple[float, float, float]],
    shear_modulus_Pa: float,
    target_moment_Nm: float | None,
    strike_spacing_m: float = 12_000.0,
    down_dip_order: int = 5,
    radius_m: float = 6_371_000.0,
) -> list[dict]:
    """Build moment-tensor patches for a surface-breaking rectangular fault.

    Args:
      centre_latitude_deg:
        Latitude in degrees of the surface projection of the fault centre.
        Uses geographic latitude as the angle on the model sphere.
      centre_longitude_deg:
        Longitude in degrees east of the same point.
      strike_deg:
        Strike measured clockwise from north, in degrees.
      dip_deg:
        Dip measured down from horizontal, in degrees.
      vertical_extent_m:
        Vertical distance in metres from the surface to the bottom of the fault.
      sections:
        Tuples of length in metres, slip in metres and rake in degrees,
        ordered along strike. Each section spans the full down-dip width.
      shear_modulus_Pa:
        Uniform shear modulus in pascals used to compute moment from slip.
      target_moment_Nm:
        Target scalar moment sum in N m. Use None for the moments derived
        from slip, area and shear modulus. A target scales all moments uniformly.
      strike_spacing_m:
        Length scale in metres used to choose the number of Gauss points
        along strike. The number is rounded up from section length over this scale.
      down_dip_order:
        Number of Gauss points down dip per section.
      radius_m:
        Radius of the model sphere in metres.

    Returns:
        Patch records ordered by section, then along strike, then down dip.
        Each record gives angular location in degrees, depth in kilometres,
        Earth-centred Cartesian position in metres, and scalar moment and
        Cartesian moment tensor in N m.

    Raises:
        ValueError: If the geometry, material, quadrature or scalar moment is invalid.
    """
    if not 0 < dip_deg <= 90 or not 0 < vertical_extent_m < radius_m:
        raise ValueError('Require a downward dip and a positive fault extent within the sphere')
    if shear_modulus_Pa <= 0:
        raise ValueError('shear_modulus_Pa must be positive')
    if strike_spacing_m <= 0:
        raise ValueError('strike_spacing_m must be positive')
    if down_dip_order < 1:
        raise ValueError('down_dip_order must be at least 1')
    if not sections or any(length_m <= 0 or slip_m < 0 for length_m, slip_m, _ in sections):
        raise ValueError('Provide sections with positive lengths and nonnegative slip')

    strike, dip = np.deg2rad(strike_deg), np.deg2rad(dip_deg)
    width_m = vertical_extent_m / np.sin(dip)
    slip_moment_Nm = sum(
        length_m * slip_m * width_m * shear_modulus_Pa
        for length_m, slip_m, _ in sections
    )
    if not np.isfinite(slip_moment_Nm) or slip_moment_Nm <= 0:
        raise ValueError('The prescribed slips must give a finite positive scalar moment')
    if target_moment_Nm is not None and (not np.isfinite(target_moment_Nm) or target_moment_Nm <= 0):
        raise ValueError('The target moment must be finite and positive')

    north, east, down = ned_basis(centre_latitude_deg, centre_longitude_deg).T
    up = -down
    along_strike = north * np.cos(strike) + east * np.sin(strike)
    across_strike = -north * np.sin(strike) + east * np.cos(strike)

    dip_nodes, dip_weights = leggauss(down_dip_order)
    down_dip_distances_m = 0.5 * width_m * (dip_nodes + 1)
    down_dip_weights_m = 0.5 * width_m * dip_weights
    section_start_m = -0.5 * sum(length_m for length_m, _, _ in sections)

    patches = []
    for length_m, slip_m, rake_deg in sections:
        strike_nodes, strike_weights = leggauss(int(np.ceil(length_m / strike_spacing_m)))
        strike_distances_m = section_start_m + 0.5 * length_m * (strike_nodes + 1)
        strike_weights_m = 0.5 * length_m * strike_weights
        for strike_m, strike_weight_m in zip(strike_distances_m, strike_weights_m):
            for down_dip_m, dip_weight_m in zip(down_dip_distances_m, down_dip_weights_m):
                # Map the horizontal offset along a great circle from the centre.
                horizontal = (
                    strike_m * along_strike
                    + (down_dip_m - 0.5 * width_m) * np.cos(dip) * across_strike
                )
                distance_m = float(np.linalg.norm(horizontal))
                direction = up
                if distance_m > 0:
                    angle = distance_m / radius_m
                    direction = up * np.cos(angle) + horizontal / distance_m * np.sin(angle)
                latitude_deg = float(np.rad2deg(np.arcsin(direction[2])))
                longitude_deg = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
                depth_km = float(down_dip_m * np.sin(dip) / 1000)

                # Each pair of quadrature weights represents a patch area.
                moment_Nm = shear_modulus_Pa * slip_m * strike_weight_m * dip_weight_m
                if target_moment_Nm is not None:
                    moment_Nm = moment_Nm * target_moment_Nm / slip_moment_Nm
                tensor = moment_tensor_ecef(
                    moment_Nm, strike_deg, dip_deg, rake_deg, latitude_deg, longitude_deg,
                )
                position = spherical_point_ecef(latitude_deg, longitude_deg, radius_m - depth_km * 1000)
                patches.append(dict(
                    latitude_deg=latitude_deg,
                    longitude_deg=longitude_deg,
                    depth_km=depth_km,
                    scalar_moment_Nm=float(moment_Nm),
                    moment_ecef_Nm=tensor.tolist(),
                    position_ecef_m=position.tolist(),
                ))
        section_start_m += length_m

    expected_moment = slip_moment_Nm if target_moment_Nm is None else target_moment_Nm
    scalar_sum_Nm = sum(patch['scalar_moment_Nm'] for patch in patches)
    if abs(scalar_sum_Nm / expected_moment - 1) >= 1e-12:
        raise ValueError('The patch moments do not sum to the expected scalar moment')
    return patches


def normalised_gaussian_delta(
    mesh,
    source_coordinates,
    standard_deviation: float,
    *,
    degree: int = 2,
    name: str = "regularised delta",
) -> tuple[Function, float]:
    """Interpolate and discretely normalise a three-dimensional Gaussian.

    The returned Function integrates to one on the actual mesh.  The second
    return value is the integral before discrete normalisation and is a useful
    source-resolution diagnostic: it should converge towards one as the mesh
    resolves the Gaussian and as boundary truncation becomes negligible.
    """
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    if len(source_coordinates) != 3:
        raise ValueError("source_coordinates must contain three components")

    source_coordinates = np.asarray(source_coordinates, dtype=float)
    space = FunctionSpace(mesh, "CG", degree)
    coordinates = SpatialCoordinate(mesh)
    squared_distance = sum(
        (coordinates[index] - source_coordinates[index]) ** 2
        for index in range(3)
    )
    prefactor = 1 / ((2 * pi) ** 1.5 * standard_deviation**3)
    delta = Function(space, name=name).interpolate(
        prefactor * exp(-squared_distance / (2 * standard_deviation**2))
    )

    unnormalised_integral = float(assemble(delta * dx))
    if not np.isfinite(unnormalised_integral) or unnormalised_integral <= 0:
        raise ValueError(
            "The regularised delta has a non-positive or non-finite integral; "
            "check the source location and mesh resolution"
        )
    delta.assign(delta / unnormalised_integral)
    return delta, unnormalised_integral


def ecef_vectors_to_enu(
    vectors,
    latitudes_degrees,
    longitudes_degrees,
) -> np.ndarray:
    """Rotate one or more ECEF vectors into local east-north-up components."""
    vectors = np.asarray(vectors, dtype=float)
    latitudes = np.broadcast_to(np.asarray(latitudes_degrees, dtype=float), vectors.shape[:-1])
    longitudes = np.broadcast_to(np.asarray(longitudes_degrees, dtype=float), vectors.shape[:-1])
    if vectors.shape[-1] != 3:
        raise ValueError("vectors must have final dimension three")

    latitude = np.deg2rad(latitudes)
    longitude = np.deg2rad(longitudes)
    east = np.stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)),
        axis=-1,
    )
    north = np.stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ),
        axis=-1,
    )
    up = np.stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )
    return np.stack(
        (
            np.sum(vectors * east, axis=-1),
            np.sum(vectors * north, axis=-1),
            np.sum(vectors * up, axis=-1),
        ),
        axis=-1,
    )
