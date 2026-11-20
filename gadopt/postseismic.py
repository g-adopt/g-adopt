"""Utilities for constructing regularised earthquake stress-glut sources.

Moment-tensor formulae commonly use a local north-east-down (NED) coordinate
system, whereas a spherical G-ADOPT mesh uses Earth-centred Cartesian (ECEF)
coordinates.  This module keeps that rotation and the source
nondimensionalisation explicit.
"""

from __future__ import annotations

import numpy as np

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

