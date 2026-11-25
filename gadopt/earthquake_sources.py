"""Utilities for constructing regularised earthquake stress-glut sources.

A finite fault is represented by patches, each with a position and a moment
tensor. The fault helpers express the local north-east-down (NED) moment
tensors in Earth-centred, Earth-fixed Cartesian coordinates (ECEF).
`build_stress_glut` spreads the patches over a volume mesh using Gaussians,
preserving their summed moment tensor.

For a nondimensional mesh, divide physical positions and the Gaussian width
by the reference length `L`. Divide the moment tensors, in N m, by
`stress_scale * L**3`. The returned field is in the stress units used by
the solver. The caller controls when the source is applied.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike

from firedrake import Function, FunctionSpace, SpatialCoordinate, TensorFunctionSpace, assemble, dx, exp, pi
from gdrift import ecef_to_ned_matrix, geodetic_to_cartesian


def moment_tensor_ned(
    scalar_moment: float,
    strike_degrees: float,
    dip_degrees: float,
    rake_degrees: float,
) -> np.ndarray:
    """Construct a double-couple moment tensor in north-east-down coordinates.

    The component formulae follow Bock (2012), NMSOP-2 Information Sheet
    IS 3.8, equation (5).

    Args:
        scalar_moment: Scalar seismic moment, typically in N m.
        strike_degrees: Strike measured clockwise from north, in degrees.
        dip_degrees: Dip measured down from horizontal, in degrees.
        rake_degrees: Slip angle in the fault plane, in degrees, with zero
            along strike.

    Returns:
        A symmetric 3-by-3 moment tensor with rows and columns ordered north,
        east and down. Its components have the units of `scalar_moment`.
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
    """Construct a double-couple moment tensor in ECEF coordinates.

    Args:
        scalar_moment: Scalar seismic moment, typically in N m.
        strike_degrees: Strike measured clockwise from north, in degrees.
        dip_degrees: Dip measured down from horizontal, in degrees.
        rake_degrees: Slip angle in the fault plane, in degrees.
        latitude_degrees: Latitude at which the local fault angles are defined.
        longitude_degrees: Longitude of the local frame, in degrees east.

    Returns:
        A 3-by-3 Cartesian moment tensor in the units of `scalar_moment`.
        With `R` the ECEF-to-NED rotation, the tensor is `R.T @ M_ned @ R`.
    """
    moment_ned = moment_tensor_ned(
        scalar_moment,
        strike_degrees,
        dip_degrees,
        rake_degrees,
    )
    ecef_to_ned = ecef_to_ned_matrix(latitude_degrees, longitude_degrees)
    return ecef_to_ned.T @ moment_ned @ ecef_to_ned


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

    Each along-strike section has uniform slip and rake. The shared down-dip
    width is the vertical extent divided by the sine of the dip. The product
    of the along-strike and down-dip Gauss–Legendre weights gives the area
    represented by each point on the rectangle. The scalar patch moment is
    shear modulus times slip times that area.

    The horizontal offset from the fault centre is mapped onto the sphere
    along a great circle. The point's radius is then reduced by its depth.
    The area weights are those of the original rectangle.

    Args:
        centre_latitude_deg: Spherical latitude in degrees of the surface
            projection of the rectangle's midpoint.
        centre_longitude_deg: Longitude of that midpoint in degrees east.
        strike_deg: Strike measured clockwise from north, in degrees.
        dip_deg: Dip measured down from horizontal, in degrees, in the range
            `(0, 90]`.
        vertical_extent_m: Depth of the bottom edge below the surface in metres.
        sections: Tuples of length in metres, slip in metres and rake in
            degrees, ordered in the direction of strike. Each section spans
            the full down-dip width.
        shear_modulus_Pa: Uniform shear modulus in pascals used to compute
            moment from slip.
        target_moment_Nm: Target scalar moment sum in N m. A target scales all
            moments uniformly. With `None`, moments follow from slip, area
            and shear modulus.
        strike_spacing_m: Length scale in metres for choosing the along-strike
            quadrature order. Each order is the section length divided by this
            scale and rounded up.
        down_dip_order: Number of Gauss points down dip in each section.
        radius_m: Radius of the model sphere in metres.

    Returns:
        Patch records ordered by section, then along strike, then down dip.
        Each record contains `latitude_deg`, `longitude_deg`, `depth_km`,
        `position_ecef_m`, `scalar_moment_Nm` and `moment_ecef_Nm`.

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

    north, east, down = ecef_to_ned_matrix(centre_latitude_deg, centre_longitude_deg)
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
                position = geodetic_to_cartesian(
                    latitude_deg, longitude_deg, depth=depth_km * 1000, earth_radius=radius_m,
                )[0]
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

    The Gaussian has unit integral in unbounded space. Interpolation onto
    the mesh and truncation at its boundaries change that integral. We divide
    by the assembled integral over the mesh so the regularised delta has
    unit integral on the model domain.

    Args:
        mesh: Three-dimensional volume mesh used to represent the source.
        source_coordinates: Three Cartesian coordinates of the Gaussian
            centre in mesh units.
        standard_deviation: Gaussian standard deviation in mesh units.
        degree: Degree of the continuous Lagrange space for the Gaussian.
        name: Name of the returned Firedrake field.

    Returns:
        The normalised scalar field and its integral before normalisation.
        The latter approaches one for a well-resolved Gaussian whose centre
        is far enough from the boundary for truncation to be negligible.
        Near a boundary, a resolved Gaussian can have a smaller integral.

    Raises:
        ValueError: If the width is nonpositive, the centre has the wrong
            number of coordinates, or the integral is nonpositive or nonfinite.
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


def build_stress_glut(
    mesh,
    centres: ArrayLike,
    moment_tensors: ArrayLike,
    standard_deviation: float,
    *,
    degree: int = 2,
) -> tuple[Function, list[float]]:
    """Assemble a fixed stress glut from Gaussian moment-tensor patches.

    Each patch contributes its moment tensor multiplied by a separately
    normalised Gaussian. The accumulation writes directly into the field's
    nodal values and is intended for prescribed source tensors.

    Args:
        mesh: Three-dimensional volume mesh on which to represent the source.
        centres: Patch positions with shape `(n, 3)` in mesh coordinates.
        moment_tensors: Patch tensors with shape `(n, 3, 3)` in the mesh's
            Cartesian basis. Divide dimensional moment tensors by the stress
            scale times the length scale cubed for a nondimensional mesh.
        standard_deviation: Common Gaussian standard deviation in mesh units.
        degree: Degree of the continuous Lagrange spaces for the Gaussians
            and stress glut.

    Returns:
        The stress-glut field and the Gaussian integrals before normalisation,
        in patch order. Integrating the assembled field over the mesh gives
        the sum of the supplied tensors.

    Raises:
        ValueError: If the mesh, patch arrays, or Gaussian width is invalid.
    """
    if mesh.topological_dimension != 3:
        raise ValueError("The stress glut requires a three-dimensional volume mesh")
    centres = np.asarray(centres, dtype=float)
    moment_tensors = np.asarray(moment_tensors, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 3 or not len(centres):
        raise ValueError("centres must have shape (n, 3) with at least one patch")
    if moment_tensors.shape != (len(centres), 3, 3):
        raise ValueError("moment_tensors must have shape (n, 3, 3) with one tensor per centre")
    if not np.isfinite(centres).all() or not np.isfinite(moment_tensors).all():
        raise ValueError("Patch centres and moment tensors must be finite")
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("standard_deviation must be finite and positive")

    glut = Function(TensorFunctionSpace(mesh, "CG", degree), name="stress glut")
    values = glut.dat.data
    integrals = []
    for centre, tensor in zip(centres, moment_tensors, strict=True):
        delta, integral = normalised_gaussian_delta(mesh, centre, standard_deviation, degree=degree)
        values[:] += delta.dat.data_ro[:, None, None] * tensor
        integrals.append(integral)
    return glut, integrals
