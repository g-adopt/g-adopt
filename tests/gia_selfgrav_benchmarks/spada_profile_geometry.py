"""NumPy geometry for Spada near-surface profiles; no Firedrake required."""

import numpy as np


def profile_geometry(case, load_centre_deg, *, longitudes=None, bearings=None,
                     centre=None, extent_deg=180.0, spacing_deg=0.1):
    """Meridians or great circles through a chosen centre.

    Bearings are clockwise from geographic north at the profile centre.
    Positive signed distance follows the bearing; negative distance follows
    its opposite. The right-hand tangent is along x radial. At geographic
    poles the saved longitude defines the limiting geographic basis.
    """
    if not np.isfinite(extent_deg) or not 0 < extent_deg <= 180:
        raise ValueError("profile extent must be in (0, 180] degrees")
    if not np.isfinite(spacing_deg) or not 0 < spacing_deg <= 180:
        raise ValueError("profile spacing must be in (0, 180] degrees")
    if longitudes is not None and (bearings is not None or centre is not None):
        raise ValueError("profile longitudes cannot be combined with bearings or a centre")
    meridians = longitudes is not None or (
        case == "cap" and np.isclose(load_centre_deg[0], 0.)
        and bearings is None and centre is None)
    if meridians:
        longitudes = np.asarray([0., 90., 180.] if longitudes is None else longitudes,
                                dtype=float).copy()
        if longitudes.ndim != 1 or not longitudes.size or not np.all(np.isfinite(longitudes)):
            raise ValueError("profile longitudes must be a nonempty list of finite angles")
        longitudes %= 360
        coordinate = np.linspace(0, extent_deg, int(np.ceil(extent_deg / spacing_deg)) + 1)
        theta, phi = np.deg2rad(coordinate)[None, :], np.deg2rad(longitudes)[:, None]
        theta, phi = np.broadcast_arrays(theta, phi)
        radial = np.stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),
                           np.cos(theta)), axis=-1)
        tangent = np.stack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),
                            -np.sin(theta)), axis=-1)
        data = {"profile_kind": "meridian", "coordinate_deg": coordinate,
                "colatitude_deg": coordinate, "longitude_deg": longitudes,
                "profile_labels": np.array([f"Longitude {x:g}°" for x in longitudes])}
    else:
        centre = np.asarray(load_centre_deg if centre is None else centre, dtype=float)
        if centre.shape != (2,) or not np.all(np.isfinite(centre)) or not 0 <= centre[0] <= 180:
            raise ValueError("profile centre must be finite (colatitude in [0, 180], longitude)")
        centre = np.array([centre[0], centre[1] % 360])
        bearings = np.asarray([0., 90.] if bearings is None else bearings, dtype=float).copy()
        if bearings.ndim != 1 or not bearings.size or not np.all(np.isfinite(bearings)):
            raise ValueError("profile bearings must be a nonempty list of finite angles")
        bearings %= 360
        tc, pc = np.deg2rad(centre)
        axis = np.array([np.sin(tc)*np.cos(pc), np.sin(tc)*np.sin(pc), np.cos(tc)])
        north = np.array([-np.cos(tc)*np.cos(pc), -np.cos(tc)*np.sin(pc), np.sin(tc)])
        east = np.array([-np.sin(pc), np.cos(pc), 0.])
        bearing = np.deg2rad(bearings)[:, None]
        direction = np.cos(bearing)*north + np.sin(bearing)*east
        intervals = int(np.ceil(extent_deg / spacing_deg))
        coordinate = np.linspace(-extent_deg, extent_deg, 2*intervals + 1)
        angle = np.deg2rad(coordinate)[None, :, None]
        radial = np.cos(angle)*axis + np.sin(angle)*direction[:, None, :]
        tangent = -np.sin(angle)*axis + np.cos(angle)*direction[:, None, :]
        theta = np.arccos(np.clip(radial[..., 2], -1, 1))
        phi = np.arctan2(radial[..., 1], radial[..., 0])
        data = {"profile_kind": "load-centred", "coordinate_deg": coordinate,
                "centre_deg": centre, "bearing_deg": bearings,
                "profile_labels": np.array([f"Bearing {x:g}°" for x in bearings])}
    south = np.stack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),
                      -np.sin(theta)), axis=-1)
    east = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    return {**data, "unit_vectors": radial, "tangent_vectors": tangent,
            "right_vectors": np.cross(tangent, radial),
            "south_vectors": south, "east_vectors": east,
            "point_colatitude_deg": np.rad2deg(theta),
            "point_longitude_deg": np.rad2deg(phi) % 360}


def profile_settings(geometry, depth_fraction):
    """Compact sampling metadata for the run summary/checkpoint."""
    coordinate = geometry["coordinate_deg"]
    return {"kind": geometry["profile_kind"],
            "centre_deg": geometry.get("centre_deg"),
            "bearing_deg": geometry.get("bearing_deg"),
            "longitude_deg": geometry.get("longitude_deg"),
            "extent_deg": float(coordinate[-1]),
            "spacing_deg": float(coordinate[1] - coordinate[0]),
            "depth_fraction": depth_fraction}
