"""Helpers of `tests/unit/test_gravity_quad_degree.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gravity_quad_degree.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
import numpy as np
from gadopt import GravitySolver


RMIN, RMAX = 1.22, 2.22


def _extrude(base, dr=0.25):
    heights = list(np.diff(np.linspace(RMIN, RMAX, round(1 / dr) + 1)))
    return fd.ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                           extrusion_type="radial")


def annulus(n_azimuthal=96, irregular=False, seed=0):
    """Extruded annulus; `irregular` gives it structureless facet spacing.

    The mesh stays an exact circle either way - only the spacing changes. This
    matters because a composite Gauss rule on a *uniform* boundary annihilates
    every harmonic but multiples of the facet count, so a uniform annulus is a
    far easier test than it looks and passing on it proves little.
    """
    base = fd.CircleManifoldMesh(n_azimuthal, radius=RMIN, degree=2)
    if irregular:
        xy = np.asarray(base.coordinates.dat.data_ro, dtype=float)
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        rng = np.random.default_rng(seed)
        grid = np.linspace(-np.pi, np.pi, n_azimuthal + 1)
        steps = rng.uniform(0.4, 1.6, n_azimuthal)
        target = np.concatenate(
            [[-np.pi], -np.pi + 2 * np.pi * np.cumsum(steps) / steps.sum()])
        theta = np.interp(theta, grid, target)
        base.coordinates.dat.data[:, 0] = RMIN * np.cos(theta)
        base.coordinates.dat.data[:, 1] = RMIN * np.sin(theta)
    return _extrude(base)


def shell(refinement_level=2, warp=0.0):
    """Extruded cubed sphere; `warp` is a smooth diffeomorphism of the sphere.

    Applied to every coordinate dof including the Q2 mid-side ones, so the
    curved facets stay valid. A per-node random shove does not: it folds the
    mid-side nodes, and the symptom is a self-convergence that floors out flat
    in the degree rather than converging.
    """
    base = fd.CubedSphereMesh(radius=RMIN, refinement_level=refinement_level,
                              degree=2)
    if warp:
        xyz = np.asarray(base.coordinates.dat.data_ro, dtype=float)
        r = np.linalg.norm(xyz, axis=1)
        theta = np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0)) + \
            warp * np.sin(2 * np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0)))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi = phi + warp * np.sin(3 * phi)
        base.coordinates.dat.data[:, 0] = r * np.sin(theta) * np.cos(phi)
        base.coordinates.dat.data[:, 1] = r * np.sin(theta) * np.sin(phi)
        base.coordinates.dat.data[:, 2] = r * np.cos(theta)
    return _extrude(base)


def icosahedral_shell(refinement_level=2):
    """Extruded icosahedral sphere: triangular boundary facets."""
    return _extrude(fd.IcosahedralSphereMesh(
        radius=RMIN, refinement_level=refinement_level, degree=2))


def build(mesh, descriptor, element_degree=1, **kwargs):
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", element_degree))
    parameters = "iterative" if mesh.geometric_dimension == 3 else "direct"
    return GravitySolver(psi, 0.0, bcs={"top": {"dtn": descriptor}},
                         solver_parameters=parameters, **kwargs)


def incumbent(descriptor, element_degree=1):
    """The default this change replaces."""
    return 2 * (descriptor.max_degree + element_degree)
