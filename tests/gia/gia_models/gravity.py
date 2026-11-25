"""Helpers of `tests/unit/test_gravity_solver.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gravity_solver.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
import numpy as np
from gadopt import CylindricalDtN, GravitySolver
from gadopt.gravity_solver import (
    direct_gravity_solver_parameters,
    iterative_gravity_solver_parameters,
)


RMIN, RMAX = 1.22, 2.22
R1_SHELL, R2_SHELL = 1.9, 2.0


def radial_layers(dr, *interfaces):
    """Radial node ladder from RMIN to RMAX conforming to given interfaces."""
    nodes = [RMIN]
    for a, b in zip((RMIN,) + interfaces, interfaces + (RMAX,)):
        nodes += list(np.linspace(a, b, max(1, round((b - a) / dr)) + 1))[1:]
    return list(np.diff(np.array(nodes)))


def annulus_mesh(n_azimuthal=192, dr=0.1, interfaces=()):
    heights = radial_layers(dr, *interfaces)
    base = fd.CircleManifoldMesh(n_azimuthal, radius=RMIN, degree=2)
    return fd.ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                           extrusion_type="radial")


def shell_mesh_3d(refinement_level=2, dr=0.15, interfaces=()):
    heights = radial_layers(dr, *interfaces)
    base = fd.CubedSphereMesh(radius=RMIN, refinement_level=refinement_level,
                              degree=2)
    return fd.ExtrudedMesh(base, layers=len(heights), layer_height=heights,
                           extrusion_type="radial")


def relative_l2_error(psi, reference, mesh, degree=6):
    dxq = fd.dx(domain=mesh, degree=degree)
    error = fd.assemble((psi - reference) ** 2 * dxq)
    norm = fd.assemble(reference**2 * dxq)
    return np.sqrt(error / norm)


def field_number_parameters(preset, i_R, n_multipliers):
    """The pre-index-set solver options: a Schur split described by field number.

    This is the configuration the shipped preset used before the two blocks
    were described by index sets, kept here as the control arm of the parity
    tests below. It reaches PETSc only through a full `solver_parameters`
    dictionary, which the class passes through verbatim - so this helper also
    pins that pass-through, without which the old split could no longer be
    driven at all and the parity comparison would have nothing to compare
    against. It carries PETSc's 128-field enumeration limit, so it cannot set
    up once the mixed space exceeds 128 sub-fields.
    """
    if preset == "direct":
        fieldsplit_0 = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": dict(direct_gravity_solver_parameters),
        }
    else:
        fieldsplit_0 = dict(iterative_gravity_solver_parameters)
    return {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-11,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": ",".join(map(str, range(i_R))),
        "pc_fieldsplit_1_fields": ",".join(
            map(str, range(i_R, i_R + n_multipliers))),
        "fieldsplit_0": fieldsplit_0,
        "fieldsplit_1": {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "pc_type": "none",
        },
    }


def annulus_source(mesh, m_mode=3):
    """Azimuthally pure density: zero net mass, as the 2-D exterior map needs."""
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    phi = fd.atan2(X[1], X[0])
    return fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(
        fd.cos(m_mode * phi) * fd.exp(-(((r - 1.7) / 0.2) ** 2)))


def solve_annulus(mesh, rho, M, solver_parameters):
    """Solves the two-boundary DtN annulus problem and returns (psi, solver)."""
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    solver = GravitySolver(
        psi, rho,
        bcs={"top": {"dtn": CylindricalDtN(M=M)},
             "bottom": {"dtn": CylindricalDtN(M=M)}},
        solver_parameters=solver_parameters)
    solver.solve()
    return psi, solver


def max_coefficient_difference(first, second):
    """Largest difference over the trace coefficients two solves share."""
    return max(abs(first[bc_id][key] - second[bc_id][key])
               for bc_id in first for key in first[bc_id])
