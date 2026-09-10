"""Fingerprint of the pointwise viscoelastic weak-"un" Jacobian.

The properties of the weak boundary terms, symmetry and the variational
structure, are tested in `tests/weak_bc_gia`, where each case runs as its own
doit step. This file keeps the one check that pins entries instead of
properties, and that needs no solve: a stored reference matrix.

It is self-contained, so that the entrypoints in `tests/weak_bc_gia` and the
tests here share no module.
"""

from pathlib import Path

import firedrake as fd
import gadopt
import numpy as np

# Directory holding the stored Jacobian fingerprint.
DATA_DIR = Path(__file__).parent.resolve() / "data"

# Elastic shear modulus and viscosity of the single Maxwell element, chosen so
# the Maxwell time tau = viscosity / shear_modulus is 1 and dt / tau is simply
# dt. The bulk modulus and the bulk-to-shear ratio give the stress a volumetric
# part, which the weak boundary penalty has to pick up.
SHEAR_MODULUS = 2.0
VISCOSITY = 2.0
GIA_DT = 0.25
GIA_BULK_MODULUS = 3.0
GIA_BULK_SHEAR_RATIO = 1.5
# Boundary data for the weak "un" condition. A nonzero value keeps the normal
# jump away from zero, so a wrong coefficient in any term proportional to it
# changes the matrix.
WEAK_UN_VALUE = 0.3


def generic_velocity(mesh):
    """A generic displacement field.

    Unlike u = X (identity), this has n.u != 0 on every boundary and an
    anisotropic strain, so a symmetric-but-wrong penalty coefficient in the weak
    boundary terms changes the assembled matrix. At u = X the normal jump is
    zero on every boundary face of this mesh, which would hide such an error.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    return X + fd.Constant([float(i + 1) for i in range(dim)]) + 0.3 * X[0] * X


def maxwell_approximation(mesh):
    """A single-element compressible Maxwell approximation on `mesh`.

    The density is a DG0 field so that the buoyancy term, which differentiates
    it, is well defined.
    """
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    return gadopt.MaxwellApproximation(
        GIA_BULK_MODULUS,
        fd.Function(DG0).assign(1),
        SHEAR_MODULUS,
        VISCOSITY,
        bulk_shear_ratio=GIA_BULK_SHEAR_RATIO,
        exponent=1,
        transition_stress=5.0,
        B_mu=1.27,
    )


def history_state(mesh, space, factor=0.1):
    """A nonzero, anisotropic internal-variable field.

    A zero history would make every term proportional to the internal variables
    vanish, hiding a wrong coefficient in front of them.
    """
    X = fd.SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension
    return fd.Function(space).interpolate(
        factor * fd.sym(fd.outer(X, fd.as_vector([float(i + 1) for i in range(dim)])))
    )


def fingerprint_jacobian():
    """Assemble the fixed pointwise weak-"un" Jacobian stored under `data/`.

    Kept as a module-level function so that the stored matrix and the matrix the
    test compares against come from the same definition.
    """
    mesh = fd.UnitSquareMesh(3, 2)
    mesh.cartesian = True
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)

    u = fd.Function(V).interpolate(generic_velocity(mesh))
    m = history_state(mesh, S)
    approximation = maxwell_approximation(mesh)
    solver = gadopt.InternalVariableSolver(
        u, approximation, dt=GIA_DT, internal_variables=m,
        bcs={1: {"un": WEAK_UN_VALUE}, 2: {"un": 0.0}},
        solver_parameters="direct",
    )
    jacobian = fd.assemble(fd.derivative(solver.F, u), mat_type="aij")
    return jacobian.petscmat.convert("dense").getDenseArray().copy()


def test_pointwise_maxwell_jacobian_fingerprint():
    """The pointwise weak-"un" Jacobian must match a stored reference matrix.

    `data/gia_pointwise_maxwell_weak_un_jacobian.npy` is the reference matrix
    for the pointwise weak-"un" Jacobian on this mesh. It pins entries, not
    properties, so it catches a change that no symmetry or variational-structure
    test can see. Regenerate it with `fingerprint_jacobian()` only for a
    deliberate change to the pointwise boundary terms.

    The reference is a dense matrix in the degree-of-freedom ordering Firedrake
    produces for this mesh and element pair, so a change of that ordering also
    breaks this test. That is a deliberate trade: the test is meant to be
    sensitive.
    """
    stored = np.load(DATA_DIR / "gia_pointwise_maxwell_weak_un_jacobian.npy")
    computed = fingerprint_jacobian()
    assert computed.shape == stored.shape
    assert np.abs(computed - stored).max() <= 1e-14 * np.abs(stored).max()
