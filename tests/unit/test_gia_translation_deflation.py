r"""The translation deflation of the retained `(u, psi)` solve of `ReducedGIABlockPC`.

`gadopt.preconditioners._TranslationDeflation` splits the retained solve
`A x = b` into a coarse part on the rigid translations and a Krylov solve of
the deflated operator `P A`, `x = Q b + (I - Q A) x~` with `P A x~ = P b`.
The identity is exact for any deflation space with an invertible coarse
matrix, so the deflated solve must return the solution of `A x = b`, and
the Krylov method must see an operator that maps the deflation space to
zero.

The checks run on a small stand-in for the retained system, assembled on
the unit square with no GIA solver and no time step: an elasticity block
with a weak boundary spring, so that the rigid translations are near-null
directions of the displacement block (as the sea-level load makes them in
the GIA block, see the class docstring), a potential block (Laplacian plus
mass), and a weak coupling between the two. The preconditioner object is a
`ReducedGIABlockPC` with these blocks put in place of the ones its
`initialize` assembles from a Jacobian, so that `_initialise_deflation` and
`apply` run as they do in the solver.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd
from firedrake.petsc import PETSc

from gadopt.preconditioners import ReducedGIABlockPC

#: The options prefix of the retained solve of the stand-in.
PREFIX = "translation_deflation_test_condensed_"

#: The stiffness of the boundary spring relative to the elastic stiffness.
#: Small, so that the translations are near-null directions of the
#: displacement block: the spring is the only term that resists them. The
#: smallest eigenvalues of the nest are then about 5e-4 against a largest of
#: 15. A much weaker spring makes the undeflated FGMRES lose orthogonality
#: and stagnate above any tight tolerance (measured at 1e-3: 2000
#: iterations without convergence), which would test the arithmetic and not
#: the deflation.
SPRING = 1e-2

#: The strength of the coupling between displacement and potential.
COUPLING = 0.1


def retained_block(deflate):
    """A `ReducedGIABlockPC` on the stand-in system, with or without deflation.

    The blocks, the nest and the inner KSP are made here in place of
    `ReducedGIABlockPC.initialize`, which reads them off a Jacobian. The
    inner KSP has the structure of the shipped one: FGMRES around a
    multiplicative fieldsplit over the nest, with a deliberately weak
    displacement split (SOR) so that the near-null translations show in
    the iteration count, and an exact potential split (LU), whose PC also
    serves the potential solves of the deflation build.

    Args:
      deflate: the value of the option `deflate_translations`.

    Returns:
      The preconditioner object, ready for `apply`.
    """
    mesh = fd.UnitSquareMesh(8, 8)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    W = V * Q
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    p, q = fd.TrialFunction(Q), fd.TestFunction(Q)

    def eps(w):
        return fd.sym(fd.grad(w))

    pc = object.__new__(ReducedGIABlockPC)
    pc.displacement_space, pc.potential_space = V, Q
    # Elasticity with a weak boundary spring: the rigid translations have the
    # energy of the spring alone, `SPRING` times the boundary length.
    pc.S_uu = fd.assemble((fd.inner(eps(u), eps(v)) + fd.div(u) * fd.div(v)) * fd.dx
                          + SPRING * fd.inner(u, v) * fd.ds, mat_type="aij")
    # A coupling through the first displacement component, in both rows, so
    # that the potential part of the deflation space is not zero.
    pc.A_upsi = fd.assemble(COUPLING * p * v[0] * fd.dx, mat_type="aij")
    pc.A_psiu = fd.assemble(COUPLING * u[0] * q * fd.dx, mat_type="aij")
    pc.A_psipsi = fd.assemble((fd.inner(fd.grad(p), fd.grad(q)) + p * q) * fd.dx,
                              mat_type="aij")
    pc.condensed_operator = PETSc.Mat().createNest(
        [[pc.S_uu.petscmat, pc.A_upsi.petscmat],
         [pc.A_psiu.petscmat, pc.A_psipsi.petscmat]], comm=mesh.comm)
    pc.condensed_operator.setUp()
    pc.projected_operator = None
    pc._rhs_blocks = (pc.S_uu.petscmat.createVecRight(),
                      pc.A_psipsi.petscmat.createVecRight())
    pc._solution_blocks = (pc.S_uu.petscmat.createVecRight(),
                           pc.A_psipsi.petscmat.createVecRight())
    pc._rhs = PETSc.Vec().createNest(list(pc._rhs_blocks), comm=mesh.comm)
    pc._solution_vec = PETSc.Vec().createNest(list(pc._solution_blocks), comm=mesh.comm)
    pc.residual = fd.Cofunction(W.dual())
    pc.solution = fd.Function(W)
    pc.assembly_count = 1

    options = PETSc.Options()
    for key, value in {
            "ksp_type": "fgmres", "ksp_rtol": 1e-10, "ksp_atol": 0.0,
            "ksp_max_it": 2000, "ksp_gmres_restart": 2000,
            "pc_type": "fieldsplit", "pc_fieldsplit_type": "multiplicative",
            "fieldsplit_0_ksp_type": "preonly", "fieldsplit_0_pc_type": "sor",
            "fieldsplit_1_ksp_type": "preonly", "fieldsplit_1_pc_type": "lu",
            "deflate_translations": deflate}.items():
        options[PREFIX + key] = value
    ksp = PETSc.KSP().create(comm=mesh.comm)
    ksp.setOptionsPrefix(PREFIX)
    ksp.setOperators(pc.condensed_operator, pc.condensed_operator)
    ksp.setFromOptions()
    # The fieldsplit reads the options of its splits when it is set up, so
    # it is set up here, while the options are in the database.
    ksp.getPC().setUp()
    pc.condensed_ksp = ksp
    # `_initialise_deflation` reads only the communicator of the PETSc PC.
    pc._initialise_deflation(SimpleNamespace(comm=mesh.comm), PREFIX)
    for key in list(options.getAll()):
        if key.startswith(PREFIX):
            options.delValue("-" + key)
    return pc


def mixed_vector(pc, seed):
    """A random right-hand side on the mixed `(u, psi)` layout, with a translation in it.

    The translation component is what makes an undeflated solve slow: the
    right-hand side then has a large part along the near-null directions.
    """
    rng = np.random.default_rng(seed)
    f = fd.Function(pc.solution.function_space())
    for sub in f.subfunctions:
        sub.dat.data[...] = rng.standard_normal(sub.dat.data.shape)
    f.subfunctions[0].dat.data[:, 0] += 10.0
    with f.dat.vec_ro as vec:
        return vec.copy()


def exact_solution(pc, b):
    """The solution of the nest system by LU of its monolithic copy."""
    A = pc.condensed_operator.convert("aij")
    ksp = PETSc.KSP().create(comm=A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    x = b.duplicate()
    ksp.solve(b, x)
    return x


def test_the_deflated_operator_maps_the_deflation_space_to_zero():
    """`P A z = 0` for every translation vector `z` of the deflation space."""
    pc = retained_block(True)
    deflation = pc.deflation
    assert len(deflation.Z) == 2
    for z, az in zip(deflation.Z, deflation.AZ):
        y = az.duplicate()
        pc.deflated_operator.mult(z, y)
        assert y.norm() <= 1e-12 * az.norm()
    # The potential parts are the solutions of the potential split.
    u, psi = deflation.Z[0].getNestSubVecs()
    assert psi.norm() > 0.0


@pytest.mark.parametrize("deflate", [True, False])
def test_the_retained_solve_returns_the_solution_of_the_nest_system(deflate):
    """`apply` returns the solution of `A x = b` with and without deflation."""
    pc = retained_block(deflate)
    b = mixed_vector(pc, 1)
    y = b.duplicate()
    pc.apply(None, b, y)
    # The mixed layout of `y` against the nest layout of the exact solution:
    # compare field by field.
    with pc.residual.dat.vec_wo as residual:
        b.copy(residual)
    for source, target in zip(pc.residual.subfunctions, pc._rhs_blocks):
        with source.dat.vec_ro as vec:
            vec.copy(target)
    exact = exact_solution(pc, pc._rhs)
    result = fd.Function(pc.solution.function_space())
    with result.dat.vec_wo as vec:
        y.copy(vec)
    for field, part in zip(result.subfunctions, exact.getNestSubVecs()):
        with field.dat.vec_ro as vec:
            difference = vec.copy()
            difference.axpy(-1.0, part)
            assert difference.norm() <= 1e-7 * part.norm()


def test_the_deflation_removes_the_slow_translations():
    """The deflated solve needs fewer inner iterations on the same system.

    The translations are near-null directions of the nest and the right-hand
    side holds a translation. The weak SOR split does not represent them, so
    the undeflated Krylov solve spends iterations on them that the deflated
    one does not need. The gain here is moderate (40 against 95 iterations
    when the test was written) because the rigid rotation of the unit square
    is a third near-null direction that the deflation leaves in place; the
    GIA solver declares the rotations as a nullspace instead.
    """
    counts = {}
    for deflate in (True, False):
        pc = retained_block(deflate)
        b = mixed_vector(pc, 2)
        pc.apply(None, b, b.duplicate())
        assert pc.condensed_ksp.getConvergedReason() > 0
        counts[deflate] = pc.condensed_ksp.getIterationNumber()
    assert counts[True] < counts[False], counts
