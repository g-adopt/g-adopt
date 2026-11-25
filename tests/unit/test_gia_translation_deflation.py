r"""The translation deflation of the retained `(u, psi)` solve of `ReducedGIABlockPC`.

`gadopt.preconditioners._TranslationDeflation` splits the retained solve
`A x = b` into a coarse part on the rigid translations and a Krylov solve of
the deflated operator `P A`, `x = Q b + (I - Q A) x~` with `P A x~ = P b`.
The identity is exact for any deflation space with an invertible coarse
matrix `E = Z^T A Z`, so the deflated solve must return the solution of
`A x = b`, and the Krylov method must see an operator that maps the
deflation space to zero.

The checks run on a small stand-in for block 0 of the GIA solver, on the
unit square, with no GIA solver, no time step and one rank. The stand-in is
a two-field `(u, psi)` system: an elasticity block with a weak boundary
spring, so that the rigid translations are near-null directions of the
displacement block (in the GIA block 0 they are near-null because only the
`Real` rows of `CentreOfMass` hold them, see the class docstring), a
potential block (Laplacian plus mass), and a coupling between the two
through the first displacement component.

The preconditioner is built through the public path: a Firedrake
`LinearVariationalSolver` with a matrix-free operator and
`pc_python_type: gadopt.ReducedGIABlockPC`, so that Firedrake calls
`initialize` and `update` as it does inside the GIA solver. The outer KSP is
`preonly`, so one solve is one application of the preconditioner. The
application context carries the two staleness markers of the GIA solver
(`operator_version`, `gia_solve_index`), which `_RebuildRule` reads to decide
on a reassembly.
"""
import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd
from firedrake.petsc import PETSc

from gadopt.preconditioners import ReducedGIABlockPC

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

#: The relative tolerance of the inner Krylov solve.
INNER_RTOL = 1e-10


class StandIn:
    """The stand-in block 0 and the Firedrake solver whose PC is `ReducedGIABlockPC`.

    Args:
      deflate: the value of the option `condensed_deflate_translations`.
      spring: the stiffness of the boundary spring.
      inner_rtol: the relative tolerance of the inner Krylov solve.
      declared: the translation axes to declare as the nullspace and the
        transpose nullspace of the displacement (`condensed_field_nullspace`
        and `condensed_field_transpose_nullspace` in the application
        context), as the GIA solver declares its rotations.

    The coefficients `spring` and `coupling` are `Constant`s, so that a test
    can change the operator between two solves.
    """

    def __init__(self, deflate=True, spring=SPRING, inner_rtol=INNER_RTOL, declared=()):
        mesh = fd.UnitSquareMesh(8, 8)
        self.V = fd.VectorFunctionSpace(mesh, "CG", 1)
        self.Q = fd.FunctionSpace(mesh, "CG", 1)
        self.W = self.V * self.Q
        u, p = fd.TrialFunctions(self.W)
        v, q = fd.TestFunctions(self.W)
        self.spring = fd.Constant(spring)
        self.coupling = fd.Constant(COUPLING)

        def eps(w):
            return fd.sym(fd.grad(w))

        # Elasticity with a weak boundary spring: the rigid translations have
        # the energy of the spring alone, `spring` times the boundary length.
        # The coupling acts through the first displacement component, in both
        # rows, so that the potential part of the deflation space is not zero.
        self.a = ((fd.inner(eps(u), eps(v)) + fd.div(u) * fd.div(v)) * fd.dx
                  + self.spring * fd.inner(u, v) * fd.ds
                  + self.coupling * (p * v[0] + u[0] * q) * fd.dx
                  + (fd.inner(fd.grad(p), fd.grad(q)) + p * q) * fd.dx)
        # A random right-hand side with a large translation in it: the
        # translation component is what makes an undeflated solve slow.
        rng = np.random.default_rng(1)
        f = fd.Function(self.V)
        f.dat.data[...] = rng.standard_normal(f.dat.data.shape)
        f.dat.data[:, 0] += 10.0
        g = fd.Function(self.Q)
        g.dat.data[...] = rng.standard_normal(g.dat.data.shape)
        self.L = fd.inner(f, v) * fd.dx + g * q * fd.dx
        self.solution = fd.Function(self.W)

        appctx = {"operator_version": 0, "gia_solve_index": 0}
        if declared:
            def provider(space, axes=tuple(declared)):
                vectors = []
                for k in axes:
                    direction = [0.0] * space.value_size
                    direction[k] = 1.0
                    vectors.append(fd.Function(space).assign(fd.Constant(direction)))
                basis = fd.VectorSpaceBasis(vectors)
                basis.orthonormalize()
                return basis
            appctx["condensed_field_nullspace"] = provider
            appctx["condensed_field_transpose_nullspace"] = provider
        # The inner solve has the structure of the shipped one: FGMRES around
        # a multiplicative fieldsplit over the nest, with a deliberately weak
        # displacement split (SOR) so that the near-null translations show in
        # the iteration count, and an exact potential split (LU), whose PC
        # also serves the potential solves of the deflation build.
        parameters = {
            "mat_type": "matfree", "ksp_type": "preonly",
            "pc_type": "python", "pc_python_type": "gadopt.ReducedGIABlockPC",
            "condensed_ksp_type": "fgmres", "condensed_ksp_rtol": inner_rtol,
            "condensed_ksp_atol": 0.0, "condensed_ksp_max_it": 2000,
            "condensed_ksp_gmres_restart": 2000,
            "condensed_pc_type": "fieldsplit",
            "condensed_pc_fieldsplit_type": "multiplicative",
            "condensed_fieldsplit_0_ksp_type": "preonly",
            "condensed_fieldsplit_0_pc_type": "sor",
            "condensed_fieldsplit_1_ksp_type": "preonly",
            "condensed_fieldsplit_1_pc_type": "lu",
            "condensed_deflate_translations": deflate}
        self.solver = fd.LinearVariationalSolver(
            fd.LinearVariationalProblem(self.a, self.L, self.solution),
            solver_parameters=parameters, appctx=appctx,
            options_prefix="translation_deflation_test")

    @property
    def appctx(self):
        """The application context the preconditioner reads."""
        return self.solver._ctx.appctx

    def solve(self):
        """One solve, that is one application of `ReducedGIABlockPC`."""
        self.solution.zero()
        self.solver.solve()

    @property
    def pc(self):
        """The `ReducedGIABlockPC` instance, after the first solve."""
        context = self.solver.snes.ksp.pc.getPythonContext()
        assert isinstance(context, ReducedGIABlockPC)
        return context

    def matrix(self):
        """The monolithic matrix of the present operator, assembled from the form."""
        return fd.assemble(self.a, mat_type="aij").petscmat

    def exact(self):
        """The solution of the present system by LU of its monolithic matrix."""
        b = fd.assemble(self.L)
        x = fd.Function(self.W)
        with b.dat.vec_ro as rhs, x.dat.vec_wo as out:
            ksp = PETSc.KSP().create(comm=rhs.getComm())
            ksp.setOperators(self.matrix())
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.solve(rhs, out)
        return x

    def mixed(self, nest):
        """A copy of a nest vector of the retained solve on the mixed layout."""
        f = fd.Function(self.W)
        for part, field in zip(nest.getNestSubVecs(), f.subfunctions):
            with field.dat.vec_wo as vec:
                part.copy(vec)
        return f


def operator_product(matrix, f):
    """`matrix f` for a mixed Function `f`, as a mixed Function."""
    out = fd.Function(f.function_space())
    with f.dat.vec_ro as x, out.dat.vec_wo as y:
        matrix.mult(x, y)
    return out


def norm(f):
    """The 2-norm of a Function or a field of it."""
    with f.dat.vec_ro as vec:
        return vec.norm()


def assert_solution(case, rtol):
    """The solution of the solve agrees with the LU solution, field by field.

    Args:
      case: the `StandIn`, after a solve.
      rtol: the largest relative 2-norm difference in each field.
    """
    exact = case.exact()
    for field, reference in zip(case.solution.subfunctions, exact.subfunctions):
        with field.dat.vec_ro as x, reference.dat.vec_ro as y:
            difference = x.copy()
            difference.axpy(-1.0, y)
            assert difference.norm() <= rtol * y.norm()


def check_deflation_space(case, matrix):
    """Check `Z`, `A Z` and `E` of the deflation against `matrix`.

    `A Z` must be the product of the present operator with `Z`, `E` must be
    `Z^T A Z`, and the potential part of each `z_k` must solve the potential
    rows, `A_psiu e_k + A_psipsi psi_k = 0`, so the potential rows of
    `A z_k` vanish to the tolerance of the potential solve (1e-8).

    Args:
      case: the `StandIn`.
      matrix: the monolithic matrix of the operator the deflation must match.
    """
    deflation = case.pc.deflation
    products = []
    for z, az in zip(deflation.Z, deflation.AZ):
        z_mixed = case.mixed(z)
        expected = operator_product(matrix, z_mixed)
        products.append((z_mixed, expected))
        stored = case.mixed(az)
        for a, b in zip(stored.subfunctions, expected.subfunctions):
            with a.dat.vec_ro as x, b.dat.vec_ro as y:
                difference = x.copy()
                difference.axpy(-1.0, y)
                assert difference.norm() <= 1e-12 * max(norm(expected), 1e-300)
        # The potential rows of `A z_k`, against the size of the coupling
        # term `A_psiu e_k` alone.
        coupling_only = fd.Function(case.W)
        coupling_only.subfunctions[0].assign(z_mixed.subfunctions[0])
        coupling = operator_product(matrix, coupling_only).subfunctions[1]
        assert norm(expected.subfunctions[1]) <= 1e-7 * norm(coupling)
    for i, (z_i, _) in enumerate(products):
        for j, (_, az_j) in enumerate(products):
            with z_i.dat.vec_ro as x, az_j.dat.vec_ro as y:
                assert abs(deflation.E[i, j] - x.dot(y)) <= 1e-12 * np.abs(deflation.E).max()


def test_the_deflated_operator_maps_the_deflation_space_to_zero():
    """`P A z = 0` for every translation vector `z` of the deflation space."""
    case = StandIn()
    case.solve()
    deflation = case.pc.deflation
    assert len(deflation.Z) == 2
    for z, az in zip(deflation.Z, deflation.AZ):
        y = az.duplicate()
        case.pc.deflated_operator.mult(z, y)
        assert y.norm() <= 1e-12 * az.norm()
    # The potential parts are the solutions of the potential split.
    u, psi = deflation.Z[0].getNestSubVecs()
    assert psi.norm() > 0.0
    check_deflation_space(case, case.matrix())


@pytest.mark.parametrize("deflate", [True, False])
def test_the_retained_solve_returns_the_solution_of_the_nest_system(deflate):
    """`apply` returns the solution of `A x = b` with and without deflation."""
    case = StandIn(deflate)
    case.solve()
    assert (case.pc.deflation is not None) == deflate
    assert_solution(case, 1e-7)


def test_the_inner_tolerance_is_relative_to_the_undeflated_right_hand_side():
    """The returned solution meets `rtol |b|` against the undeflated system.

    The Krylov solve works on `P b`, whose norm differs from `|b|`. Its
    `rtol` is scaled by `|b| / |P b|`, so the true residual of the returned
    solution is below `rtol |b|`, as without the deflation. At a loose
    `rtol` the check sees a missing scaling whenever `|P b| > |b|`, and it
    sees a wrong scaling in either direction through the KSP tolerance.
    """
    rtol = 1e-4
    case = StandIn(inner_rtol=rtol)
    case.solve()
    pc = case.pc
    b = fd.assemble(case.L)
    residual = operator_product(case.matrix(), case.solution)
    with b.dat.vec_ro as rhs, residual.dat.vec_rw as r:
        r.aypx(-1.0, rhs)
        assert r.norm() <= 1.0001 * rtol * rhs.norm()
    # The KSP ran with the scaled tolerance, `rtol |b| / |P b|`.
    projected = pc._rhs.copy()
    pc.deflation.project(projected)
    scaled = pc.condensed_ksp.getTolerances()[0]
    assert scaled == pytest.approx(rtol * pc._rhs.norm() / projected.norm(), rel=1e-12)


def test_the_deflation_removes_the_slow_translations():
    """The deflated solve needs fewer inner iterations on the same system.

    The translations are near-null directions of the nest and the right-hand
    side holds a translation. The weak SOR split does not represent them, so
    the undeflated Krylov solve spends iterations on them that the deflated
    one does not need. The gain here is moderate (about 40 against 95
    iterations) because the rigid rotation of the unit square is a third
    near-null direction that the deflation leaves in place; the GIA solver
    declares the rotations as a nullspace instead.
    """
    counts = {}
    for deflate in (True, False):
        case = StandIn(deflate)
        case.solve()
        assert case.pc.condensed_ksp.getConvergedReason() > 0
        counts[deflate] = case.pc.condensed_ksp.getIterationNumber()
    assert counts[True] < counts[False], counts


def test_update_rebuilds_the_deflation_after_an_operator_change():
    """`update`: a reassembly rebuilds `Z`; otherwise `A Z` and `E` are recomputed.

    The coupling and the spring change between two solves and the operator
    version moves, so the rebuild rule reassembles the blocks. The potential
    parts of `Z` must then solve the potential rows of the new operator, and
    `A Z` and `E` must belong to it; the deflation space of the old operator
    would leave a large residual in those rows. A third solve with no new
    version keeps the blocks and the space, and `A Z` and `E` stay those of
    the assembled operator. The KSP must keep the deflated operator through
    both updates, although `CondensedBlockPC.update` hands it the undeflated
    one after a reassembly. Each solve must return the solution of its
    system.
    """
    case = StandIn()
    case.solve()
    pc = case.pc
    assert (pc.assembly_count, pc.deflation.build_count) == (1, 1)
    old_E = pc.deflation.E.copy()

    case.coupling.assign(3 * COUPLING)
    case.spring.assign(2 * SPRING)
    case.appctx["operator_version"] = 1
    case.solve()
    assert (pc.assembly_count, pc.deflation.build_count) == (2, 2)
    assert pc.condensed_ksp.getOperators()[0].handle == pc.deflated_operator.handle
    assert not np.allclose(pc.deflation.E, old_E)
    check_deflation_space(case, case.matrix())
    assert_solution(case, 1e-7)

    # The same operator version: no reassembly and no new space, and the
    # deflation still matches the assembled operator. The coefficient change
    # below is seen by the matrix-free outer operator only; the rebuild rule
    # keeps the assembled blocks until the version moves.
    assembled = case.matrix()
    case.coupling.assign(5 * COUPLING)
    case.solve()
    assert (pc.assembly_count, pc.deflation.build_count) == (2, 2)
    assert pc.condensed_ksp.getOperators()[0].handle == pc.deflated_operator.handle
    check_deflation_space(case, assembled)


def test_a_translation_in_the_declared_nullspace_is_not_deflated():
    """A declared translation is left out; with both declared, no deflation.

    A declared translation is removed by the projection of the retained
    solve already, and its coarse entry would hold only the residual of the
    potential solve. The deflation must leave it out and keep the other one.
    """
    case = StandIn(declared=(0,))
    case.solve()
    deflation = case.pc.deflation
    assert len(deflation.Z) == 1
    u, _ = deflation.Z[0].getNestSubVecs()
    values = u.getArray().reshape(-1, 2)
    assert np.all(values[:, 0] == 0.0) and np.all(values[:, 1] == 1.0)

    case = StandIn(declared=(0, 1))
    case.solve()
    assert case.pc.deflation is None


def test_a_singular_coarse_matrix_is_refused_with_the_switch_named():
    """An exactly free translation gives a singular `E`, refused with a message.

    Without the spring, the `y` translation costs no energy at all (no
    coupling either), so its row and column of `E` hold round-off only. The
    error must name the option that switches the deflation off. Firedrake
    raises a `PETSc.Error` from the solve, with the Python exception of the
    preconditioner as its cause.
    """
    case = StandIn(spring=0.0)
    # PETSc prints its error stack through the default handler. Under the
    # output capture of pytest that print crashes the process (measured: a
    # segmentation fault without `-s`, a pass with it), so the test pushes
    # PETSc's "return" handler, which prints nothing, around the failing
    # solve. The exception and its cause are the same with either handler.
    PETSc.Sys.pushErrorHandler("return")
    try:
        with pytest.raises(PETSc.Error) as raised:
            case.solve()
    finally:
        PETSc.Sys.popErrorHandler()
    cause = raised.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert "singular to working precision" in str(cause)
    assert "translation_deflation_test_condensed_deflate_translations" in str(cause)


def test_a_near_singular_coarse_matrix_is_accepted():
    """A very small but resolved `E` is the purpose of the deflation.

    With the spring at 1e-9 of the elastic stiffness, the `y` translation is
    nearly free, and `E` is about 1e-8 in that direction. The check must
    accept it, and the deflated solve must still return the solution of the
    system: the deflation removes the direction that an undeflated solve
    cannot resolve.
    """
    case = StandIn(spring=1e-9)
    case.solve()
    deflation = case.pc.deflation
    assert deflation.scaled_smallest_singular_value < 1e-10
    assert_solution(case, 1e-5)
