"""Static condensation of the internal variables inside block 0 of the
self-gravity preconditioner.

On the uncondensed layout the mixed space is `(u, M, psi, Real...)`, with one
combined internal-variable field `M` of shape `(n, d, d)`. The iterative
preset of `SelfGravitatingGIASolver` hands the pair `(u, M)` to
`gadopt.InternalVariableSCPC` as split 0 of the block-0 sweep, so the
displacement operator the Krylov solver sees is the exact condensed matrix
and the internal variables never appear as a split of their own.

These tests check that the nesting reaches the condensation class, that the
solve it produces is the direct solve's answer, that the condensed operator is
reused across fixed-`dt` steps and rebuilt when `dt` changes, that the Krylov
method on it is CG where the operator is symmetric, and that the condensed
layout accepts every history storage `history_slices` knows.

The 2-D cases reuse the annulus of `test_gia_gravity.py`. The 3-D case builds
the level-1 cubed sphere of `TestBlockOneInThreeDimensions` uncondensed and
runs one solve on its 24 cells. `dt` is a `Constant` throughout,
because the operator-reuse fingerprint reads its value and a plain number
cannot change.
"""

import firedrake as fd
import numpy as np
import pytest

from gadopt import CoupledInternalVariableSolver
from gadopt.gia_gravity import (
    SelfGravitatingGIASolver,
    selfgrav_dtn_iterative_solver_parameters,
    self_gravitating_gia_space,
)
from gadopt.internal_variable_equation import history_slices
from gadopt.preconditioners import InternalVariableSCPC

from test_gia_gravity import (  # noqa: E402  (module-level fixtures and helpers)
    LAMBDA,
    SIGMA_HAT,
    approximation,
    gravity_bcs,
    mechanics_bcs,
    meshes,  # noqa: F401  (pytest fixture)
)


def build(meshes, *, condensed=False, n_internal_variables=1,
          approximation_kwargs=None, internal_variables=None,
          dt=1.0, **kwargs):
    """The annulus solver of `test_gia_gravity.build`, with a `Constant` dt.

    `test_gia_gravity.build` hard-codes `dt=1.0` as a float, which the
    operator-reuse fingerprint ignores by design; the reuse tests here need a
    `Constant` they can assign to.
    """
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
        n_internal_variables=n_internal_variables,
        condense_internal_variables=condensed, self_gravity_number=LAMBDA)
    z = fd.Function(Z)
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    C = fd.assemble(fd.dot(Xm, Xm) * dx_m)
    solver = SelfGravitatingGIASolver(
        z, approximation(**(approximation_kwargs or {})), layout=layout,
        dt=fd.Constant(dt), bcs=mechanics_bcs(sub), rotation_moments={"C": C},
        internal_variables=internal_variables, **kwargs)
    return solver, z, layout


def nested_preset(**kwargs):
    """The uncondensed iterative preset, tightened for a direct comparison."""
    settings = dict(condensed=False, snes_type="ksponly",
                    outer_rtol=1e-10, block0_rtol=1e-4, block0_max_it=200)
    settings.update(kwargs)
    return selfgrav_dtn_iterative_solver_parameters(**settings)


def condensation_context(solver) -> InternalVariableSCPC:
    """Reach the nested condensation class through the PETSc objects.

    Outer PC (`DtNTwoBlockSchurPC`) -> its Schur fieldsplit -> block-0 KSP
    -> its multiplicative fieldsplit -> split 0 -> python context.
    """
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    block0_ksp, _ = outer.pc.getFieldSplitSchurGetSubKSP()
    split0_ksp = block0_ksp.getPC().getFieldSplitSubKSP()[0]
    context = split0_ksp.getPC().getPythonContext()
    assert isinstance(context, InternalVariableSCPC)
    return context


def relative_difference(a, b):
    return fd.norm(a - b) / fd.norm(b)


def symmetry_defect(petsc_matrix):
    """`|S - S^T| / |S|` in the Frobenius norm of the assembled operator."""
    transpose = petsc_matrix.transpose()
    difference = petsc_matrix.copy()
    difference.axpy(-1.0, transpose)
    return difference.norm() / petsc_matrix.norm()


class TestNestedSolve:
    """The nested preset solves the annulus to the direct preset's answer."""

    @pytest.fixture(scope="class")
    def direct(self, meshes):
        solver, z, layout = build(meshes, solver_parameters="direct",
                                  solver_parameters_extra={"snes_type": "ksponly"})
        solver.solve()
        return z.copy(deepcopy=True), layout

    @pytest.fixture(scope="class")
    def nested(self, meshes):
        solver, z, layout = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        return solver, z, layout

    def test_the_condensation_class_sits_in_split_zero(self, nested):
        solver, _, _ = nested
        context = condensation_context(solver)
        W = context.cxt.a.arguments()[0].function_space()
        assert len(W) == 2
        assert W[1].value_shape == (1, 2, 2)
        assert context.cxt.appctx is solver.appctx

    def test_every_krylov_level_converged(self, nested):
        solver, _, _ = nested
        ksp = solver.solver.snes.ksp
        assert ksp.getConvergedReason() > 0
        block0_ksp, _ = ksp.pc.getPythonContext().pc.getFieldSplitSchurGetSubKSP()
        assert block0_ksp.getConvergedReason() > 0
        assert condensation_context(solver).condensed_ksp.getConvergedReason() > 0

    def test_it_matches_the_direct_solve(self, nested, direct):
        _, z, layout = nested
        z_direct, _ = direct
        u, u_direct = (f.subfunctions[layout.displacement] for f in (z, z_direct))
        psi, psi_direct = (f.subfunctions[layout.potential] for f in (z, z_direct))
        assert fd.norm(u_direct) > 0.0
        assert relative_difference(u, u_direct) < 1e-8
        assert relative_difference(psi, psi_direct) < 1e-8
        m, = history_slices(z.subfunctions[layout.internal_variable_field])
        m_direct, = history_slices(
            z_direct.subfunctions[layout.internal_variable_field])
        assert relative_difference(m, m_direct) < 1e-8


class TestOperatorReuse:
    """The condensed operator is built once per `dt`, not once per solve."""

    def test_assembly_count_follows_dt(self, meshes):
        solver, _, _ = build(meshes, solver_parameters=nested_preset())
        solver.solve()
        context = condensation_context(solver)
        assert context.assembly_count == 1
        assert solver.appctx["operator_version"] == 0
        solver.solve()
        assert context.assembly_count == 1
        assert solver.appctx["operator_version"] == 0
        solver.dt.assign(0.5)
        solver.solve()
        assert context.assembly_count == 2
        assert solver.appctx["operator_version"] == 1
        solver.solve()
        assert context.assembly_count == 2


class TestKrylovSelection:
    """CG where the condensed operator is symmetric, GMRES where it is not."""

    def test_newtonian_two_elements_is_symmetric_and_gets_cg(self, meshes):
        solver, _, _ = build(
            meshes, n_internal_variables=2,
            approximation_kwargs={"viscosity": [1.0, 3.0],
                                  "shear_modulus": [1.0, 0.5]},
            solver_parameters=nested_preset())
        solver.solve()
        context = condensation_context(solver)
        assert context.condensed_ksp.getType() == "cg"
        assert symmetry_defect(context.S.petscmat) < 1e-12

    def test_power_law_gets_gmres(self, meshes):
        solver, _, _ = build(
            meshes, approximation_kwargs={"exponent": 3.0,
                                          "transition_stress": 1.0},
            solver_parameters=nested_preset(snes_type="newtonls"))
        key = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_ksp_type"
        assert solver.solver_parameters[key] == "gmres"
        assert not solver.condensed_operator_symmetric()

    def test_a_caller_named_krylov_type_wins(self, meshes):
        key = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_ksp_type"
        solver, _, _ = build(
            meshes, approximation_kwargs={"exponent": 3.0,
                                          "transition_stress": 1.0},
            solver_parameters=nested_preset(snes_type="newtonls"),
            solver_parameters_extra={key: "fgmres"})
        assert solver.solver_parameters[key] == "fgmres"


class TestThreeDimensions:
    """The production path: 3-D, weak `un` at the CMB, Newtonian, CG."""

    @staticmethod
    def build(L=1, refinement_level=1, n_internal_variables=1):
        from gadopt import SphericalDtN

        base = fd.CubedSphereMesh(radius=1.0,
                                  refinement_level=refinement_level, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False
        Z, layout = self_gravitating_gia_space(
            mesh, mesh,
            gravity_bcs={"top": {"dtn": SphericalDtN(L)},
                         "bottom": {"dtn": SphericalDtN(L)}},
            rotation=True, condense_internal_variables=False,
            n_internal_variables=n_internal_variables,
            self_gravity_number=LAMBDA)
        z = fd.Function(Z)
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        solver = SelfGravitatingGIASolver(
            z, approximation(), layout=layout, dt=fd.Constant(1.0),
            bcs={"bottom": {"un": 0.0},
                 "top": {"normal_stress": SIGMA_HAT * X[2]}},
            rotation_moments={"C": C, "C_minus_A": 0.1 * C},
            solver_parameters=nested_preset())
        return solver, z, layout

    def test_condensed_operator_is_symmetric_and_solved_by_cg(self):
        """One solve on 24 cells, then read the operator the Krylov solver saw.

        A bare `pc.setUp()` outside a solve has no solver context on the DM
        (PETSc error 101 from `DMCreateSubDM`), and the setup-only recipe
        needs Firedrake's private solver context, so the cheapest reliable
        route to the assembled condensed operator is one solve.
        """
        solver, _, layout = self.build()
        assert layout.internal_variable_field == 1
        solver.solve()
        assert solver.solver.snes.ksp.getConvergedReason() > 0
        context = condensation_context(solver)
        assert context.condensed_ksp.getType() == "cg"
        W = context.cxt.a.arguments()[0].function_space()
        assert W[1].value_shape == (1, 3, 3)
        assert symmetry_defect(context.S.petscmat) < 1e-12


class TestCondensedLayoutStorage:
    """The condensed layout accepts every storage `history_slices` knows.

    The B5 restart drivers keep a list of `(d, d)` fields; the default is one
    combined `(n, d, d)` field. Both must give the same displacement.
    """

    def test_list_and_combined_storage_agree(self, meshes):
        _, sub = meshes
        combined, z_c, layout = build(
            meshes, condensed=True, solver_parameters="direct",
            solver_parameters_extra={"snes_type": "ksponly"})
        assert len(history_slices(combined.internal_variables)) == 1
        listed, z_l, _ = build(
            meshes, condensed=True, solver_parameters="direct",
            solver_parameters_extra={"snes_type": "ksponly"},
            internal_variables=[fd.Function(fd.TensorFunctionSpace(sub, "DG", 1))])
        assert len(history_slices(listed.internal_variables)) == 1
        for solver in (combined, listed):
            solver.solve()
            solver.solve()
        u_c = z_c.subfunctions[layout.displacement]
        u_l = z_l.subfunctions[layout.displacement]
        assert fd.norm(u_c) > 0.0
        # After the first step the two agree to 1e-17 in `u` and 1e-15 in
        # the history (measured), the residual being the same to the bit.
        # The second step feeds that 1e-15 through the direct solve of the
        # mixed system with its `Real` rows, and the measured gap is 4e-11:
        # solver roundoff, not a storage effect.
        assert relative_difference(u_c, u_l) < 1e-9
        m_c, = history_slices(combined.internal_variables)
        m_l, = history_slices(listed.internal_variables)
        assert fd.norm(m_l) > 0.0
        assert relative_difference(m_c, m_l) < 1e-9


class TestLayoutsAgree:
    """The two layouts are two consistent Nitsche discretisations at the CMB.

    They differ by a boundary discretisation error that shrinks with
    refinement (`tests/weak_bc_gia` measures order 2 on a unit square); on this
    one annulus resolution the gap is asserted at a single tolerance and no
    rate is claimed.
    """

    def test_uncondensed_and_condensed_agree_in_displacement(self, meshes):
        common = dict(n_internal_variables=2,
                      approximation_kwargs={"viscosity": [1.0, 3.0],
                                            "shear_modulus": [1.0, 0.5]},
                      solver_parameters="direct",
                      solver_parameters_extra={"snes_type": "ksponly"})
        uncondensed, z_u, layout = build(meshes, condensed=False, **common)
        condensed, z_c, _ = build(meshes, condensed=True, **common)
        uncondensed.solve()
        condensed.solve()
        u_u = z_u.subfunctions[layout.displacement]
        u_c = z_c.subfunctions[0]
        assert fd.norm(u_u) > 0.0
        assert relative_difference(u_u, u_c) < 1e-3
