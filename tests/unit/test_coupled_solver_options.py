"""How the coupled GIA solver picks its outer method, and how its blocks are preconditioned.

Two defects are pinned here.

**The tolerance inversion (E1).** `coupled_gia_solver_parameters` used to set
`snes_rtol = 1e-4` against `ksp_rtol = 1e-3`, while its own docstring claimed the
`snes_rtol` was the looser of the two. A Newton step cannot reduce the residual by more
than its linear solve does, so SNES refused every step it was given and took another
one. Measured on `tests/3d_weerdesteijn_coupled` at `--dx 250 --nz 6`, the second
timestep cost three Newton steps where two suffice, and each extra step is a Jacobian
assembly plus a full GAMG setup.

**The linearity test.** The base class asks `depends_on(approximation.mu, solution)`,
but `mu` holds a fixed field, so that reads False even for a power-law rheology, whose
nonlinearity enters through `power_law_factor(dev_stress)` instead. The coupled solver
used to answer by forcing `newtonls` unconditionally. `approximation.exponent` is the
discriminator that actually separates the two cases, and getting it wrong is expensive
in both directions: `newtonls` on a linear residual buys a wasted solve, and `ksponly`
on a power-law residual silently truncates the answer -- 0.61% in the surface
displacement on `3d_weerdesteijn_coupled`, whose exponent field is `[1, 3, 3, 1]`.

**The `AssembledPC` factoring.** `SPDAssembledPC`, `RigidBodyAssembledPC` and
`NearlyIncompressibleAssembledPC` each carried one fixed combination, so `pc_python_type`
selected exactly one. But the MAT_SPD flag is orthogonal to the choice of near-nullspace,
so a driver that wanted modes silently gave up SPD. They now share one implementation,
`_AssembledBlockPC`, with two independent settings, and remain three sibling names that
change only the defaults.

Note:
  The first two groups here test `gadopt/stokes_integrators.py`, which is the E1 work,
  and the third tests `gadopt/preconditioners.py`. They are separable changes and belong
  in different pull requests; keeping them in one file is a convenience for the branch,
  not a statement that they ship together.
"""

import firedrake as fd
import pytest

from gadopt.approximations import CompressibleInternalVariableApproximation
from gadopt.preconditioners import (
    NearlyIncompressibleAssembledPC,
    _AssembledBlockPC,
    RigidBodyAssembledPC,
    SPDAssembledPC,
    near_nullspace_basis,
)
from gadopt.stokes_integrators import (
    CoupledInternalVariableSolver,
    coupled_gia_solver_parameters,
)


@pytest.fixture(scope="module")
def mesh():
    m = fd.UnitCubeMesh(2, 2, 2)
    m.cartesian = True
    return m


def build(mesh, **approx_kwargs):
    """A coupled (u, m) solver on the unit cube, built but never solved."""
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    S = fd.TensorFunctionSpace(mesh, "DG", 1)
    Z = fd.MixedFunctionSpace([V, S])
    settings = dict(bulk_modulus=1.0, density=1.0, shear_modulus=1.0,
                    viscosity=1.0, g=1.0)
    settings.update(approx_kwargs)
    return CoupledInternalVariableSolver(
        fd.Function(Z), CompressibleInternalVariableApproximation(**settings),
        dt=1.0, bcs={1: {"ux": 0}},
    )


class TestOuterMethodFollowsTheExponent:
    def test_newtonian_uses_ksponly(self, mesh):
        assert build(mesh).solver_parameters["snes_type"] == "ksponly"

    def test_power_law_uses_newton(self, mesh):
        solver = build(mesh, exponent=3, transition_stress=1.0)
        assert solver.solver_parameters["snes_type"] == "newtonls"

    def test_a_spatially_varying_exponent_is_treated_as_power_law(self, mesh):
        """The safe direction. A `Function` exponent cannot be compared to 1.

        `tests/3d_weerdesteijn_coupled` builds exactly this: a DG0 field holding
        `[1, 3, 3, 1]` over the four layers, so two of them are genuinely
        power-law however the driver is invoked.
        """
        DG0 = fd.FunctionSpace(mesh, "DG", 0)
        exponent = fd.Function(DG0).assign(1.0)  # value is 1, type is not
        solver = build(mesh, exponent=exponent, transition_stress=1.0)
        assert solver.solver_parameters["snes_type"] == "newtonls"

    def test_a_direct_newtonian_solve_does_not_ask_for_newton(self, mesh):
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        S = fd.TensorFunctionSpace(mesh, "DG", 1)
        Z = fd.MixedFunctionSpace([V, S])
        solver = CoupledInternalVariableSolver(
            fd.Function(Z),
            CompressibleInternalVariableApproximation(
                bulk_modulus=1.0, density=1.0, shear_modulus=1.0,
                viscosity=1.0, g=1.0),
            dt=1.0, bcs={1: {"ux": 0}}, solver_parameters="direct",
        )
        assert solver.solver_parameters["snes_type"] == "ksponly"


class TestToleranceOrdering:
    """E1: a Newton step cannot reduce the residual by more than its linear solve.

    Under the static-condensation preset the outer Krylov method is `preonly`
    and the Krylov tolerance that does the work lives on the condensed
    displacement operator, under `condensed_field`. The preset carries no
    SNES tolerance of its own; for a power-law rheology the solver adds
    `newton_stokes_solver_parameters`, whose `snes_rtol` is the number the
    linear tolerance must beat.
    """

    def test_the_shipped_preset_puts_the_krylov_solve_on_the_condensed_field(self):
        assert coupled_gia_solver_parameters["ksp_type"] == "preonly"
        assert "ksp_rtol" not in coupled_gia_solver_parameters
        assert "snes_rtol" not in coupled_gia_solver_parameters

    @pytest.mark.xfail(
        strict=True,
        reason="The shipped preset pairs `spd_ksp_parameters` (ksp_rtol 1e-5) "
               "with `newton_stokes_solver_parameters` (snes_rtol 1e-5): equal, "
               "not strictly ordered. Same pairing as the substituted solver "
               "on main. Open item in NOTES/PLAN-COUPLED-TRANSITION.md §12.",
    )
    def test_the_ordering_survives_into_a_built_solver(self, mesh):
        p = build(mesh, exponent=3, transition_stress=1.0).solver_parameters
        assert p["condensed_field"]["ksp_rtol"] < p["snes_rtol"]

    def test_the_linear_solve_is_never_looser_than_newton(self, mesh):
        """The weaker statement that the shipped preset does satisfy."""
        p = build(mesh, exponent=3, transition_stress=1.0).solver_parameters
        assert p["condensed_field"]["ksp_rtol"] <= p["snes_rtol"]


class TestAssembledPCFactoring:
    """One SPD flag, one three-valued near-nullspace axis, not three siblings."""

    def test_the_three_names_are_siblings_over_one_implementation(self):
        """Siblings, not a chain.

        `RigidBodyAssembledPC` does not set MAT_SPD, so inheriting it from a
        class called `SPDAssembledPC` would assert in its name something its
        defaults switch off.
        """
        base = _AssembledBlockPC
        for cls in (SPDAssembledPC, RigidBodyAssembledPC,
                    NearlyIncompressibleAssembledPC):
            assert cls.__bases__ == (base,)
        assert not issubclass(RigidBodyAssembledPC, SPDAssembledPC)

    def test_defaults_are_unchanged_by_the_refactor(self):
        """The published iteration counts were measured with these defaults."""
        assert (SPDAssembledPC._spd, SPDAssembledPC._near_nullspace) == (True, "none")
        assert (RigidBodyAssembledPC._spd,
                RigidBodyAssembledPC._near_nullspace) == (False, "rigid")
        assert (NearlyIncompressibleAssembledPC._spd,
                NearlyIncompressibleAssembledPC._near_nullspace) == (
                    False, "incompressible")

    def test_the_modes_really_are_a_superset_ladder(self, mesh):
        """none < rigid < incompressible, measured, which is why one slot suffices.

        The point of collapsing the near-nullspace onto a single three-valued
        axis is that the three settings are nested. Asserting the ladder by
        counting vectors, rather than by re-reading the class attributes that
        define it.
        """
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        n_rigid = len(near_nullspace_basis(V, "rigid")._petsc_vecs)
        n_incomp = len(near_nullspace_basis(V, "incompressible")._petsc_vecs)
        assert 0 < n_rigid < n_incomp

    def test_an_unknown_mode_set_is_refused(self, mesh):
        """The failure mode being removed was silence, so this must not be silent.

        Checked on `near_nullspace_basis` rather than through a solve, because a
        `ValueError` raised inside a python PC callback does not come back as
        itself. What it does instead depends on the configuration: on a plain
        assembled `CG1` solve it took the whole pytest process down with SIGSEGV
        (exit 139), while under `mat_type: matfree` it surfaces as
        `PETSc.Error: error code 101`, naming nothing. Neither is catchable as
        the original exception, which is why the name check lives in a function
        callable without a `PC`, and why `initialize` prints before it raises.
        """
        V = fd.VectorFunctionSpace(mesh, "CG", 1)
        with pytest.raises(ValueError, match="none.*rigid.*incompressible"):
            near_nullspace_basis(V, "rigid-body")

    @pytest.mark.parametrize("modes,expected", [("rigid", 6),
                                                ("incompressible", 11)])
    def test_each_mode_set_has_the_size_it_claims(self, mesh, modes, expected):
        """3-D: six rigid modes, eleven for the complete linear div-free space."""
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        assert len(near_nullspace_basis(V, modes)._petsc_vecs) == expected

    def test_spd_and_modes_are_now_independently_reachable(self, mesh):
        """The live defect in the old factoring: modes cost you the SPD flag.

        Three sibling classes meant `pc_python_type` chose exactly one, and the
        only class carrying MAT_SPD carried no near-nullspace. This combination
        was unreachable before and must stay reachable.

        Asserting on the near-nullspace actually attached to the assembled
        block, not merely that the solve returned something: a solve that
        silently ignored BOTH options would still converge, and that silence is
        the exact failure the refactor exists to remove.
        """
        seen = {}

        class Spy(SPDAssembledPC):
            def initialize(self, pc):
                super().initialize(pc)
                nns = self.P.petscmat.getNearNullSpace()
                seen["vecs"] = len(nns.getVecs()) if nns.handle else 0
                seen["spd"] = self._spd_wanted

        globals()["_SpyPC"] = Spy
        V = fd.VectorFunctionSpace(mesh, "CG", 1)
        u, v = fd.TrialFunction(V), fd.TestFunction(V)
        a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx + fd.inner(u, v) * fd.dx
        L = fd.inner(fd.Constant((1.0, 0.0, 0.0)), v) * fd.dx
        uh = fd.Function(V)
        fd.solve(a == L, uh, solver_parameters={
            "ksp_type": "cg", "pc_type": "python",
            "pc_python_type": f"{__name__}._SpyPC",
            "near_nullspace": "incompressible",
            "spd": True,
            "assembled_pc_type": "gamg",
        })
        # 3-D: 11 divergence-free modes, and the SPD flag alongside them.
        assert seen == {"vecs": 11, "spd": True}
        assert fd.norm(uh) > 0.0
