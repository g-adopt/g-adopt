"""The monolithic self-gravitating GIA solver: space layout, constants, gates.

What these tests are for, and what they are not. They cover the *structure* of
`gadopt.gia_gravity` - the space layout the preconditioner depends on, the two
row-scaling constants, the identity that makes the rotational body force the
transpose of the inertia row, the assertions that turn silently-empty measures
into exceptions, and V4 - and they run the solver once end to end on a small
mesh to show that it solves at all.

They are **not** the physics gates. V1 (per-block Jacobian symmetry), V3'
(the potential against a closed form), V7 (the rotational closure), V8 (the
geoid sign), V9 and V9b (the fluid limit) live elsewhere and are the next piece
of work. The one gate here is V4, because it is a statement about the *form*
rather than about a solution and belongs with the code that writes the form.

The mesh is gmsh, and it has to be: the geometry stands the DtN boundaries off
from the sources, which puts the Earth's surface *inside* the domain as a
tagged interior facet, and only gmsh can tag one. It is coarse - the tests here
are structural, and nothing in them is a convergence statement.
"""

import tempfile
from pathlib import Path

import firedrake as fd
import numpy as np
import pytest
from ufl.algorithms import extract_coefficients

from gadopt import (
    CompressibleInternalVariableApproximation,
    CoupledInternalVariableSolver,
    CylindricalDtN,
    SelfGravitatingGIASolver,
    rigid_rotation_nullspace,
    self_gravitating_gia_space,
)
from gadopt.gia_gravity import (
    NULL_COUPLING_ROW_SCALE,
    OMEGA_SQ_EARTH,
    FluidCore,
)
from gadopt.momentum_equation import rotational_potential

# Road-map §2.2 with the production constants.
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3

# The Phase 2 tag convention (road map §1.5), which the 3-D mesh shares.
CELL_MANTLE = 101
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5

RC, RE = 1.2037, 2.2037
N_AZIMUTHAL = 32
DR_MANTLE = 0.2


@pytest.fixture(scope="module")
def meshes():
    """The parent annulus and its mantle submesh, both P2-curved.

    A fixed path rather than `tmp_path_factory`: under MPI every rank runs its
    own pytest and would otherwise be handed a different directory. Curved on
    both sides, because `Submesh` does not inherit the parent's P2 coordinates
    and the straight-sided version misplaces exactly the two circles the
    interface mass sheets live on.
    """
    pytest.importorskip("gmsh")
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "demos" / "gravity"))
    import generate_selfgrav_annulus as gen
    from validate_selfgrav_annulus import curve_mesh

    path = Path(tempfile.gettempdir()) / (
        f"gadopt_selfgrav_gia_{DR_MANTLE}_{N_AZIMUTHAL}.msh")
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        gen.generate(str(path), dr_mantle=DR_MANTLE, n_azimuthal=N_AZIMUTHAL)
    fd.COMM_WORLD.barrier()

    parent = curve_mesh(fd.Mesh(str(path)))
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def approximation(**kwargs):
    """A fresh approximation every time; the solver mutates `mu` in place."""
    settings = dict(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)
    settings.update(kwargs)
    return CompressibleInternalVariableApproximation(**settings)


def gravity_bcs(parent, *, truncation=3, sheet=True):
    X = fd.SpatialCoordinate(parent)
    bcs = {
        CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    if sheet:
        bcs[CURVE_RE] = {
            "interior_sigma": SIGMA_HAT * fd.cos(2 * fd.atan2(X[1], X[0]))}
    return bcs


def mechanics_bcs(sub):
    Xm = fd.SpatialCoordinate(sub)
    return {
        CURVE_RC: {"un": 0.0},
        CURVE_RE: {"normal_stress":
                   B_MU * SIGMA_HAT * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))},
    }


def build(meshes, *, rotation=True, n_internal_variables=1,
          approximation_kwargs=None, declare_nullspace=False, **kwargs):
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent), rotation=rotation,
        n_internal_variables=n_internal_variables,
        self_gravity_number=LAMBDA)
    z = fd.Function(Z)
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    C = fd.assemble(fd.dot(Xm, Xm) * dx_m)
    if declare_nullspace:
        kwargs.setdefault("nullspace", rigid_rotation_nullspace(Z, layout))
    solver = SelfGravitatingGIASolver(
        z, approximation(**(approximation_kwargs or {})), layout=layout,
        dt=1.0, bcs=mechanics_bcs(sub), rotation_moments={"C": C}, **kwargs)
    return solver, z, layout


@pytest.fixture(scope="module")
def solved(meshes):
    """One coupled solve, shared by the tests that only read a result."""
    solver, z, layout = build(meshes)
    solver.solve()
    return solver, z, layout


class TestSpaceLayout:
    """The invariants `DtNTwoBlockSchurPC` and the 3-D promotion depend on."""

    def test_field_order_and_meshes(self, meshes):
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            n_internal_variables=2, self_gravity_number=LAMBDA)

        assert layout.displacement == 0
        assert layout.internal_variables == (1, 2)
        assert layout.potential == 3
        assert len(Z) == layout.n_fields
        assert Z[layout.displacement].mesh() is sub
        assert Z[layout.internal_variables[0]].mesh() is sub
        assert Z[layout.potential].mesh() is parent

    def test_real_fields_are_contiguous_and_last(self, meshes):
        """The invariant `DtNTwoBlockSchurPC.initialize` asserts.

        Anything else would leave sub-fields out of both of its two blocks,
        which is a silently wrong split rather than an error.
        """
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            self_gravity_number=LAMBDA)

        real = [i for i, V in enumerate(Z)
                if V.ufl_element().family() == "Real"]
        assert real == list(range(real[0], len(Z)))
        assert tuple(real) == layout.real_fields
        assert real[0] > 0
        # Multipliers first, rotation last, and the rotation scalars are the
        # very end of the space.
        assert layout.multipliers == tuple(real[:len(layout.multipliers)])
        assert sorted(layout.rotation.values()) == real[len(layout.multipliers):]

    def test_real_spaces_live_on_the_parent(self, meshes):
        """Spike S2's decision, and it is not cosmetic.

        On the parent both families of constraint row - parent-boundary DtN
        rows and submesh-volume rotation rows - assemble against the measures
        the mechanics already needs. On the submesh every parent-boundary row
        needs a specially intersected `ds`, and an intersected measure that
        finds nothing assembles to zero without raising.
        """
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            self_gravity_number=LAMBDA)
        for i in layout.real_fields:
            assert Z[i].mesh() is parent

    def test_multiplier_count_comes_from_the_boundaries(self, meshes):
        """`2M` on the exterior boundary, `2M + 1` on the interior one.

        And the side is *measured* from the mesh rather than declared, which is
        why the space cannot be built before the boundary conditions are known
        and why the factory takes them.
        """
        parent, sub = meshes
        for M in (3, 5):
            _, layout = self_gravitating_gia_space(
                sub, parent, gravity_bcs=gravity_bcs(parent, truncation=M),
                self_gravity_number=LAMBDA)
            assert len(layout.multipliers) == (2 * M) + (2 * M + 1)

    def test_rotation_is_named_m3_in_2d(self, meshes):
        """Index **2** of the rotation triple, never "the first rotation field".

        A disc has no polar wander: `m_1` and `m_2` tilt the rotation axis out
        of a plane with no third direction, and the only surviving mode is the
        rotation-rate change. A layout that recorded it positionally would
        silently reindex on promotion to 3-D and every test reading a rotation
        coefficient by position would change meaning.
        """
        parent, sub = meshes
        _, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            self_gravity_number=LAMBDA)
        assert set(layout.rotation) == {"m3"}
        assert layout.rotation_slots() == (None, None, layout.rotation["m3"])

    def test_no_rotation_by_default(self, meshes):
        parent, sub = meshes
        _, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent),
            self_gravity_number=LAMBDA)
        assert layout.rotation == {}
        assert layout.rotation_slots() == (None, None, None)

    def test_parent_must_declare_cartesian(self, meshes):
        """G-ADOPT reads `mesh.cartesian` everywhere and a file mesh has none.

        Refused in the factory, where both meshes are in scope and the message
        can say what to set, rather than several frames deep in
        `upward_normal`.
        """
        _, sub = meshes
        undeclared = fd.UnitSquareMesh(2, 2)
        assert not hasattr(undeclared, "cartesian")
        with pytest.raises(ValueError, match="cartesian"):
            self_gravitating_gia_space(
                sub, undeclared, gravity_bcs={}, self_gravity_number=LAMBDA)

    def test_submesh_inherits_cartesian_from_the_parent(self, meshes):
        """`Submesh` does not inherit it, and the failure is far from the cause.

        Without it the first thing that asks the submesh whether it is
        Cartesian raises `AttributeError: 'MeshTopology' object has no
        attribute 'cartesian'` from inside `upward_normal`.
        """
        parent, sub = meshes
        fresh = fd.Submesh(parent, 2, CELL_MANTLE)
        assert not hasattr(fresh, "cartesian")
        self_gravitating_gia_space(
            fresh, parent, gravity_bcs=gravity_bcs(parent),
            self_gravity_number=LAMBDA)
        assert fresh.cartesian == parent.cartesian


class TestScalingConstants:
    """Asserted from the derivation, and never fitted.

    A symmetry test is an equation *for* a row scaling, so a scaling adjusted
    until the test passes absorbs any sign error in the block it scales and
    reports success. These are the values the later symmetry gate must pin.
    """

    def test_theta_psi_is_B_mu_over_Lambda(self, meshes):
        solver, _, _ = build(meshes)
        assert float(solver.theta_psi) == pytest.approx(B_MU / LAMBDA, rel=1e-14)
        # The review's number, to the digits it prints.
        assert float(solver.theta_psi) == pytest.approx(1.1487, abs=5e-5)

    def test_theta_psi_carries_the_residual_scaling_factor(self, meshes):
        """The part neither design document mentions.

        `CoupledInternalVariableSolver` multiplies the whole momentum residual
        by `scaling_factor`, the two body forces included, so a `theta_psi`
        that ignored it would make the coupled Jacobian asymmetric by exactly
        that factor - which reads as a sign error in the new code.
        """
        solver, _, _ = build(meshes, scaling_factor=3.0)
        assert float(solver.theta_psi) == pytest.approx(
            3.0 * B_MU / LAMBDA, rel=1e-14)

    def test_theta_rot_is_a_third_constant_carrying_Omega_squared(self, meshes):
        """`theta_rot_i = s_i B_mu Omega^2`, and **negative** on the m_3 row.

        Not `theta_psi`: it carries an `Omega^2` that `theta_psi` has none of.
        The closure sign `s_3 = -1` differs from the polar-wander pair, and m_3
        is the only component a disc has - so the 2-D prototype exercises only
        the row with the negative scaling.
        """
        solver, _, _ = build(meshes)
        expected = -B_MU * OMEGA_SQ_EARTH
        assert float(solver._theta_rot(2)) == pytest.approx(expected, rel=1e-14)
        assert float(solver._theta_rot(2)) < 0.0
        assert float(solver._theta_rot(0)) == pytest.approx(
            +B_MU * OMEGA_SQ_EARTH, rel=1e-14)
        # And it is not theta_psi, by three orders of magnitude.
        assert abs(float(solver._theta_rot(2))) < 1e-2 * float(solver.theta_psi)

    def test_omega_sq_earth(self):
        """`Omega^2 L / g_bar`, recorded nowhere else."""
        omega, length, g_bar = 7.292e-5, 2.891e6, 9.815
        assert OMEGA_SQ_EARTH == pytest.approx(
            omega**2 * length / g_bar, rel=2e-3)


class TestInertiaTranspose:
    """The identity that makes the operator symmetric rather than merely close.

    `psi_rot = Omega^2 sum_i m_i p_i` with the *same* `p_i` whose gradient
    contracts with `u` to give `dI_i3`. If the two ever drift apart the
    rotational body force stops being the transpose of the closure row, and no
    choice of `theta_rot` repairs it.
    """

    def test_polynomial_matches_the_rotational_potential_in_2d(self, meshes):
        _, sub = meshes
        X = fd.SpatialCoordinate(sub)
        p3 = SelfGravitatingGIASolver.inertia_polynomial(2, X)
        psi_rot = rotational_potential([1.0], sub, Omega_sq=1.0)
        assert fd.assemble((psi_rot - p3) ** 2 * fd.dx(domain=sub)) < 1e-24

    def test_polynomial_matches_the_rotational_potential_in_3d(self):
        """Written in three components unconditionally, so 3-D comes free."""
        mesh = fd.UnitCubeMesh(1, 1, 1)
        mesh.cartesian = True
        X = fd.SpatialCoordinate(mesh)
        for i, m in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                               (0.0, 0.0, 1.0)]):
            p_i = SelfGravitatingGIASolver.inertia_polynomial(i, X)
            psi_rot = rotational_potential(list(m), mesh, Omega_sq=1.0)
            assert fd.assemble((psi_rot - p_i) ** 2 * fd.dx(domain=mesh)) < 1e-24

    def test_2d_refuses_a_polar_wander_component(self):
        mesh = fd.UnitSquareMesh(1, 1)
        X = fd.SpatialCoordinate(mesh)
        with pytest.raises(ValueError, match="third coordinate"):
            SelfGravitatingGIASolver.inertia_polynomial(0, X)

    def test_inertia_of_a_cos2_sheet_vanishes(self, solved):
        """`int sigma (x^2 + y^2) dS` over a circle kills a `cos 2phi` load.

        Not a tautology: it is the sheet half of `dI_33`, it goes through
        `sheet_integral` on a tagged *interior* facet, and a sheet written on
        the wrong kind of measure would give the same zero - which is why the
        measure assertions exist separately.
        """
        solver, _, _ = solved
        assert abs(solver.inertia_perturbation()["dI_33"]) < 1e-14
        assert abs(solver.rotation_values()["m3"]) < 1e-14


class TestGeometryAssertions:
    """Silent zeros turned into exceptions, at construction."""

    def test_a_sheet_on_the_wrong_kind_of_facet_is_refused(self, meshes):
        """Re is an interior facet of the parent; `sigma` would find nothing.

        A load sheet written the shipped exterior-facet way contributes
        nothing, warns rather than raising, and the solve converges with the
        potential missing the largest single term in the geoid. Nothing
        downstream looks: a sheet is a right-hand side with no Jacobian
        contribution, so a symmetry test cannot see it, and two solvers sharing
        the form omit it identically.
        """
        parent, sub = meshes
        bcs = gravity_bcs(parent, sheet=False)
        bcs[CURVE_RE] = {"sigma": SIGMA_HAT}
        with pytest.raises(ValueError, match="sheet measure is empty"):
            self_gravitating_gia_space(
                sub, parent, gravity_bcs=bcs, self_gravity_number=LAMBDA)

    def test_dtn_boundary_measures_match_their_radii(self, solved):
        """What `check_geometry` asserts, restated so a reader can see it.

        A tag matching half a circle gives half the perimeter and is otherwise
        undetectable.
        """
        solver, _, _ = solved
        for bc_id, _ in solver.form.dtn_boundaries:
            _, radius = solver.form.boundary_geometry[bc_id]
            length = solver.form.boundary_area[bc_id]
            assert length == pytest.approx(2 * np.pi * radius, rel=1e-2)

    def test_cross_mesh_measure_finds_the_parent(self, solved):
        """The one check that tests the entity maps rather than the measure.

        A parent coefficient integrated over the mantle must give the mantle's
        area. Phase 2 measured 1.2e-08 for the curved pair against 4.02e-04 for
        an un-recurved submesh, and the error lands exactly at Rc and Re.
        """
        solver, _, _ = solved
        probe = fd.Function(
            fd.FunctionSpace(solver.potential_mesh, "CG", 1)).assign(1.0)
        area = fd.assemble(probe * solver.dx_m)
        assert area == pytest.approx(np.pi * (RE**2 - RC**2), rel=1e-3)

    def test_intersected_parent_measure_covers_the_whole_parent(self, solved):
        """Intersecting must not restrict the Laplacian to the mantle.

        Firedrake keeps the parent's own cells and uses the intersection only
        to widen the admissible domains; the alternative reading is a converged
        solve with the stand-off buffer and both DtN boundaries absent.
        """
        solver, _, _ = solved
        whole = fd.assemble(fd.Constant(1.0) * solver.dx_parent)
        assert fd.assemble(fd.Constant(1.0) * solver.dx_g) == pytest.approx(
            whole, rel=1e-12)
        assert whole == pytest.approx(np.pi * ((2 * RE) ** 2 - (0.5 * RC) ** 2),
                                      rel=1e-3)

    def test_lambda_must_agree_between_factory_and_approximation(self, meshes):
        """The one error class a symmetry test provably cannot see.

        The factory turns `Lambda` into the boundary form's `G = Lambda/(4 pi)`
        because the sheets carry an explicit `4 pi G sigma` while the volume
        source has it absorbed. A mismatch scales the sheet against the volume
        source by a constant and the geoid is wrong by that factor with a
        perfectly plausible magnitude.
        """
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent),
            self_gravity_number=LAMBDA)
        with pytest.raises(ValueError, match="self_gravity_number disagrees"):
            SelfGravitatingGIASolver(
                fd.Function(Z), approximation(self_gravity_number=2 * LAMBDA),
                layout=layout, dt=1.0, bcs=mechanics_bcs(sub))

    def test_a_sheet_without_a_declared_lambda_is_refused(self, meshes):
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent))
        with pytest.raises(ValueError, match="mass sheets"):
            SelfGravitatingGIASolver(
                fd.Function(Z), approximation(), layout=layout, dt=1.0,
                bcs=mechanics_bcs(sub))


class TestConstruction:
    """What the base classes do on a mixed space spanning two meshes."""

    def test_mesh_is_the_mechanics_mesh(self, solved):
        """`StokesSolverBase.__init__` writes a `MeshSequenceGeometry` here.

        It succeeds and then `upward_normal` raises one line later, so
        re-assigning after `super().__init__` is not an option - it never
        returns. The property discards anything that is not a real mesh.
        """
        solver, _, _ = solved
        assert isinstance(solver.mesh, fd.MeshGeometry)
        assert solver.mesh is solver.layout.mechanics_mesh
        assert solver.mesh is not solver.potential_mesh

    def test_one_equation_per_mechanics_field(self, solved):
        """And not one per sub-field: the multipliers are not `Equation`s.

        The parent's `set_form` zips equations against sub-fields, and here
        that would truncate - correctly, by accident, which is exactly the kind
        of accident that stops being true when somebody reorders the space.
        """
        solver, _, layout = solved
        assert len(solver.equations) == 1 + len(layout.internal_variables)

    def test_power_law_rheology_is_refused(self, meshes):
        """`DtNTwoBlockSchurPC.update` is a no-op, so the block goes stale.

        Correct for Newtonian GIA and false the moment the viscosity depends on
        the state. A stale preconditioner fails silently, which is this
        project's recurring failure mode, so the combination is refused rather
        than warned about.
        """
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent),
            self_gravity_number=LAMBDA)
        with pytest.raises(ValueError, match="exponent"):
            SelfGravitatingGIASolver(
                fd.Function(Z), approximation(exponent=3.0), layout=layout,
                dt=1.0, bcs=mechanics_bcs(sub))

    def test_it_solves(self, solved):
        solver, z, layout = solved
        assert fd.norm(solver.displacement) > 0.0
        assert fd.norm(solver.potential) > 0.0
        assert np.isfinite(fd.norm(solver.potential))

    def test_the_answer_is_degree_two(self, solved):
        """A `cos 2phi` load excites `cos 2phi` and nothing else.

        The cheapest end-to-end statement that the coupling is wired to the
        right places at all: every other trace coefficient, on both DtN
        boundaries, sits at roundoff.
        """
        solver, _, _ = solved
        for bc_id, modes in solver.coefficients().items():
            assert abs(modes["cos2"]) > 1e-6
            for key, value in modes.items():
                if key != "cos2":
                    assert abs(value) < 1e-9 * abs(modes["cos2"]) + 1e-15


class TestV4:
    """V4 - discrete mass conservation of the divergence source.

    `Lambda int rho_0 u . grad(v)` with `v = 1` is zero because the gradient is,
    so the discrete perturbation carries exactly zero net mass - to roundoff,
    not to `O(h^p)`, and independently of mesh, quadrature, `Lambda` and every
    sign in the system. It fails only if somebody rewrites the source in
    non-divergence form, which is precisely what it is guarding.

    In 2-D it is also what keeps the solver inside the regime its monopole and
    log-gauge treatment was built for.
    """

    def test_source_form_with_v_equal_one_is_machine_zero(self, solved):
        solver, _, _ = solved
        value = fd.assemble(solver.source_mass_form())
        u = solver.solution_split[solver.layout.displacement]
        scale = float(solver.Lambda) * fd.assemble(
            solver.approximation.density * fd.sqrt(fd.dot(u, u)) * solver.dx_m)
        assert scale > 0.0, "the test is vacuous on a zero displacement"
        assert abs(value) < 1e-16 * scale

    def test_the_constant_is_a_real_function_not_a_ufl_constant(self, solved):
        """Otherwise the test is a statement about UFL's constant folding.

        `grad(Constant(1))` folds to symbolic zero and the integral disappears
        before anything is discretised. Built as a `Function` of unit degrees
        of freedom, the gradient is the real one and the zero is the discrete
        statement that `v = 1` lies in the test space.
        """
        solver, _, _ = solved
        form = solver.source_mass_form()
        assert form.integrals(), "the source-mass form folded away entirely"
        # No family filter: on a quadrilateral mesh the continuous element is
        # reported as "Q", not "Lagrange".
        unit = [c for c in extract_coefficients(form)
                if isinstance(c, fd.Function)
                and c.function_space() is solver.solution_space[
                    solver.layout.potential]
                and np.allclose(c.dat.data_ro, 1.0)]
        assert unit, "no unit function on the potential space in the form"

    def test_net_enclosed_mass_is_the_sheets_alone(self, solved):
        """A `cos 2phi` sheet carries no net mass, so the datum is idle.

        The volume source contributes nothing by construction, which is the
        whole difference from `GravitySolver`'s mass bookkeeping.
        """
        solver, _, _ = solved
        assert abs(solver.total_enclosed_mass()) < 1e-14


class TestNullCoupling:
    """`B_mu = 0` is a supported configuration, not a singular Jacobian.

    Defect D-1. Both row scalings are `B_mu` times something and both multiply
    whole residual rows, so following `B_mu` to zero deleted the potential
    equation, every DtN constraint row and the rotation row rather than
    decoupling them - 23 identically zero rows and `DIVERGED_LINEAR_SOLVE` at
    the first Krylov iteration. `NULL_COUPLING_ROW_SCALE` floors them, on the
    grounds that the symmetry condition the scalings are derived from is
    vacuous at `B_mu = 0`: the block it constrains is itself zero.
    """

    def test_row_scalings_are_floored_and_not_zero(self, meshes):
        solver, _, _ = build(meshes, approximation_kwargs={"B_mu": 0.0})
        assert float(solver.theta_psi) == pytest.approx(
            NULL_COUPLING_ROW_SCALE / LAMBDA, rel=1e-14)
        assert float(solver._theta_rot(2)) == pytest.approx(
            -NULL_COUPLING_ROW_SCALE * OMEGA_SQ_EARTH, rel=1e-14)
        assert float(solver.theta_psi) != 0.0
        assert float(solver._theta_rot(2)) != 0.0

    def test_a_nonzero_B_mu_is_untouched_to_the_bit(self, meshes):
        """The floor is a strict test on zero, so nothing production changes.

        A tolerance would put a discontinuity in the middle of the parameter
        range, which a continuation study would walk straight into.
        """
        solver, _, _ = build(meshes, approximation_kwargs={"B_mu": 1e-12})
        assert float(solver.theta_psi) == 1e-12 / LAMBDA

    @pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs a global dense transpose")
    def test_the_potential_row_is_not_annihilated(self, meshes):
        """The measurement the defect was found by: `max|A(psi, psi)|`.

        Serial: `getNestSubMatrix(...).convert("dense")` is spike S5's route B,
        which builds a global dense block and is the reason every per-block
        measurement in this project is a serial one.
        """
        solver, z, layout = build(meshes, approximation_kwargs={"B_mu": 0.0})
        A = fd.assemble(fd.derivative(solver.F, z), mat_type="nest").petscmat

        def amax(i, j):
            M = A.getNestSubMatrix(i, j)
            return 0.0 if M is None else float(
                np.abs(M.convert("dense").getDenseArray()).max())

        assert amax(layout.potential, layout.potential) > 0.0
        assert amax(layout.multipliers[0], layout.potential) > 0.0
        # The coupling itself *is* off: the body force is gone.
        assert amax(layout.displacement, layout.potential) == 0.0
        # ... while the one-way leg survives, so psi is still driven by u.
        assert amax(layout.potential, layout.displacement) > 0.0

    def test_it_reproduces_the_uncoupled_solver(self, meshes):
        """V2, run as specified rather than around the defect.

        Both sides declare the rigid-rotation kernel (defect D-2); without it
        the displacements agree only to ~1e-07, the difference being entirely
        that zero-energy mode.
        """
        _, sub = meshes
        solver, z, layout = build(
            meshes, approximation_kwargs={"B_mu": 0.0}, declare_nullspace=True)
        solver.solve()

        V = fd.VectorFunctionSpace(sub, "CG", 2)
        S = fd.TensorFunctionSpace(sub, "DG", 1)
        Zm = fd.MixedFunctionSpace([V, S])
        zm = fd.Function(Zm)
        X = fd.SpatialCoordinate(sub)
        basis = fd.VectorSpaceBasis(
            [fd.Function(Zm.sub(0)).interpolate(fd.as_vector([-X[1], X[0]]))])
        basis.orthonormalize()
        ref = CoupledInternalVariableSolver(
            zm, approximation(B_mu=0.0), dt=1.0, bcs=mechanics_bcs(sub),
            solver_parameters="direct",
            nullspace=fd.MixedVectorSpaceBasis(Zm, [basis, Zm.sub(1)]))
        ref.solve()
        basis.orthogonalize(zm.subfunctions[0])

        uc = z.subfunctions[layout.displacement]
        ur = zm.subfunctions[0]
        assert fd.norm(ur) > 0.0
        assert fd.norm(fd.assemble(uc - ur)) / fd.norm(ur) < 1e-11
        for k, i in enumerate(layout.internal_variables):
            mr = zm.subfunctions[1 + k]
            assert fd.norm(
                fd.assemble(z.subfunctions[i] - mr)) / fd.norm(mr) < 1e-11
        # ... and psi is still driven, so the test is not vacuous.
        assert fd.norm(solver.potential) > 0.0


class TestRigidRotationNullspace:
    """Defect D-2: the zero-energy mode `u = (-y, x)`, declared and removed.

    Free slip `un = 0` at the CMB and traction at the surface leave the
    tangential displacement unconstrained, so a rigid rotation of the whole
    mantle costs no energy - and costs none in the coupling either, since it is
    divergence free, tangential to both circles and contributes nothing to
    `dI_33`. Discretely it survives only through facet geometry error.
    """

    def test_the_mode_is_nearly_annihilated_by_the_jacobian(self, meshes):
        """The measurement: `||J u_rot|| / ||u_rot||`, small but not zero."""
        _, sub = meshes
        solver, z, layout = build(meshes)
        X = fd.SpatialCoordinate(sub)
        probe = fd.Function(z.function_space())
        probe.subfunctions[layout.displacement].interpolate(
            fd.as_vector([-X[1], X[0]]))
        J = fd.assemble(fd.derivative(solver.F, z), mat_type="matfree")
        with probe.dat.vec_ro as xv:
            yv = J.petscmat.createVecLeft()
            J.petscmat.mult(xv, yv)
            ratio = yv.norm() / xv.norm()
        # Nonzero, so the discrete operator is not exactly singular; tiny, so
        # it is nonsingular only through the facets. Both halves matter.
        assert 0.0 < ratio < 1e-3

    def test_it_spans_the_displacement_block_only(self, meshes):
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            self_gravity_number=LAMBDA)
        ns = rigid_rotation_nullspace(Z, layout)
        bases = list(ns)
        assert isinstance(bases[layout.displacement], fd.VectorSpaceBasis)
        assert len(bases[layout.displacement]._petsc_vecs) == 1
        for i, entry in enumerate(bases):
            if i != layout.displacement:
                assert not isinstance(entry, fd.VectorSpaceBasis)

    def test_declaring_it_removes_it_from_the_answer(self, meshes):
        """PETSc alone does not: FGMRES is right-preconditioned.

        The declaration removes the kernel from the right-hand side but not
        from the preconditioner's output, and with an almost-exact
        preconditioner the answer is the preconditioner's output. So the solver
        projects after solving; this is the measurement that it does.
        """
        _, sub = meshes
        X = fd.SpatialCoordinate(sub)

        def rotation_content(solver):
            V = solver.displacement.function_space()
            r0 = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))
            dxm = fd.Measure("dx", domain=sub)
            c = (fd.assemble(fd.dot(solver.displacement, r0) * dxm)
                 / fd.assemble(fd.dot(r0, r0) * dxm))
            return abs(c) * fd.norm(r0) / fd.norm(solver.displacement)

        plain, _, _ = build(meshes)
        plain.solve()
        declared, _, _ = build(meshes, declare_nullspace=True)
        declared.solve()

        assert rotation_content(declared) < 1e-14
        assert rotation_content(declared) < 1e-3 * rotation_content(plain)


class TestFluidCore:
    """`FluidCore`: the CMB condition that replaces the legacy `un = 0`.

    Structural only, as everything in this file is. The physics - the
    contrast-zero cancellation, the sign of the CMB stiffness, its magnitude
    against 1.744945, and the nullspace consequences - is
    `demos/gravity/spikes/gate_fluidcore.py`.
    """

    @staticmethod
    def build_fluid(meshes, **fluid_core_kwargs):
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=False,
            self_gravity_number=LAMBDA)
        z = fd.Function(Z)
        Xm = fd.SpatialCoordinate(sub)
        bcs = {CURVE_RE: {"normal_stress":
                          B_MU * SIGMA_HAT * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))}}
        bcs.update(fluid_core_kwargs.pop("extra_bcs", {}))
        settings = {"boundary": CURVE_RC, "rho_core": 2.0}
        settings.update(fluid_core_kwargs)
        solver = SelfGravitatingGIASolver(
            z, approximation(), layout=layout, dt=1.0, bcs=bcs,
            fluid_core=FluidCore(**settings))
        return solver, z, layout

    def test_refuses_un_on_the_same_boundary(self, meshes):
        """The two treatments are a switch, not layers: `un` pins `dot(u, n)`."""
        with pytest.raises(ValueError, match="rigid-core switch"):
            self.build_fluid(meshes, extra_bcs={CURVE_RC: {"un": 0.0}})

    def test_refuses_a_tag_that_is_not_a_mechanics_facet(self, meshes):
        """A wrong-kind tag gives zero and a warning, so it is refused instead.

        The parent's outer DtN boundary carries no facet of the mantle at all,
        which is the same silent failure as asking for an interior tag as an
        exterior one.
        """
        with pytest.raises(ValueError, match="empty measure"):
            self.build_fluid(meshes, boundary=CURVE_OUTER)

    def test_absent_by_default(self, meshes):
        """No `fluid_core` means an empty form, not a zero-valued one."""
        solver, z, _ = build(meshes, rotation=False)
        assert solver.fluid_core is None
        assert solver.fluid_core_residual().empty()

    def test_the_pair_transposes_exactly(self, meshes):
        """One energy, two variations: the blocks are transposes by construction.

        Exactly, and not to a tolerance - `derivative` of a single scalar form
        is the only spelling in which this cannot be got wrong, which is why
        the condition is written that way.
        """
        solver, z, layout = self.build_fluid(meshes)
        A = fd.assemble(fd.derivative(solver.fluid_core_residual(), z),
                        mat_type="nest").petscmat
        iu, ip = layout.displacement, layout.potential
        upsi = A.getNestSubMatrix(iu, ip).convert("dense").getDenseArray()
        psiu = A.getNestSubMatrix(ip, iu).convert("dense").getDenseArray()
        assert np.abs(upsi).max() > 0.0
        assert np.abs(upsi - psiu.T).max() == 0.0

    def test_the_measure_is_the_cmb_circle(self, meshes):
        """`2 pi Rc`, measured on the facets the condition actually integrates.

        The measure is the mantle's own `ds` intersected with the parent's
        `dS`: a facet integral must be paired with a *facet* measure, and
        pairing it with the parent's cell measure evaluates the parent's field
        at the wrong points with no warning at all.
        """
        solver, _, _ = self.build_fluid(meshes)
        length = fd.assemble(
            fd.Constant(1.0) * solver.fluid_core_measure()(CURVE_RC))
        assert abs(length - 2 * np.pi * RC) < 1e-3 * 2 * np.pi * RC

    def test_the_sheet_reaches_the_inertia_row(self, meshes):
        """`dI` must see the core sheet, or polar motion is silently short.

        The core boundary moves and the core's mass moves with it, so the sheet
        is a genuine mass redistribution and contributes to the inertia
        perturbation exactly as the ice load does. Nothing else in the fluid
        core's gates looks at `dI`: the potential would be right and this
        contribution absent, surfacing much later as a polar-motion answer
        wrong by an amount nobody can attribute.

        With `u = rhat` the sheet is `rho_core` and `p_3 = x^2 + y^2 = Rc^2` on
        the CMB, so its contribution is `rho_core Rc^2 2 pi Rc`. Measured
        against that, and against the same form with the core switched off.
        """
        rho_core = 2.0
        solver, z, layout = self.build_fluid(meshes, rho_core=rho_core)
        X = fd.SpatialCoordinate(layout.mechanics_mesh)
        z.subfunctions[layout.displacement].interpolate(X / fd.sqrt(fd.dot(X, X)))

        predicted = rho_core * RC ** 2 * 2 * np.pi * RC
        with_core = fd.assemble(solver.inertia_form(2))
        solver.fluid_core, saved = None, solver.fluid_core
        without = fd.assemble(solver.inertia_form(2))
        solver.fluid_core = saved

        assert with_core - without == pytest.approx(predicted, rel=1e-4)
        assert fd.assemble(solver.fluid_core_sheet_integral(
            solver.inertia_polynomial(2, X))) == pytest.approx(
                predicted, rel=1e-4)

    def test_the_sheet_reaches_the_monopole_datum(self, meshes):
        """The 2-D enclosed-mass datum must see the core's volume change.

        The 2-D exterior DtN needs a monopole datum carrying *all* the enclosed
        mass; the divergence-form volume source contributes identically zero
        (V4), so the datum is the sheets alone - and the CMB sheet is not in
        `sigma_bcs`, so `enclosed_mass_forms` has to add it by hand. Nothing
        else looks: the potential stays smooth, the geoid is merely wrong.

        Exercised where the term is *nonzero*, which needs saying. With the
        prototype's `cos 2 phi` load the net CMB sheet mass is 8.3e-16, because
        the load has no degree-0 content and the CMB does not breathe - so a
        test on a solved state would pass with the term missing. Here the
        displacement is set to `u = rhat` by hand, a pure degree-0 expansion,
        and the datum must then read `rho_core 2 pi Rc` (the `cos 2 phi` load
        sheet contributing nothing to a net mass).
        """
        rho_core = 2.0
        solver, z, layout = self.build_fluid(meshes, rho_core=rho_core)
        X = fd.SpatialCoordinate(layout.mechanics_mesh)
        z.subfunctions[layout.displacement].interpolate(X / fd.sqrt(fd.dot(X, X)))

        solver.update_total_mass()
        assert float(solver.source_mass) == pytest.approx(
            rho_core * 2 * np.pi * RC, rel=1e-4)

    def test_the_datum_is_unmoved_without_a_fluid_core(self, meshes):
        """The same state, rigid core: the datum sees the load sheet alone.

        The guard against the fix having changed the configuration everything
        else in this project was measured on.
        """
        solver, z, layout = build(meshes, rotation=False)
        X = fd.SpatialCoordinate(layout.mechanics_mesh)
        z.subfunctions[layout.displacement].interpolate(X / fd.sqrt(fd.dot(X, X)))
        solver.update_total_mass()
        assert abs(float(solver.source_mass)) < 1e-12

    def test_the_rotation_pair_transposes(self, meshes):
        """The centrifugal traction on the CMB, without which the pair is 97 % off.

        The fluid core's sheet enters `dI`, so it enters the `(m_i, u)` block;
        its transpose partner is a CMB traction proportional to `psi_rot`, and
        `rotational_potential_term` is a *volume* term with no such piece.
        Measured before `fluid_core_rotational_traction` existed:
        `max|A(u,m3) - A(m3,u)^T| = 9.07e-04` against a block maximum of
        9.32e-04, where the rigid core gives 2.6e-15.

        It went unseen because every transpose gate in the fluid core's suite
        ran with `rotation=False`. This test exists so that cannot recur.
        """
        parent, sub = meshes
        Z, layout = self_gravitating_gia_space(
            sub, parent, gravity_bcs=gravity_bcs(parent), rotation=True,
            self_gravity_number=LAMBDA)
        z = fd.Function(Z)
        Xm = fd.SpatialCoordinate(sub)
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        solver = SelfGravitatingGIASolver(
            z, approximation(), layout=layout, dt=1.0,
            bcs={CURVE_RE: {"normal_stress": B_MU * SIGMA_HAT * fd.cos(
                2 * fd.atan2(Xm[1], Xm[0]))}},
            rotation_moments={"C": fd.assemble(fd.dot(Xm, Xm) * dx_m)},
            fluid_core=FluidCore(boundary=CURVE_RC, rho_core=2.0))

        A = fd.assemble(fd.derivative(solver.F, z), mat_type="nest").petscmat
        iu, im = layout.displacement, layout.rotation["m3"]
        um = A.getNestSubMatrix(iu, im).convert("dense").getDenseArray()
        mu = A.getNestSubMatrix(im, iu).convert("dense").getDenseArray()
        scale = max(np.abs(um).max(), np.abs(mu).max())
        assert scale > 0.0
        assert np.abs(um - mu.T).max() / scale < 1e-14
