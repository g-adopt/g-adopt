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
    selfgrav_dtn_schur_solver_parameters,
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


def build(meshes, *, rotation=True, n_internal_variables=1, truncation=3,
          condensed=False, approximation_kwargs=None, declare_nullspace=False,
          **kwargs):
    parent, sub = meshes
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs(parent, truncation=truncation),
        rotation=rotation, n_internal_variables=n_internal_variables,
        condense_internal_variables=condensed,
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


class TestTheBlockOneDiagonalReachesTheAppctx:
    """`DtNMultiplierDiagPC` reads its diagonal from the appctx, so the appctx
    has to carry it on **every** way of specifying solver parameters.

    It did not. The entry was built only after the `Mapping` early return, and
    every 3-D driver passes a `Mapping` -- `b1_elastic.condensed_solver_
    parameters`, `b4_polar_motion.coupled_solver_parameters`, the B2 probe --
    so the shipped preconditioner was unreachable from all of them. Naming it
    gave a bare `PETSc.Error: error code 101`, PETSc having flattened the
    `ValueError`. Measured on Gadi at both coarse and medium (jobs 175339746,
    175340813, 175340815) before the fix; the one number on record for that
    preconditioner, 111 -> 48, came from the probe's own copy in `b2_pc.py`,
    which takes its diagonal from a module-level global instead.

    The first attempt at the fix ALSO failed, identically, and this class is
    written the way it is because of that: building the dictionary *before*
    calling the base class is silently undone, since
    `StokesSolverBase.set_solver_options` assigns `self.appctx = {"mu": ...}`
    as its first statement. So the assertion is on the state after
    construction, which is the only thing a preconditioner ever sees, rather
    than on any particular line having run.

    2-D here, which is the point: the defect is dimension-independent, and the
    configuration it broke costs a cluster job to reproduce.
    """

    @staticmethod
    def _diagonal(solver):
        return (solver.appctx or {}).get("dtn_block1_diagonal")

    @pytest.mark.parametrize("params", [
        None,
        "direct",
        "iterative",
        pytest.param(dict(selfgrav_dtn_schur_solver_parameters), id="mapping"),
    ])
    def test_every_way_of_asking_leaves_the_diagonal_in_the_appctx(
            self, meshes, params):
        solver, _, layout = build(meshes, solver_parameters=params)
        diag = self._diagonal(solver)
        assert diag is not None, (
            "no 'dtn_block1_diagonal' in the appctx, so "
            "gadopt.DtNMultiplierDiagPC cannot be selected on this path")
        assert len(diag) == len(layout.real_fields)
        assert np.all(diag != 0.0)

    def test_it_is_the_same_diagonal_however_it_was_asked_for(self, meshes):
        """A present-but-different diagonal would be worse than a missing one.

        The `Mapping` path reaches the appctx through two base classes; this
        pins that it arrives carrying the same numbers as the string path,
        rather than, say, a `theta_psi` recomputed somewhere along the way.
        """
        by_string = self._diagonal(build(meshes, solver_parameters="direct")[0])
        by_mapping = self._diagonal(build(
            meshes,
            solver_parameters=dict(selfgrav_dtn_schur_solver_parameters))[0])
        np.testing.assert_array_equal(by_string, by_mapping)

    def test_the_base_class_entry_survives(self, meshes):
        """The fix adds to the appctx; it must not replace it.

        `"mu"` is what `FreeSurfaceMassInvPC` and the assembled blocks read,
        and an earlier draft rebuilt the dictionary from scratch, which would
        have dropped whatever the base class decided that entry should be.
        """
        solver, _, _ = build(
            meshes,
            solver_parameters=dict(selfgrav_dtn_schur_solver_parameters))
        assert "mu" in solver.appctx


# ---------------------------------------------------------------------------
# The block-1 probe.
# ---------------------------------------------------------------------------
#: Per-row off-diagonal gate, `|A[j,i]| <= OFFDIAG_TOL * |A[j,j]|`. Measured
#: 0.0 in serial, and 0.0 at 2 and 3 ranks over five repeats each - it read
#: 2.5e-21 once, on 2026-08-03; see `SERIAL`. So the number only has to be
#: small against the planted term below; it is written per row rather
#: than against `max|diag|` because the diagonal spread is 236x in 2-D and 295x
#: in 3-D (`m_3` reads 6.748e-02 against 1.591e+01 for the outer multipliers),
#: and a global tolerance would therefore be 236x weaker on the one row that
#: matters least to the norm and most to the rotation.
OFFDIAG_TOL = 1e-12
#: `|d_k - A[k,k]| <= DIAG_TOL * |A[k,k]|`. Measured 2.7e-15 with rotation and
#: 2.0e-16 without, so this carries ~370x headroom over the worst case.
DIAG_TOL = 1e-12
#: Guards the bit-zero assertions, which are a *serial* claim. TSFC emits no
#: kernel for a term the form does not contain, so in serial the absent
#: couplings are bit zero and have always measured 0.0.
#:
#: **The parallel history is not stable and that is why this guard exists at
#: all.** On 2026-08-03 the same entries measured 2.541098841762901e-21 at 2
#: ranks and 1.694e-20 at 3; re-measured after the fact, five repeats at each of
#: 2 and 3 ranks, they are **exactly 0.0**. That was part of the intermittent
#: parallel fault in `NOTES/FD-ISSUE.md`, which nobody can now trigger. So the
#: honest statement is that parallel assembly of a `Real` row has been *seen* not
#: to reproduce bitwise, is not currently doing so, and no mechanism is known -
#: hence a guard rather than a number, and hence `assembly_floor` measuring the
#: floor instead of hard-coding one.
SERIAL = fd.COMM_WORLD.size == 1
#: The planted off-diagonal, in the units the block is assembled in. It is
#: 1.5e+04 times the per-row threshold on the *weakest* row (`m_3`, whose
#: threshold is 6.7e-14), and 1.5e-08 of that row's own diagonal - small enough
#: to be a genuine perturbation rather than a rewrite of the block.
PLANTED_EPS = 1e-9


def real_block_index_sets(Z, layout):
    """The block-1 index sets, taken the way `DtNTwoBlockSchurPC` takes them.

    `Z.dof_dset.field_ises[i_R:]`, i.e. Firedrake's own authority on where each
    sub-field sits in the monolithic row space, sliced at the first `Real`
    sub-field. `DtNTwoBlockSchurPC.initialize` merges exactly this slice into
    its "1" split, so a probe that reads through it exercises the
    preconditioner's own indexing rather than a second, parallel-untested one.
    """
    return Z.dof_dset.field_ises[layout.real_fields[0]:]


def read_real_rows(cofunction, ises, comm):
    """The `Real` entries of an assembled residual, gathered on every rank.

    **Not `.dat.data_ro[0]`.** `demos/gravity/CLAUDE.md` records that reading an
    `R` coefficient that way bypasses the ghosting and reduction machinery, and
    the whole point of this probe is that it must mean the same thing in
    parallel as in serial - the 3-D production gate runs on 48 ranks and a probe
    that silently read a stale or unowned entry would return plausible garbage
    rather than fail.

    So the access pattern is `_RealBlockPCBase._gather`'s, term for term:
    zero-fill a buffer of the global size, write only the entries this rank
    owns, and `Allreduce`. Each `Real` sub-field carries one global degree of
    freedom, so an index set here holds one index on its owner and none
    anywhere else, and the sum-reduction is exact rather than an average.

    Run on 2 ranks: the recovered block agrees with the serial one on every
    entry that any assertion here looks at, and the only difference the rank
    count makes is the one recorded at `SERIAL`.
    """
    buf = np.zeros(len(ises))
    with cofunction.dat.vec_ro as vec:
        lo, hi = vec.owner_range
        local = vec.array_r
        for position, iset in enumerate(ises):
            indices = iset.getIndices()
            assert len(indices) <= 1, (
                f"sub-field {position} of block 1 owns {len(indices)} degrees "
                "of freedom; a Real space has exactly one, globally")
            if len(indices) == 1:
                index = int(indices[0])
                assert lo <= index < hi, (
                    f"field_ises gave sub-field {position} the global index "
                    f"{index}, outside this rank's owner range [{lo}, {hi})")
                buf[position] = local[index - lo]
    out = np.zeros(len(ises))
    comm.Allreduce(buf, out)
    return out


def real_block(solver, z, layout, F=None):
    """The full `n x n` `Real` block of the Jacobian, by residual differencing.

    `F(e_j) - F(0)` read on the `Real` rows is the `j`th column exactly - the
    coupled residual is linear in every `Real` unknown, Newtonian rheology
    making the whole Jacobian constant - so this is a *measurement* of the
    block and not a finite difference with a step size to argue about. Measured
    against `mat_type="nest"` it agrees to 2.234e-16 relative.

    Differencing rather than nest assembly, and the reason is not stylistic:
    both shipped presets run `mat_type: matfree`, and asking for a nest
    Jacobian materialises every block including `(u,u)` and `(psi,psi)`, which
    is the assembled operator the configuration exists to avoid. At production
    size that is the most likely explanation for the 180 GB that killed the job
    this gate was first attempted in. Here it costs `n + 1` residual
    assemblies, and nothing else.
    """
    F = solver.F if F is None else F
    Z = z.function_space()
    comm = Z.mesh().comm
    ises = real_block_index_sets(Z, layout)
    real = layout.real_fields

    z.assign(0.0)
    base = read_real_rows(fd.assemble(F), ises, comm)
    block = np.zeros((len(real), len(real)))
    for column, field in enumerate(real):
        z.assign(0.0)
        z.subfunctions[field].assign(1.0)
        block[:, column] = read_real_rows(fd.assemble(F), ises, comm) - base
    z.assign(0.0)
    return block


def assembly_floor(solver, z, layout):
    """`max |F(0) - F(0)|` over the `Real` rows, from two identical assemblies.

    The reproducibility of the assembly itself, measured rather than assumed.

    **`F(0)` on the `Real` rows is not zero, and it has a plain cause.** It
    lives *entirely* on the rotation row **at the rank counts a laptop can
    reach**: measured at 1, 2, 3 and 4 ranks, every multiplier row is exactly
    0.0 and only the `m_3` row is not. That ceiling matters - at 48 ranks the
    multiplier block acquires a spurious coupling of 5.7199e-05 where serial
    reads 0.0 (a different quantity from this one, but the same class of claim),
    so "the multiplier rows are clean" must not be stated without a scale. `inertia_form`
    carries `sheet_integral(nu * sigma * p_g)`, and for this configuration's
    `cos 2phi` load `int cos(2 phi) r^2 ds` is **analytically zero** - the same
    quantity `test_inertia_of_a_cos2_sheet_vanishes` asserts vanishes. What
    survives is that integral's discrete remainder, scaled by `theta_rot`. So
    the residual is not structurally zero; it is one physically-zero integral
    failing to be discretely zero, in the one row carrying a prescribed sheet,
    and it cancels in the differencing.

    **Its magnitude is not stable across partitions and is deliberately not
    recorded here.** A sum of cancelling terms lands wherever the summation
    order puts it, and the order changes with the facet partition: readings at
    the 1e-20 level have differed between sessions and between machines for this
    quantity, and pinning one would invite a future reader to treat a difference
    as a finding. That is why the floor is *measured* rather than hard-coded -
    and why a structural `== 0.0` on `F(0)` would be wrong at every rank count.
    It is also the right shape for the production gate, which runs at 48 ranks
    where nobody has measured anything.
    """
    z.assign(0.0)
    ises = real_block_index_sets(z.function_space(), layout)
    comm = z.function_space().mesh().comm
    first = read_real_rows(fd.assemble(solver.F), ises, comm)
    second = read_real_rows(fd.assemble(solver.F), ises, comm)
    return float(np.abs(first - second).max())


def check_multiplier_alignment(solver, z, layout, *, silent_tol=1e-6,
                               predicted_tol=1e-12):
    """Drive `psi` with each mode expression; assert the rows its key names fire.

    Shared by the 2-D annulus and the 3-D shell, because **the claim is the
    same one and 3-D is where it carries the whole weight.** `SphericalDtN._mode`
    hands every `(l, m)` the same `scale = 1/(4 pi)`, so the multiplier diagonal
    is one value per boundary at *any* truncation - 2 distinct among 8 at L = 1,
    2 among 32 at L = 3, 2 among 72 at L = 5. It does not improve with
    truncation. 2-D is better only by accident: the interior `mean` mode's
    `scale = 1.0` against 0.5 for the azimuthal ones gives a third value. So in
    3-D the diagonal identity constrains exactly two numbers and a partition
    into two boundaries, and *everything else about the ordering is this
    function*.

    Each constraint row is `(psi e_k - scale_k c_k) mu_k ds`, so with `c = 0`
    the row reads `theta_psi * int psi_h e_k ds` over its own boundary, and the
    angular functions are orthogonal there - real spherical harmonics on the
    sphere exactly as `cos/sin(m phi)` on the circle. Driving `psi` with one
    mode expression must therefore move exactly the rows whose key is that mode,
    on *both* boundaries since the expression is global, and which row belongs
    to which boundary follows from the magnitude.

    **The prediction is the discrete integral, not the analytic norm**, and that
    is the whole reason this gates at 1e-12. Against `theta_psi * norm` the
    fired rows deviate by 1.2420e-04 at truncation 3 and 9.4322e-04 at
    truncation 5 - it **grows 7.6x** over two orders of truncation, because it
    is the boundary's resolution of the mode and nothing to do with alignment,
    and a gate set from it would fail on a finer truncation, a coarser mesh or
    the sphere for a reason that is not a defect. Against
    `theta_psi * int psi_h e_k ds` through the form's own degree-pinned `ds` it
    is 2.0447e-16 and 3.3537e-16 - exact and truncation-independent. That
    analytic deviation is worth recording, but it is
    `check_boundary_quadrature`'s measurement, not this one's.

    ## The two halves are not equally strong, and in 3-D only one of them is

    The **discrete-prediction** half is the alignment gate and it is exact
    everywhere measured - `5.810e-16` at `(L=1, ref=1)`, `7.747e-16` at
    `(L=3, ref=1)`, `1.741e-15` at `(L=3, ref=2)`, `5.606e-15` at `(L=5, ref=3)`
    with 72 multipliers, and 2.0447e-16 / 3.3537e-16 on the 2-D annulus at
    truncations 3 and 5. It needs no orthogonality: if a sub-field claiming key
    `A` were really key `B`, the row would read `int e_A e_B ds` against a
    prediction of `int e_A^2 ds`, and a boundary swap changes the magnitude by
    `R^2`. Both are caught whether or not the mesh resolves anything.

    The **silent-row** half is a statement about discrete orthogonality and is
    therefore a property of the mesh, not of the code:

        2-D annulus, truncation 3 and 5      4.0804e-12  (identical to 5 digits)
        3-D  L=1  ref=1   24 cells           2.905e-16
        3-D  L=3  ref=1   24 cells           2.724e-02   <- unusable
        3-D  L=3  ref=2   96 cells           7.902e-04
        3-D  L=3  ref=3  384 cells           3.870e-05
        3-D  L=5  ref=3  384 cells           8.132e-04

    A level-1 cubed sphere does not resolve degree 3 at all -
    `L h_max/R = 2.17`, and `DtNGravityForm` caps the boundary rule and warns -
    so `Y1,-1` moves the `Y3,-3` row by 4.264e-03 against a fired magnitude of
    1.146. That is not an alignment failure and gating it at 1e-06 would be a
    test of the mesh. Hence `silent_tol` is a parameter, set per configuration
    from the table above, and the gate that carries the claim is
    `predicted_tol`.
    """
    Z = z.function_space()
    comm = Z.mesh().comm
    ises = real_block_index_sets(Z, layout)
    form = solver.form
    theta = float(solver.theta_psi)
    psi_h = z.subfunctions[layout.potential]

    modes = {}
    for bc_id, dtn in form.dtn_boundaries:
        side, radius = form.boundary_geometry[bc_id]
        for mode in dtn.modes(side, radius, form.X):
            modes[(bc_id, mode.key)] = mode

    z.assign(0.0)
    base = read_real_rows(fd.assemble(solver.F), ises, comm)
    keys = form.multiplier_keys
    assert len(set(key for _, key in keys)) > 1, "nothing to align"

    worst_discrete = 0.0
    worst_analytic = 0.0
    worst_silent = 0.0
    for driven in sorted({key for _, key in keys}):
        expression = next(mode.expr for (_, key), mode in modes.items()
                          if key == driven)
        z.assign(0.0)
        psi_h.interpolate(fd.Constant(expression)
                          if isinstance(expression, float) else expression)
        row = read_real_rows(fd.assemble(solver.F), ises, comm) - base

        fired = [i for i, (_, key) in enumerate(keys) if key == driven]
        silent = [i for i, (_, key) in enumerate(keys) if key != driven]
        assert fired, driven
        for i in fired:
            bc_id, key = keys[i]
            mode = modes[(bc_id, key)]
            discrete = theta * fd.assemble(psi_h * mode.expr * form.ds(bc_id))
            assert row[i] == pytest.approx(discrete, rel=predicted_tol), (
                f"sub-field {layout.multipliers[i]} claims to be {keys[i]} but "
                "does not respond to it as one")
            worst_discrete = max(
                worst_discrete, abs(row[i] - discrete) / abs(discrete))
            analytic = theta * float(mode.norm)
            worst_analytic = max(
                worst_analytic, abs(row[i] - analytic) / abs(analytic))
        weakest = min(abs(row[i]) for i in fired)
        for i in silent:
            assert abs(row[i]) < silent_tol * weakest, (
                f"driving {driven} moved sub-field {layout.multipliers[i]}, "
                f"which claims to be {keys[i]}")
            worst_silent = max(worst_silent, abs(row[i]) / weakest)
        # And the rotation rows never see the potential at all.
        for i in range(len(keys), len(layout.real_fields)):
            assert abs(row[i]) < silent_tol * weakest
    z.assign(0.0)
    return {"discrete": worst_discrete, "analytic": worst_analytic,
            "silent": worst_silent}


def offdiagonal_ratios(block):
    """Per row, `max_j |A[i,j]| / |A[i,i]|` over `j != i`."""
    off = block.copy()
    np.fill_diagonal(off, 0.0)
    return np.abs(off).max(axis=1) / np.abs(np.diag(block))


def block_asymmetry(F, z):
    """Per-block `max|A_ij - A_ji^T|`, and the max-abs grid beside it.

    A hand copy of `demos/gravity/selfgrav_gia_annulus.py`, deliberately, and
    not an import: that module imports `generate_selfgrav_annulus` and
    `validate_selfgrav_annulus` at module scope, which pull in `gmsh` and
    `scipy`, so importing it at the top of this file would hard-fail collection
    of the whole module on a machine without them instead of skipping the
    tests that need a mesh.

    `mat_type="nest"` with `getNestSubMatrix` is spike S5's route B: it
    assembles with `Real` blocks present, where monolithic `aij` does not, it is
    sharp to ~1e-17, and it names the block. It needs a global dense transpose,
    so it is serial.
    """
    Z = z.function_space()
    A = fd.assemble(fd.derivative(F, z), mat_type="nest").petscmat
    n = len(Z)
    diff = np.zeros((n, n))
    size = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Mij = A.getNestSubMatrix(i, j)
            Mji = A.getNestSubMatrix(j, i)
            if Mij is None or Mji is None:
                continue
            aij = Mij.convert("dense").getDenseArray()
            aji = Mji.convert("dense").getDenseArray()
            diff[i, j] = np.abs(aij - aji.T).max()
            size[i, j] = np.abs(aij).max()
    return diff, size


class TestBlockZeroSymmetry:
    """The condensed `(u, psi)` operator is symmetric, and the uncondensed is not.

    Condensing the internal variables is what makes block 0 a symmetric
    operator: the internal-variable rows are the asymmetric ones, and the
    driver's own instrument reads `diff[0,1] = 4.768e-01` against
    `size[0,1] = 3.576e-01` on the uncondensed annulus, a ratio of 1.333, i.e.
    the `(u, sigma)` pair is not even close to transposing. That is not a defect
    - `CoupledInternalVariableSolver`'s free-slip Nitsche terms carry `mu`
    against `mu0`, recorded in `NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md` -
    but it does mean the symmetry statement belongs to the condensed system
    alone.

    The `3.2e-02` quoted at `demos/gravity/selfgrav_gia_annulus.py:66` is a
    different measurement: the whole matrix on a 4x4 Cartesian square, not the
    annulus and not block 0.

    Serial, like every per-block measurement in this project: `block_asymmetry`
    builds a global dense transpose. The block-1 probe below has no such
    restriction and runs under MPI.
    """

    pytestmark = pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs a global dense transpose")

    def test_the_condensed_block_is_symmetric(self, meshes):
        """Relative, at 1e-12: measured 1.056e-16, so ~1e+04 of headroom.

        Absolute would be meaningless - `max|A| = 1.346e+02` here and scales
        with the mesh, the load and every material constant.
        """
        solver, z, layout = build(meshes, condensed=True)
        block0 = [i for i in range(len(z.function_space()))
                  if i not in layout.real_fields]
        assert len(block0) == 2, (
            "condensation should leave displacement and potential alone")
        diff, size = block_asymmetry(solver.F, z)
        window = np.ix_(block0, block0)
        biggest = size[window].max()
        assert biggest > 0.0
        assert diff[window].max() <= 1e-12 * biggest

    def test_the_uncondensed_block_is_not(self, meshes):
        """The rejecting partner for the test above, and only that.

        Measured **2.528e-03** relative here against `1.056e-16` condensed:
        thirteen orders apart, which is what makes the condensed `1e-12` a
        statement about condensation rather than about a loose tolerance.

        **One-sided on purpose.** An upper bound would turn a symptom of the
        separately-recorded `mu`/`mu0` Nitsche asymmetry
        (`NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md`) into a second gate on it,
        living in a file about block 1 - and that is not hypothetical:
        `gate_g0`'s docstring records the analogous `(u,u)` asymmetry being
        fixed at source once already, on 2026-08-01.

        The bound is `> 1e-6` and not `> 1e-3`, because **the quantity is
        `dt`-dependent**: `gate_g0` measured the related asymmetry tracking
        `dt/(tau + dt)` linearly over four decades, and `build` fixes `dt = 1.0`.
        Whoever first parametrises `dt` will walk this number down, and at small
        `dt` even this bound will eventually fail - correctly, and the fix then
        is to delete this test rather than to lower it again. If the Nitsche
        asymmetry is fixed at source, likewise: delete, do not loosen. Its only
        job is to keep the test above from being vacuous.
        """
        solver, z, layout = build(meshes, condensed=False)
        block0 = [i for i in range(len(z.function_space()))
                  if i not in layout.real_fields]
        assert len(block0) == 3
        diff, size = block_asymmetry(solver.F, z)
        window = np.ix_(block0, block0)
        assert diff[window].max() / size[window].max() > 1e-6


class TestBlockOneIsTheDiagonalItClaims:
    """`block1_diagonal()` against the assembled `Real` block, entry by entry.

    Road map §9.6 lists `B_mu` as pinned by nothing and the block-1
    preconditioner as unmeasured; this closes the second of those for the
    *diagonal*, which is what `DtNMultiplierDiagPC` divides by. The whole block
    is recovered, not just its diagonal, because a claim that a matrix "is" a
    diagonal is two claims and the second one - that nothing else is there - is
    the one nobody had checked.

    Three things are asserted and they are not equally strong:

    1. **The diagonal.** `theta_psi * (-scale_k * A_h)` per multiplier and
       `theta_rot_i * K_i` per rotation row, against the assembled entry.
       Measured 2.673e-15 relative with rotation, 2.045e-16 without.

       **The rotation row is already the worst row here, by an order**
       (2.664535e-15 against 2.0447e-16 for the multipliers), and that is not
       noise: `rotation_residual` divides by a separately assembled mantle
       volume, so its diagonal carries the ratio of two assemblies of the same
       integral under different quadrature rules. At production size the ratio
       is much further from 1 - `gate_phase1_diagonal_3d.py` measures
       **8.8775e-11** on the coarse Spada mesh, against 1.2735e-15 for the 72
       multiplier rows there - so **`DIAG_TOL` at 1e-12 holds on these meshes
       and does not hold at scale.** Read that gate's `ROT_DIAG_TOL` before
       concluding anything about the rotation rows from this file.
    2. **The off-diagonals.** In serial, exactly 0.0, bit zero, not 1e-16 -
       TSFC emits no kernel for a term the form does not contain. That makes
       the assertion vacuous on its own, which is why (3) exists. On 2 ranks it
       is 2.5e-21 instead, for the reason recorded at `SERIAL`.
    3. **The alignment**, which the diagonal cannot see at all. `CylindricalDtN`
       gives `scale = 0.5` to every azimuthal mode and 1.0 to the interior mean,
       so truncation 3 has **4 distinct values among 14 rows**; `SphericalDtN`
       gives `1/(4 pi)` to every mode, so 3-D at L = 5 has **2 among 72**.
       Permuting the modes within a boundary therefore leaves the diagonal
       bit-identical. It does not leave `coefficients()` alone: that method
       (`gia_gravity.py:1756`) zips `form.multiplier_keys` against
       `layout.multipliers` to attribute each solved trace coefficient to a
       mode, so a shuffle gives a silently wrong geoid spectrum with a
       perfectly plausible magnitude.

    What none of it establishes is recorded in
    `NOTES/measurements/PHASE1-DIAGONAL-GATE.md`; the short version is that a
    flip of `CLOSURE_SIGNS` is invisible here, because `block1_diagonal` builds
    the rotation entry as `_theta_rot(k) * _closure_constant(k)` and both
    factors carry `s_i`, so a flip moves the prediction and the assembled row
    together.
    """

    @pytest.mark.parametrize("rotation", [True, False])
    @pytest.mark.parametrize("truncation", [3, 5])
    def test_the_block_is_the_claimed_diagonal(self, meshes, rotation,
                                               truncation):
        """Truncation 5 is production's `L`; 3 is what the rest of this file uses.

        Both are carried because the multiplier count is the one thing that
        changes with it - 14 rows at truncation 3, 22 at 5 - and a diagonal
        built by zipping two sequences is exactly the kind of thing that is
        right at one length and wrong at another.
        """
        solver, z, layout = build(meshes, rotation=rotation,
                                  truncation=truncation, condensed=True)
        block = real_block(solver, z, layout)
        claimed = solver.block1_diagonal()

        assert len(claimed) == len(layout.real_fields)
        assert len(claimed) == (2 * truncation) + (2 * truncation + 1) + (
            1 if rotation else 0)
        assert np.all(np.abs(np.diag(block)) > 0.0), (
            "a zero diagonal entry would make the gate vacuous and the "
            "preconditioner a division by zero")

        assert np.abs(claimed - np.diag(block)).max() <= DIAG_TOL * np.abs(
            np.diag(block)).max()
        assert np.max(np.abs(claimed - np.diag(block))
                      / np.abs(np.diag(block))) <= DIAG_TOL
        assert offdiagonal_ratios(block).max() <= OFFDIAG_TOL

    @pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="planting a Real x Real term in this multi-mesh form "
               "over-counts under MPI; see the docstring")
    def test_a_planted_coupling_is_recovered(self, meshes):
        """The rejecting partner, and the only reason (2) above means anything.

        Two genuine couplings are added to the residual - one multiplier to
        another, and one multiplier to the rotation row - each written as
        `eps * (c_i / vol) * mu_j * dx`, so that the entry it plants is `eps`
        exactly. The probe must read both back, and the gate must reject the
        result.

        Measured: `A[1,0] = A[13,0] = 1.000000e-09` at `eps = 1e-9`, against
        `0.0` for both entries on the shipped residual, so what is recovered is
        the plant and not the probe's own noise.

        ## Serial, because of a real MPI defect in the shipped form

        Under MPI this test recovered 2.000e-09 on the multiplier row and
        4.000e-09 on the rotation row at 2 ranks. On a laptop the effect is
        intermittent and has not been reproducible since; **at 48 ranks on Gadi
        it is deterministic** (job 175368128) and it is not confined to the
        plant. `NOTES/FD-ISSUE.md` is the record.

        The rule is a **pairing**: a `Real` test function integrated over a
        *different mesh* from the one its space was built on is over-counted by
        approximately the rank count, while a matched pairing is correct in the
        same run. Measured at 48 ranks - multiplier rows, which are parent
        `Real` tests on parent facets, correct to 2.6677e-16; rotation rows,
        which integrate a parent `Real` test over the submesh, over-counted by
        48.000025, 47.999962 and 47.997255.

        **So the shipped rotation row is affected.** An earlier version of this
        docstring said it was clean; that rested on 2-4 ranks and is refuted.
        Two withdrawn accounts, recorded so nobody re-derives them: that the
        trigger is a `Real x Real` term, and that it is a `Real` row mixing a
        cell with a facet integral. No static guard encoding either should be
        added.

        **2-D cannot see the consequence that matters**, which is why this file
        keeps passing. The rotation row's volume terms are mismatched and its
        sheet term is matched, so the closure solves
        `K m = s (dI_vol + dI_sheet/n)` - and with this configuration's
        `cos 2phi` load `dI_sheet` is analytically zero
        (`test_inertia_of_a_cos2_sheet_vanishes`), so the diluted term vanishes
        identically whatever `n` is. Only the diagonal over-counts. In 3-D the
        cap load's sheet contribution is not zero and the solved rotation vector
        is rank-dependent.

        The skip stays: a rank-dependent failure is the worst thing to have in
        CI, and what this test guards is a synthetic plant rather than the
        shipped behaviour, which is tracked in `FD-ISSUE.md` instead. Note also
        that "the production residual has no `Real x Real` term" is false -
        `dtn_form.py:959` and `gia_gravity.py:2121` are both such terms, and
        they are the diagonal this class gates.
        """
        solver, z, layout = build(meshes, rotation=True, condensed=True)
        Z = z.function_space()
        real = layout.real_fields
        n_mult = len(layout.multipliers)
        dx_parent = fd.dx(domain=solver.potential_mesh)
        volume = fd.assemble(fd.Constant(1.0) * dx_parent)

        trials, tests = fd.split(z), fd.TestFunctions(Z)
        source = trials[real[0]] / volume
        planted = (solver.F
                   + PLANTED_EPS * source * tests[real[1]] * dx_parent
                   + PLANTED_EPS * source * tests[real[n_mult]] * dx_parent)

        clean = real_block(solver, z, layout)
        assert np.all(np.abs(np.diag(clean)) > 0.0), (
            "a zero diagonal would make offdiagonal_ratios inf or nan rather "
            "than fail, so it is checked before the ratio is taken")
        assert offdiagonal_ratios(clean).max() <= OFFDIAG_TOL
        # Against the assembly's own reproducibility, measured here rather than
        # asserted as bit zero: `F(0)` is 3.388e-21 on these rows even in
        # serial, it merely reproduces, and in parallel it does not.
        floor = assembly_floor(solver, z, layout)
        assert abs(clean[1, 0]) <= floor and abs(clean[n_mult, 0]) <= floor
        assert floor < 1e-3 * PLANTED_EPS, (
            f"the assembly floor {floor:.3e} is not comfortably below the "
            f"plant {PLANTED_EPS:.3e}, so the partner proves nothing")

        block = real_block(solver, z, layout, F=planted)
        assert block[1, 0] == pytest.approx(PLANTED_EPS, rel=1e-9)
        assert block[n_mult, 0] == pytest.approx(PLANTED_EPS, rel=1e-9)
        # ... and the gate says so, on the multiplier row and on the rotation
        # row separately: the second is the weakest row in the block and the
        # one a global tolerance would let through most easily.
        ratios = offdiagonal_ratios(block)
        assert ratios[1] > OFFDIAG_TOL
        assert ratios[n_mult] > OFFDIAG_TOL
        # The diagonal is untouched by the plant, so the claim stays true and
        # the two assertions are independent.
        assert np.max(np.abs(solver.block1_diagonal() - np.diag(block))
                      / np.abs(np.diag(block))) <= DIAG_TOL

    def test_condensation_does_not_move_the_block(self, meshes):
        """Measured identical, so it is asserted rather than parametrised over.

        Condensing eliminates the internal variables from block 0 and touches
        no constraint row, so the `Real` block is the same matrix either way -
        `max rel |d - diag(J)| = 4.113e-16` on both. Carrying every other test
        in this class over both settings would double the cost to re-measure a
        constant.
        """
        condensed = real_block(*build(meshes, rotation=True, condensed=True))
        uncondensed = real_block(*build(meshes, rotation=True, condensed=False))
        assert np.abs(condensed - uncondensed).max() <= 1e-14 * np.abs(
            condensed).max()

    def test_the_block_one_index_sets_are_what_the_split_assumes(self, meshes):
        """What `DtNTwoBlockSchurPC.initialize` merges, asserted here instead.

        That method builds its "1" split as
        `merge(field_ises[i_R:i_R + n])` and its "0" split as
        `merge(field_ises[:i_R])`, and nothing in it checks that the two
        partition the row space or that the merged block-1 set is ordered - it
        checks the *sub-field* indices and trusts Firedrake for the rest. A
        merged set that was out of order, or that overlapped block 0, would be
        a silently wrong split, which is the failure mode this whole file
        exists for.

        Runs under MPI, where it is not free: every entry here is rank-local,
        and on 2 ranks rank 0 owns all 14 while rank 1 owns none.
        """
        _, z, layout = build(meshes, rotation=True)
        Z = z.function_space()
        comm = Z.mesh().comm
        i_R = layout.real_fields[0]
        block1 = real_block_index_sets(Z, layout)
        block0 = Z.dof_dset.field_ises[:i_R]

        assert len(block1) == len(layout.real_fields)
        merged = np.concatenate([iset.getIndices() for iset in block1]) \
            if block1 else np.array([], dtype=int)
        assert np.all(np.diff(merged) > 0), (
            "the merged block-1 index set is not increasing, so it is not the "
            "in-order concatenation the matrix-free submatrix extraction needs")
        # One global degree of freedom per Real sub-field, summed over ranks.
        assert comm.allreduce(len(merged)) == len(layout.real_fields)

        owned0 = np.concatenate([iset.getIndices() for iset in block0])
        assert not np.intersect1d(owned0, merged).size, (
            "a degree of freedom is in both blocks")
        # Contiguous and trailing, locally: block 1 owns the top of this rank's
        # range or nothing at all.
        if merged.size:
            assert merged.max() > owned0.max()
            assert merged.size == merged.max() - merged.min() + 1

    def test_the_multiplier_keys_are_the_boundary_and_mode_order(self, meshes):
        """The cheap half of the alignment claim: the key *sequence*.

        `boundary_bilinear` consumes the `(trial, test)` pairs in
        `dtn_boundaries` order and, within a boundary, in the order the
        descriptor's `modes` returns, and records that pairing in
        `multiplier_keys`.

        **The sequence is written out as a literal, deliberately.** An earlier
        version built the expectation by iterating `dtn_boundaries` and
        `dtn.modes(...)` - which is verbatim the loop `boundary_bilinear` uses
        (`dtn_form.py:946-955`), so it re-executed the implementation and
        compared it with itself. That catches a post-hoc mutation of
        `multiplier_keys` and cannot catch a reordering of either loop, because
        the expectation reorders with it. For an *ordering*, a literal is a
        specification and not a golden number, so it does not fall foul of this
        file's rule against stored values: exterior boundary first, `cos` before
        `sin` within an order, orders ascending, then the interior boundary with
        its `mean` mode leading.
        """
        solver, _, layout = build(meshes, rotation=True)
        assert solver.form.multiplier_keys == [
            (CURVE_OUTER, "cos1"), (CURVE_OUTER, "sin1"),
            (CURVE_OUTER, "cos2"), (CURVE_OUTER, "sin2"),
            (CURVE_OUTER, "cos3"), (CURVE_OUTER, "sin3"),
            (CURVE_INNER, "mean"),
            (CURVE_INNER, "cos1"), (CURVE_INNER, "sin1"),
            (CURVE_INNER, "cos2"), (CURVE_INNER, "sin2"),
            (CURVE_INNER, "cos3"), (CURVE_INNER, "sin3"),
        ]
        assert len(solver.form.multiplier_keys) == len(layout.multipliers)

    @pytest.mark.parametrize("truncation", [3, 5])
    def test_each_multiplier_row_is_the_mode_its_key_names(self, meshes,
                                                           truncation):
        """The alignment assertion proper, and the one the diagonal cannot make.

        `check_multiplier_alignment` carries the argument and the measurements.
        Both truncations, because that is the axis a zip-length error lives on,
        because 5 is production's `L`, and because the analytic deviation this
        test deliberately does *not* gate on grows 7.6x between them - which is
        how the previous version of this test came to sit 6% inside its own
        threshold at truncation 5 without anyone noticing.
        """
        solver, z, layout = build(meshes, rotation=True, condensed=True,
                                  truncation=truncation)
        worst = check_multiplier_alignment(solver, z, layout)
        # Recorded, not gated: this is the boundary's resolution of the mode.
        assert worst["analytic"] < 1e-2, (
            "the analytic deviation has grown far beyond the 1.2e-04 and "
            "9.4e-04 measured at truncations 3 and 5; that is a statement "
            "about the boundary quadrature, not about alignment, but at this "
            "size it is worth knowing")


class TestBlockOneInThreeDimensions:
    """The same probe on a 3-D shell, because 2-D cannot see the polar pair.

    A disc has only `m_3`, so every rotation statement the 2-D gate makes is
    about one row. The `(m_1, m_2)` block - whether the two polar-wander
    components couple to each other, and whether they couple to the
    multipliers - has no 2-D analogue at all, and it is the one that carries
    the shared closure constant `C - A`.

    **Assembly only. There is no solve here and there must not be one**: this
    file's standing rule sends every 3-D solve to Gadi, and the arithmetic
    behind it is that the coarse elastic snapshot takes 674 s on 64 ranks. One
    residual evaluation per `Real` sub-field is twelve of them on 24 cells, and
    costs seconds.

    Cheap by construction: `CubedSphereMesh` extruded radially, so no gmsh and
    no mesh file, and the same mesh passed twice to `self_gravitating_gia_space`
    rather than a `Submesh`. That makes the mechanics and the potential share a
    domain, which the production geometry does not - the stand-off buffer is the
    whole point there - but nothing this class asserts is about the buffer.

    It warns, and the warning is expected: `L h_max/R = 0.72` on a level-1
    cubed sphere is too coarse for the boundary rule to resolve `L = 1`, so
    `DtNGravityForm` caps the quadrature degree and says so. Nothing here is a
    statement about the boundary integrals' accuracy - the diagonal is
    `-scale_k A_h` with `A_h` the *discrete* boundary measure, whatever that
    measure happens to be on this mesh.
    """

    @staticmethod
    def build(L=1, refinement_level=1):
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
            rotation=True, condense_internal_variables=True,
            self_gravity_number=LAMBDA)
        z = fd.Function(Z)
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        solver = SelfGravitatingGIASolver(
            z, approximation(), layout=layout, dt=1.0,
            bcs={"bottom": {"un": 0.0},
                 "top": {"normal_stress": SIGMA_HAT * X[2]}},
            # Both moments, and it has to be both: `_closure_constant` reads
            # `C_minus_A` for the polar-wander pair and `C` for `m_3`, and asks
            # for whichever is missing by name. 0.1 C is not a physical
            # dynamical ellipticity - it is a number that makes the two closure
            # constants distinguishable in the diagonal, which is the point.
            rotation_moments={"C": C, "C_minus_A": 0.1 * C})
        return solver, z, layout

    def test_the_case_is_the_small_one(self):
        """A tripwire on the cost, not a statement about the discretisation.

        24 cells and 1971 degrees of freedom. If a future edit makes this a
        refinement level larger it stops being a unit test, and the rule this
        class is written under stops holding.

        The cell count is summed over `cell_set.size`, the *owned* count, and
        not `num_cells()`, which is rank-local and includes the halo: on 2 ranks
        it reads 20 on rank 0, so the obvious spelling is a test that passes
        only in serial.
        """
        solver, z, layout = self.build()
        assert z.function_space().dim() == 1971
        assert solver.mesh.comm.allreduce(solver.mesh.cell_set.size) == 24
        assert len(layout.real_fields) == 11
        # (L+1)^2 = 4 modes on each of two boundaries, then m_1, m_2, m_3.
        assert len(layout.multipliers) == 8
        assert set(layout.rotation) == {"m1", "m2", "m3"}

    def test_the_block_is_the_claimed_diagonal(self):
        """Measured `max rel |d - diag(A)| = 1.787e-15`; gate at 1e-12.

        `SphericalDtN._mode` gives `scale = 1/(4 pi)` to **every** mode, so
        unlike 2-D there is no per-mode variation at all: the eight multiplier
        entries here take 2 distinct values, one per boundary radius
        (`-4.5889137914` and `-1.1472284479`), and at production `L = 5` it is 2
        distinct values among 72 rows. The diagonal is even blinder to a mode
        shuffle in 3-D than in 2-D, which is the reason
        `TestBlockOneIsTheDiagonalItClaims` pins the alignment separately.

        Spread over the whole block, rotation included: 295.5, against 235.7 in
        2-D. That is what the per-row off-diagonal form is for.
        """
        solver, z, layout = self.build()
        block = real_block(solver, z, layout)
        claimed = solver.block1_diagonal()
        assert np.max(np.abs(claimed - np.diag(block))
                      / np.abs(np.diag(block))) <= DIAG_TOL
        assert offdiagonal_ratios(block).max() <= OFFDIAG_TOL
        # As in 2-D: bit zero, because the form contains no such term.
        off = block.copy()
        np.fill_diagonal(off, 0.0)
        if SERIAL:
            assert np.abs(off).max() == 0.0
        # Two distinct multiplier values, one per boundary, from 8 rows.
        n_mult = len(layout.multipliers)
        assert len(set(np.round(np.diag(block)[:n_mult], 12))) == 2

    @pytest.mark.parametrize("L, refinement, silent_tol",
                             [(1, 1, 1e-6), (3, 2, 1e-2)])
    def test_each_multiplier_row_is_the_mode_its_key_names(self, L, refinement,
                                                           silent_tol):
        """**In 3-D this is the entire content of the ordering claim.**

        `SphericalDtN._mode` gives every `(l, m)` the same `scale = 1/(4 pi)`
        and there is no monopole special case - the class docstring says so and
        gives the reason - so `multiplier_diagonal` is `-scale * A_h`, one value
        per boundary, at any truncation. Measured here: 8 multipliers with **2
        distinct values** at `L = 1`, 32 multipliers with the **same 2** at
        `L = 3`, and it is 2 among 72 at production's `L = 5`. It does not
        improve with truncation.

        So 3-D is strictly worse than 2-D, not merely bigger. In 2-D the
        interior `mean` mode carries `scale = 1.0` against 0.5 for the azimuthal
        ones, which at least separates `(5, 'mean')` from its neighbours; here
        every mode on a boundary is interchangeable as far as the diagonal is
        concerned. The diagonal identity constrains two numbers and a partition
        into two boundaries. Everything else about which sub-field is which mode
        is asserted here and nowhere else.

        Both truncations, because 8 multipliers against 32 exercises a different
        length in the same zip for a few seconds.

        `L = 3` runs on a level-2 base and not a level-1 one, and the reason is
        a measurement rather than caution: at level 1 the shell does not resolve
        degree 3 (`L h_max/R = 2.17`), and `Y1,-1` moves the `Y3,-3` row by
        4.264e-03 against a fired magnitude of 1.146. The gate that carries the
        claim - the discrete prediction - is 7.747e-16 there regardless, so the
        alignment is fine and it is discrete orthogonality that is not.
        `silent_tol` is set per row of `check_multiplier_alignment`'s table.
        """
        solver, z, layout = self.build(L=L, refinement_level=refinement)
        assert len(layout.multipliers) == 2 * (L + 1) ** 2
        block = real_block(solver, z, layout)
        n_mult = len(layout.multipliers)
        assert len(set(np.round(np.diag(block)[:n_mult], 12))) == 2, (
            "the premise of this test: the diagonal cannot tell the modes "
            "apart, so if it suddenly can, read this docstring again")
        check_multiplier_alignment(solver, z, layout, silent_tol=silent_tol)

    def test_the_gate_scripts_copy_of_the_probe_agrees_with_this_one(self):
        """The duplication in `gate_phase1_diagonal_3d.py`, closed by measurement.

        That script reimplements `read_real_rows` and `real_block` rather than
        importing them, because a demo reaching into `tests/` is the wrong
        dependency direction and would make the production gate depend on the
        test tree being importable on Gadi. The direction that *is* allowed is
        this one, and without it "the two are pinned to agree" means somebody
        read two numbers off two screens.

        The 24-cell case, so it costs a second, and it is the same case the
        script's own `--selfcheck` runs.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                               / "demos" / "gravity" / "spikes"))
        import gate_phase1_diagonal_3d as gate

        solver, z, layout = self.build()
        mine = real_block(solver, z, layout)
        theirs = gate.real_block(solver, z, layout)
        assert np.abs(mine - theirs).max() == 0.0, (
            "the gate script's copy of the probe no longer agrees with this "
            "file's; they have drifted and the production gate is measuring "
            "something else")
        # And the tolerances it gates on are this file's.
        assert gate.DIAG_TOL == DIAG_TOL
        assert gate.OFFDIAG_TOL == OFFDIAG_TOL

    def test_the_polar_pair_is_uncoupled_and_carries_C_minus_A(self):
        """The one thing 2-D cannot say, in three parts.

        `m_1` and `m_2` do not couple to each other, neither couples to the
        multipliers, and both carry `C - A` while `m_3` carries `C` with the
        opposite closure sign. Measured rotation diagonal
        `[+0.01552799, +0.01552799, -0.15527988]`: the pair equal, `m_3` of the
        opposite sign, and ten times the magnitude because `C_minus_A` was set
        to `0.1 C` above. The sign is `s_3 = -1` showing through
        `theta_rot_3 K_3`, both factors of which carry it - so this reads the
        product and **cannot** see a flip of `CLOSURE_SIGNS`, which moves the
        prediction and the assembled row together.
        """
        solver, z, layout = self.build()
        block = real_block(solver, z, layout)
        n_mult = len(layout.multipliers)
        rotation = np.diag(block)[n_mult:]

        assert rotation[0] == pytest.approx(rotation[1], rel=1e-14)
        assert rotation[0] > 0.0 and rotation[2] < 0.0
        assert rotation[2] / rotation[0] == pytest.approx(-10.0, rel=1e-12)
        assert rotation[0] == pytest.approx(+0.01552799, rel=1e-6)
        assert rotation[2] == pytest.approx(-0.15527988, rel=1e-6)

        # The two off-diagonal corners, separately, because they fail for
        # different reasons: `(m, c)` would mean the closure row saw the
        # potential's trace, `(c, m)` that a constraint row saw the rotation.
        floor = OFFDIAG_TOL * np.abs(rotation).min()
        assert np.abs(block[:n_mult, n_mult:]).max() <= floor
        assert np.abs(block[n_mult:, :n_mult]).max() <= floor
        assert np.abs(block[n_mult:, n_mult:] - np.diag(rotation)).max() <= floor
        if SERIAL:
            assert np.abs(block[:n_mult, n_mult:]).max() == 0.0
            assert np.abs(block[n_mult:, :n_mult]).max() == 0.0


class TestPresetWiring:
    """`"iterative"` must reach the iterative preset, and be seen to.

    It did not. The string was accepted, validated against the allowed set,
    and then ignored: every branch handed back the direct dictionary, whose
    own docstring says "2-D ONLY. Do not use this in 3-D". The failure is
    invisible in 2-D (where direct is right anyway) and in any driver that
    passes an explicit dictionary (which all of them did, because of this),
    so nothing caught it and four hand-copies of the iterative dictionary
    grew in the drivers instead. These tests exist so it cannot recur
    silently: each asserts a key that only one of the two presets carries.
    """

    def test_iterative_selects_the_iterative_preset(self, meshes):
        solver, _, _ = build(meshes, solver_parameters="iterative")
        p = solver.solver_parameters
        # LU on block 0 is the direct preset's signature and appears nowhere
        # in the iterative one.
        assert p["dtn_fieldsplit_0_pc_type"] == "fieldsplit"
        assert "dtn_fieldsplit_0_assembled_pc_type" not in p

    def test_direct_still_selects_the_direct_preset(self, meshes):
        solver, _, _ = build(meshes, solver_parameters="direct")
        p = solver.solver_parameters
        assert p["dtn_fieldsplit_0_assembled_pc_type"] == "lu"

    def test_the_two_presets_actually_differ(self, meshes):
        """The rejecting partner: a test that passes on the old code is useless.

        Before the wiring, both strings produced byte-identical dictionaries
        and every assertion above would have held for the wrong reason.
        """
        a, _, _ = build(meshes, solver_parameters="direct")
        b, _, _ = build(meshes, solver_parameters="iterative")
        assert a.solver_parameters != b.solver_parameters

    def test_the_default_is_direct_in_two_dimensions(self, meshes):
        """Dimension-aware, mirroring `GravitySolver` and `StokesSolverBase`.

        These meshes are 2-D, so the default must still be the direct preset
        and no existing 2-D result moves. The 3-D half of the rule cannot be
        exercised here - building a 3-D coupled space is not a unit test - so
        it is asserted on the branch itself below.
        """
        solver, _, _ = build(meshes)
        assert solver.mesh.topological_dimension == 2
        assert solver.solver_parameters["dtn_fieldsplit_0_assembled_pc_type"] \
            == "lu"

    def test_the_block_zero_sweep_follows_the_layout_not_the_caller(
            self, meshes):
        """`condensed` comes from the layout, so the mismatch is unreachable.

        `_check_block0_split_matches_layout` exists because the preset's
        `condensed` and the space's `condense_internal_variables` are
        independent arguments that nothing reconciles. On the preset path they
        are now the same fact, so the split count must match the space's
        block-0 field count for free.
        """
        solver, _, layout = build(meshes, solver_parameters="iterative")
        n_split = sum(
            1 for k in solver.solver_parameters
            if k.startswith("dtn_fieldsplit_0_pc_fieldsplit_")
            and k.endswith("_fields"))
        assert n_split == 2 + len(layout.internal_variables)

    def test_the_displacement_split_gets_the_rigid_body_pc(self, meshes):
        """Naming `firedrake.AssembledPC` here is the defect, not a milder choice.

        A `near_nullspace` declared on the outer mixed space never reaches
        GAMG underneath `DtNTwoBlockSchurPC`; the modes are dropped with no
        error and no warning. `b4_polar_motion` shipped exactly that.
        """
        solver, _, _ = build(meshes, solver_parameters="iterative")
        p = solver.solver_parameters
        u_split = next(
            s for s in range(3)
            if p.get(f"dtn_fieldsplit_0_pc_fieldsplit_{s}_fields") == "0")
        assert p[f"dtn_fieldsplit_0_fieldsplit_{u_split}_pc_python_type"] \
            == "gadopt.RigidBodyAssembledPC"

    def test_the_iterative_preset_solves_the_same_system(self, meshes, solved):
        """The one that is not about dictionaries: does the new path solve?

        Wiring a preset up is worth nothing if the path it now reaches does not
        converge to the same answer, and this preset had never been exercised
        *through this class* at all - every driver reached it by passing an
        explicit dictionary, and the class itself always returned the direct
        one. `solved` is the module fixture, which takes the default, which in
        2-D is direct.

        The tolerance is the iterative preset's own `outer_rtol` of 1e-6 and
        not a fudge: measured on this system the disagreement is 1.9e-06 at
        1e-6, 1.5e-08 at 1e-8 and 1.0e-10 at 1e-10, i.e. it tracks the Krylov
        tolerance four orders down. That is what says the two presets solve one
        discrete system rather than two close ones - a formulation difference
        would sit on a floor and refuse to move.
        """
        reference, z_ref, layout = solved
        solver, z, _ = build(meshes, solver_parameters="iterative")
        solver.solve()

        for field in (layout.displacement, layout.potential):
            a = z_ref.subfunctions[field].dat.data_ro
            b = z.subfunctions[field].dat.data_ro
            scale = max(abs(a).max(), 1e-300)
            assert abs(a - b).max() / scale < 1e-5

        ref_c, got_c = reference.coefficients(), solver.coefficients()
        for bc_id, modes in ref_c.items():
            scale = max(abs(v) for v in modes.values())
            for key, value in modes.items():
                assert abs(value - got_c[bc_id][key]) < 1e-5 * scale

    def test_every_sub_ksp_reports_its_exit_status(self, meshes):
        """Block 0 and block 1 are preconditioner applications inside FGMRES.

        Degrading either costs outer iterations rather than accuracy, so the
        only way to attribute a slow solve is each block's own converged
        reason. These are also where the campaign's block-0 application counts
        are read from, and from nowhere else.
        """
        solver, _, _ = build(meshes, solver_parameters="iterative")
        p = solver.solver_parameters
        for key in ("ksp_converged_reason",
                    "snes_converged_reason",
                    "dtn_fieldsplit_0_ksp_converged_reason",
                    "dtn_fieldsplit_1_ksp_converged_reason"):
            assert key in p


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
