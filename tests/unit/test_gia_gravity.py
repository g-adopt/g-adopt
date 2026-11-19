"""The monolithic self-gravitating GIA solver: space layout, constants, gates.

What these tests are for, and what they are not. They cover the *structure* of
`gadopt.gia_solver.SelfGravitatingGIASolver` and its couplings - the space
layout the preconditioner depends on, the two row-scaling constants, the
identity that makes the rotational body force the transpose of the inertia
row, the assertions that turn silently-empty measures into exceptions, and V4 -
and they run the solver once end to end on a small mesh to show that it solves
at all.

They are **not** the physics gates. V1 (per-block Jacobian symmetry), V3'
(the potential against a closed form), V7 (the rotational closure), V8 (the
geoid sign), V9 and V9b (the fluid limit) live elsewhere. The one gate here is
V4, because it is a statement about the *form* rather than about a solution
and belongs with the code that writes the form.

The mesh is gmsh, and it has to be: the geometry stands the DtN boundaries off
from the sources, which puts the Earth's surface *inside* the domain as a
tagged interior facet, and only gmsh can tag one. It is coarse - the tests here
are structural, and nothing in them is a convergence statement.

## The module helpers other test files use

`meshes`, `approximation`, `gravity_bcs`, `mechanics_bcs`, the constants and
`REAL_BLOCK_ON_PARENT_FACETS` keep the meaning they have always had: the
`gravity_bcs` sheet and the `mechanics_bcs` traction describe the load as the
earlier solver took it, in two places. The new solver takes the load once, as a
`SurfaceLoad` coupling, and refuses a sheet in its gravity conditions; `build`,
`TestFluidCore.build_fluid` and `TestCentreOfMassFrame.build` construct the new
solver that way and return `(solver, u, psi)`.
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
    rigid_body_modes,
)
from gadopt.gia_couplings import (
    CentreOfMass,
    FluidCore,
    Rotation,
    SurfaceLoad,
)
from gadopt.gia_presets import (
    NULL_COUPLING_ROW_SCALE,
    OMEGA_SQ_EARTH,
    selfgrav_dtn_iterative_solver_parameters,
)
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.internal_variable_equation import history_slices
from gadopt.momentum_equation import rotational_potential
from gadopt.nullspaces import apply_kernel_projection
from gadopt.utility import scalar_value

# Road-map §2.2 with the production constants.
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3

# The Phase 2 tag convention (road map §1.5), which the 3-D mesh shares.
CELL_MANTLE = 101
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5

RC, RE = 1.2037, 2.2037
N_AZIMUTHAL = 32

#: Applied to the tests that assemble a block of the coupled Jacobian with a
#: `Real` row or column on an interior facet of the parent (the fluid-core and
#: sea-level measures) as a matrix. The pyop3 Firedrake cannot build the
#: sparsity of such a block and raises `NotImplementedError: Currently cannot
#: deal with non-unit (vector-valued) columns` (or rows), on every pyop3 build
#: so far; the pyop2 Firedrake gave it as a matrix-free Python matrix. The
#: solver never assembles these blocks. Strict, and narrowed to that
#: exception, so that the test reports again as soon as Firedrake can assemble
#: the block, and any other failure in it stays visible. The alternative is
#: the rewrite that `test_gia_direct_bdf2.py` received: build a `Real` column
#: as the action on the unit `Real` function and a `Real` row as the action of
#: the adjoint block. Firedrake issue firedrakeproject/firedrake#5665 records
#: the limitation; once it is fixed, this mark goes and the tests assemble the
#: block directly.
REAL_BLOCK_ON_PARENT_FACETS = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason="pyop3 Firedrake cannot assemble a matrix block with a Real row or "
           "column on an interior facet of the parent (NotImplementedError: "
           "Currently cannot deal with non-unit (vector-valued) columns); "
           "this test assembles such a block. Remove this mark and assemble "
           "directly once firedrakeproject/firedrake#5665 is fixed")

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
    """A fresh approximation every time.

    The earlier solver, which other test files still build, writes `mu` into
    the approximation it is given; the new one does not, but a fresh object per
    solver keeps both safe.
    """
    settings = dict(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)
    settings.update(kwargs)
    return CompressibleInternalVariableApproximation(**settings)


def gravity_bcs(parent, *, truncation=3, sheet=True):
    """The potential's conditions: DtN on both stand-off circles.

    `sheet=True` adds the `cos 2 phi` load as an `interior_sigma` on Re, the
    form the earlier solver takes. The new solver refuses a sheet here; it
    takes the load as a `SurfaceLoad` coupling, so `build` passes
    `sheet=False`.
    """
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
    """The rigid core (`un = 0` on Rc) and the `cos 2 phi` traction on Re.

    The traction is the weight `B_mu g sigma` of the load with `g = 1`, the
    form the earlier solver takes. The new solver writes that traction itself
    from a `SurfaceLoad`, and refuses a `normal_stress` on the same boundary.
    """
    Xm = fd.SpatialCoordinate(sub)
    return {
        CURVE_RC: {"un": 0.0},
        CURVE_RE: {"normal_stress":
                   B_MU * SIGMA_HAT * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))},
    }


def relative_difference(a, b):
    """`|a - b| / |b|` in the L2 norm, for two fields on the same space."""
    return fd.norm(a - b) / fd.norm(b)


def load_density(mesh, degree=2):
    """The prescribed surface density `SIGMA_HAT cos(degree phi)`, UFL on `mesh`."""
    X = fd.SpatialCoordinate(mesh)
    return SIGMA_HAT * fd.cos(degree * fd.atan2(X[1], X[0]))


def new_fields(meshes):
    """Fresh `(u, psi)`: P2 displacement on the mantle, P2 potential on the parent.

    The degrees are those the earlier space factory used by default.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    return u, psi


def polar_moment(meshes):
    """`C = int_mantle r^2 dx`, the polar moment of the unit-density mantle (2-D)."""
    parent, sub = meshes
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    return fd.assemble(fd.dot(Xm, Xm) * dx_m)


def coupling(solver, cls):
    """The one coupling of class `cls` that `solver` holds."""
    found = [c for c in solver.couplings if isinstance(c, cls)]
    assert len(found) == 1, f"{len(found)} couplings of class {cls.__name__}"
    return found[0]


def build(meshes, *, rotation=True, truncation=3, sheet=True,
          approximation_kwargs=None, declare_nullspace=False,
          dtn_representation="lowrank", couplings=(), **kwargs):
    """The annulus solver: rigid core, `cos 2 phi` load on Re, optional rotation.

    Args:
      meshes: the `(parent, sub)` pair of the module fixture.
      rotation: add the `Rotation` coupling (`m3` only, in 2-D) with the
        mantle's polar moment and the Earth's `Omega^2`.
      truncation: the `CylindricalDtN` truncation on both DtN circles.
      sheet: `True` gives the load as a `SurfaceLoad`, which loads the
        mechanics, the potential and the rotation row. `False` gives the
        traction of `mechanics_bcs` alone, a load with no mass (the solver
        warns); a test that compares with a mechanics-only solver uses it.
      approximation_kwargs: settings merged into `approximation`.
      declare_nullspace: declare the rigid rotation, `nullspace="rigid_rotation"`.
      dtn_representation: `"lowrank"` (the default of the solver) or
        `"multiplier"`.
      couplings: further couplings, appended to those above.
      kwargs: passed to `SelfGravitatingGIASolver` (`solver_parameters`,
        `solver_parameters_extra`, `scaling_factor`, `nullspace`,
        `internal_variable_family`, ...). The Quadrature family refuses the
        rigid core's `un` condition; build a fluid core for it
        (`TestFluidCore.build_fluid`).

    Returns:
      `(solver, u, psi)`, with `u` and `psi` the Functions the solver writes
      after every solve.
    """
    parent, sub = meshes
    u, psi = new_fields(meshes)
    if sheet:
        bcs = {CURVE_RC: {"un": 0.0}}
        # `g_surface = 1` writes the traction `B_mu SIGMA_HAT cos 2phi` of
        # `mechanics_bcs` and puts the same density in the potential row.
        loads = [SurfaceLoad(boundary=CURVE_RE, sigma=load_density(sub),
                             g_surface=1.0)]
    else:
        bcs = mechanics_bcs(sub)
        loads = []
    if rotation:
        # `Omega_sq` is the earlier solver's default, now required.
        loads.append(Rotation(C=polar_moment(meshes), Omega_sq=OMEGA_SQ_EARTH))
    if declare_nullspace:
        kwargs.setdefault("nullspace", "rigid_rotation")
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(**(approximation_kwargs or {})), dt=1.0,
        gravity_bcs=gravity_bcs(parent, truncation=truncation, sheet=False),
        bcs=bcs, couplings=[*loads, *couplings],
        dtn_representation=dtn_representation, **kwargs)
    return solver, u, psi


@pytest.fixture(scope="module")
def solved(meshes):
    """One coupled solve, shared by the tests that only read a result."""
    solver, u, psi = build(meshes)
    solver.solve()
    return solver, u, psi


class TestSpaceLayout:
    """The invariants `DtNTwoBlockSchurPC` and the 3-D promotion depend on."""

    def test_field_order_and_meshes(self, meshes):
        parent, sub = meshes
        solver, _, _ = build(
            meshes, approximation_kwargs={"shear_modulus": [.4, .6],
                                          "viscosity": [.12, .6]})
        Z = solver.solution_space
        index = solver.field_index

        assert index["displacement"] == 0
        # One combined field holds both Maxwell elements, shape (n, d, d).
        assert index["history"] == 1
        assert index["potential"] == 2
        assert len(Z) == len(solver.field_names)
        assert Z[index["displacement"]].mesh() is sub
        assert Z[index["history"]].mesh() is sub
        assert Z[index["history"]].value_shape == (2, 2, 2)
        assert Z[index["potential"]].mesh() is parent

    def test_real_fields_are_contiguous_and_last(self, meshes):
        """The invariant `DtNTwoBlockSchurPC.initialize` asserts.

        Anything else would leave sub-fields out of both of its two blocks,
        which is a silently wrong split rather than an error.
        """
        solver, _, _ = build(meshes, rotation=True)
        Z = solver.solution_space

        real = [i for i, V in enumerate(Z)
                if V.ufl_element().family() == "Real"]
        assert real == list(range(real[0], len(Z)))
        assert tuple(real) == solver.real_indices
        assert real[0] > 0
        # Multipliers first, rotation last, and the rotation scalars are the
        # very end of the space.
        n_mult = len(solver.multiplier_indices)
        assert solver.multiplier_indices == tuple(real[:n_mult])
        assert [solver.field_index["m3"]] == real[n_mult:]

    @pytest.mark.parametrize("representation", ["multiplier", "lowrank"])
    @pytest.mark.parametrize("rotation", [False, True])
    @pytest.mark.parametrize("family", ["DG", "Quadrature"])
    def test_fluid_core_field_order(self, meshes, representation, rotation,
                                    family):
        """The core pressure stays between the DtN and rotation `Real` fields."""
        parent, _ = meshes
        solver, _, _ = TestFluidCore.build_fluid(
            meshes, rotation=rotation, internal_variable_family=family,
            dtn_representation=representation)
        Z = solver.solution_space
        index = solver.field_index

        core = index["core_pressure"]
        assert Z[core].mesh() is parent
        assert Z[core].ufl_element().family() == "Real"
        assert core == (solver.multiplier_indices[-1] + 1
                        if solver.multiplier_indices
                        else index["potential"] + 1)
        if rotation:
            assert core < index["m3"]
        real = tuple(i for i, V in enumerate(Z)
                     if V.ufl_element().family() == "Real")
        assert real == solver.real_indices
        assert real == tuple(range(real[0], len(Z)))
        assert len(Z) == len(solver.field_names)

    def test_no_core_pressure_by_default(self, meshes):
        solver, _, _ = build(meshes, rotation=False)
        assert "core_pressure" not in solver.field_index

    def test_real_spaces_live_on_the_parent(self, meshes):
        """Spike S2's decision, and it is not cosmetic.

        On the parent both families of constraint row - parent-boundary DtN
        rows and submesh-volume rotation rows - assemble against the measures
        the mechanics already needs. On the submesh every parent-boundary row
        needs a specially intersected `ds`, and an intersected measure that
        finds nothing assembles to zero without raising.
        """
        parent, _ = meshes
        solver, _, _ = build(meshes, rotation=True,
                             dtn_representation="multiplier")
        assert solver.real_indices
        for i in solver.real_indices:
            assert solver.solution_space[i].mesh() is parent

    def test_multiplier_count_comes_from_the_boundaries(self, meshes):
        """`2M` on the exterior boundary, `2M + 1` on the interior one.

        And the side is *measured* from the mesh rather than declared, which is
        why the space cannot be built before the boundary conditions are known.
        """
        for M in (3, 5):
            solver, _, _ = build(meshes, rotation=False, truncation=M,
                                 dtn_representation="multiplier")
            assert len(solver.multiplier_indices) == (2 * M) + (2 * M + 1)

    def test_rotation_is_named_m3_in_2d(self, meshes):
        """Index **2** of the rotation triple, never "the first rotation field".

        A disc has no polar wander: `m_1` and `m_2` tilt the rotation axis out
        of a plane with no third direction, and the only surviving mode is the
        rotation-rate change. A layout that recorded it positionally would
        silently reindex on promotion to 3-D and every test reading a rotation
        coefficient by position would change meaning.
        """
        solver, _, _ = build(meshes, rotation=True)
        rotation = coupling(solver, Rotation)
        names = set(solver.field_names)
        assert "m3" in names and not {"m1", "m2"} & names
        assert set(rotation.values()) == {"m3"}
        assert rotation.slots(solver.coupling_context) == (None, None, "m3")

    def test_no_rotation_by_default(self, meshes):
        solver, _, _ = build(meshes, rotation=False)
        assert not {"m1", "m2", "m3"} & set(solver.field_names)
        assert not [c for c in solver.couplings if isinstance(c, Rotation)]

    def test_parent_must_declare_cartesian(self, meshes):
        """G-ADOPT reads `mesh.cartesian` everywhere and a file mesh has none.

        Refused at construction, where both meshes are in scope and the message
        can say what to set, rather than several frames deep in
        `upward_normal`.
        """
        _, sub = meshes
        undeclared = fd.UnitSquareMesh(2, 2)
        assert not hasattr(undeclared, "cartesian")
        u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2))
        psi = fd.Function(fd.FunctionSpace(undeclared, "CG", 2))
        with pytest.raises(ValueError, match="cartesian"):
            SelfGravitatingGIASolver(u, psi, approximation(), dt=1.0,
                                     gravity_bcs={})

    def test_submesh_inherits_cartesian_from_the_parent(self, meshes):
        """`Submesh` does not inherit it, and the failure is far from the cause.

        Without it the first thing that asks the submesh whether it is
        Cartesian raises `AttributeError: 'MeshTopology' object has no
        attribute 'cartesian'` from inside `upward_normal`. The fresh submesh
        is curved as the fixture's is, because the solver measures the
        cross-mesh entity maps at construction and refuses a straight-sided
        submesh.
        """
        from validate_selfgrav_annulus import curve_mesh

        parent, _ = meshes
        fresh = curve_mesh(fd.Submesh(parent, 2, CELL_MANTLE))
        assert not hasattr(fresh, "cartesian")
        build((parent, fresh), rotation=False)
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

        The solver multiplies the whole momentum residual by `scaling_factor`,
        the two body forces included, so a `theta_psi` that ignored it would
        make the coupled Jacobian asymmetric by exactly that factor - which
        reads as a sign error in the new code.
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
        rotation = coupling(solver, Rotation)
        ctx = solver.coupling_context
        expected = -B_MU * OMEGA_SQ_EARTH
        assert float(rotation.theta_rot(ctx, 2)) == pytest.approx(
            expected, rel=1e-14)
        assert float(rotation.theta_rot(ctx, 2)) < 0.0
        assert float(rotation.theta_rot(ctx, 0)) == pytest.approx(
            +B_MU * OMEGA_SQ_EARTH, rel=1e-14)
        # And it is not theta_psi, by three orders of magnitude.
        assert abs(float(rotation.theta_rot(ctx, 2))) < 1e-2 * float(
            solver.theta_psi)

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
        p3 = Rotation.inertia_polynomial(2, X)
        psi_rot = rotational_potential([1.0], sub, Omega_sq=1.0)
        assert fd.assemble((psi_rot - p3) ** 2 * fd.dx(domain=sub)) < 1e-24

    def test_polynomial_matches_the_rotational_potential_in_3d(self):
        """Written in three components unconditionally, so 3-D comes free."""
        mesh = fd.UnitCubeMesh(1, 1, 1)
        mesh.cartesian = True
        X = fd.SpatialCoordinate(mesh)
        for i, m in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                               (0.0, 0.0, 1.0)]):
            p_i = Rotation.inertia_polynomial(i, X)
            psi_rot = rotational_potential(list(m), mesh, Omega_sq=1.0)
            assert fd.assemble((psi_rot - p_i) ** 2 * fd.dx(domain=mesh)) < 1e-24

    @pytest.mark.parametrize("component", [0, 1])
    def test_2d_refuses_a_polar_wander_component(self, component):
        """A polar-wander component needs `z`, which a 2-D mesh does not have."""
        mesh = fd.UnitSquareMesh(1, 1)
        X = fd.SpatialCoordinate(mesh)
        with pytest.raises(ValueError, match="third coordinate"):
            Rotation.inertia_polynomial(component, X)

    def test_inertia_of_a_cos2_sheet_vanishes(self, solved):
        """`int sigma (x^2 + y^2) dS` over a circle kills a `cos 2phi` load.

        Not a tautology: it is the sheet half of `dI_33`, it is integrated on
        the load's measure, a tagged *interior* facet of the parent, and a
        sheet written on the wrong kind of measure would give the same zero -
        which is why the measure assertions exist separately.
        """
        solver, _, _ = solved
        rotation = coupling(solver, Rotation)
        assert abs(fd.assemble(rotation.inertia_perturbation()["dI_33"])) < 1e-14
        assert abs(float(rotation.values()["m3"])) < 1e-14


class TestGeometryAssertions:
    """Silent zeros turned into exceptions, at construction."""

    def test_a_sheet_on_the_wrong_kind_of_facet_is_refused(self, meshes):
        """Re is an interior facet of the parent; `sigma` would find nothing.

        A load sheet written the exterior-facet way contributes nothing, warns
        rather than raising, and the solve converges with the potential
        missing the largest single term in the geoid. Nothing downstream
        looks: a sheet is a right-hand side with no Jacobian contribution, so
        a symmetry test cannot see it, and two solvers sharing the form omit
        it identically. The boundary form refuses it while it is built, before
        the solver's own refusal of any sheet in its gravity conditions.
        """
        parent, sub = meshes
        bcs = gravity_bcs(parent, sheet=False)
        bcs[CURVE_RE] = {"sigma": SIGMA_HAT}
        u, psi = new_fields(meshes)
        with pytest.raises(ValueError, match="sheet measure is empty"):
            SelfGravitatingGIASolver(u, psi, approximation(), dt=1.0,
                                     gravity_bcs=bcs, bcs=mechanics_bcs(sub))

    def test_dtn_boundary_measures_match_their_radii(self, solved):
        """What `check_dtn_boundaries_complete` asserts, restated so a reader can see it.

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
        area = fd.assemble(probe * solver.gravity.dx_m)
        assert area == pytest.approx(np.pi * (RE**2 - RC**2), rel=1e-3)

    def test_intersected_parent_measure_covers_the_whole_parent(self, solved):
        """Intersecting must not restrict the Laplacian to the mantle.

        Firedrake keeps the parent's own cells and uses the intersection only
        to widen the admissible domains; the alternative reading is a converged
        solve with the stand-off buffer and both DtN boundaries absent.
        """
        solver, _, _ = solved
        whole = fd.assemble(fd.Constant(1.0) * solver.gravity.dx_parent)
        assert fd.assemble(fd.Constant(1.0) * solver.gravity.dx_g) \
            == pytest.approx(whole, rel=1e-12)
        assert whole == pytest.approx(np.pi * ((2 * RE) ** 2 - (0.5 * RC) ** 2),
                                      rel=1e-3)

    def test_a_solver_without_a_declared_lambda_is_refused(self, meshes):
        """The self-gravity number comes from the approximation alone.

        The boundary form's `G = Lambda / (4 pi)` and the volume source's
        `Lambda` are one number read from one place, so a missing one is
        refused at construction instead of leaving the sheets unscaled.
        """
        parent, sub = meshes
        u, psi = new_fields(meshes)
        with pytest.raises(ValueError, match="self_gravity_number"):
            SelfGravitatingGIASolver(
                u, psi, approximation(self_gravity_number=None), dt=1.0,
                gravity_bcs=gravity_bcs(parent, sheet=False),
                bcs={CURVE_RC: {"un": 0.0}})


class TestConstruction:
    """What the solver builds on a mixed space spanning two meshes."""

    def test_mesh_is_the_mechanics_mesh(self, solved):
        """The solver's `mesh` is the displacement's mesh, not the parent.

        G-ADOPT's terms read `solver.mesh` for the normals and the measures
        of the momentum equation, which live on the mantle.
        """
        solver, u, _ = solved
        assert isinstance(solver.mesh, fd.MeshGeometry)
        assert solver.mesh is u.function_space().mesh()
        assert solver.mesh is not solver.potential_mesh

    def test_one_equation_per_mechanics_field(self, solved):
        """And not one per sub-field: the multipliers are not `Equation`s.

        The residual zips equations against the mechanics unknowns, and here
        that would truncate - correctly, by accident, which is exactly the kind
        of accident that stops being true when somebody reorders the space.
        """
        solver, _, _ = solved
        assert len(solver.equations) == 1 + ("history" in solver.field_index)
        assert len(solver.equations) == 2

    @pytest.mark.parametrize("solver_parameters", ["direct", "iterative", None])
    def test_power_law_is_accepted_on_the_uncondensed_layout(
            self, meshes, solver_parameters):
        """A power law is a supported configuration of every options path.

        The DG family keeps the internal variables as unknowns, so the
        power-law factor enters the history equation and pyadjoint tapes it,
        and the nested preconditioner follows the state-dependent Jacobian on
        its own: the inner fieldsplit re-runs its setup after every Jacobian
        reassembly, so every sub-preconditioner rebuilds once per Newton
        iteration (measured, `NOTES/PLAN-POWER-LAW-SELFGRAVITY.md` section 2).
        Construction must therefore succeed on all three ways of asking for
        options, and Newton must be the outer method on each.

        `None` selects the direct preset here, because the annulus is 2-D.

        The `"iterative"` case carries one extra fact: on the preset's own
        default the displacement split stays `preonly` even under a power law.
        A power law makes the condensed displacement operator nonsymmetric in
        2-D, and a short CG would be invalid on it, so the solver's
        `_adjust_options` replaces a CG by a GMRES of the same length. The
        default writes no CG to replace: one GAMG V-cycle is not a Krylov
        method and assumes no symmetry, so the switch does not fire and no
        restart key is written. The prefix is the default block-0 route's,
        `gadopt.CondensedBlockPC`, whose displacement split sits under
        `dtn_fieldsplit_0_condensed_fieldsplit_0_`.
        """
        solver, _, _ = build(
            meshes,
            approximation_kwargs={"exponent": 3.0, "transition_stress": 1e-3},
            solver_parameters=solver_parameters)
        assert solver.solver_parameters["snes_type"] == "newtonls"
        if solver_parameters == "iterative":
            prefix = "dtn_fieldsplit_0_condensed_fieldsplit_0_"
            assert solver.solver_parameters[prefix + "ksp_type"] == "preonly"
            assert prefix + "ksp_gmres_restart" \
                not in solver.solver_parameters

    def test_the_displacement_switch_fires_for_the_truncated_cg(self, meshes):
        """The caller who asks for the CG still gets the GMRES a power law needs.

        This is the other half of the test above. `u_ksp_max_it=4` puts a
        truncated CG on the displacement split, which is valid only on a
        symmetric operator, and a power law makes the condensed displacement
        operator nonsymmetric. `_adjust_options` must then replace it by a
        GMRES whose restart is the cap the preset wrote, so that the substitute
        costs the same Krylov vectors and the same number of GAMG V-cycles per
        block-0 iteration.

        The options come in as a dictionary and not as the string
        `"iterative"`, because that also exercises the dictionary branch of
        the switch. The representation is named on both sides so that the
        dictionary describes the space `build` makes.
        """
        params = selfgrav_dtn_iterative_solver_parameters(
            condensed=False, snes_type="newtonls", u_ksp_max_it=4,
            dtn_representation="lowrank")
        solver, _, _ = build(
            meshes, dtn_representation="lowrank",
            approximation_kwargs={"exponent": 3.0, "transition_stress": 1e-3},
            solver_parameters=params)
        prefix = "dtn_fieldsplit_0_condensed_fieldsplit_0_"
        assert solver.solver_parameters[prefix + "ksp_type"] == "gmres"
        assert solver.solver_parameters[prefix + "ksp_gmres_restart"] == 4

    def test_ksponly_with_a_power_law_is_refused(self, meshes):
        """One linear solve on a nonlinear residual reports CONVERGED.

        `ksponly` is the right method for `exponent = 1` and both preset
        docstrings recommend it there, so the misuse this guard catches is a
        copy of such a dictionary into a power-law run. The wrong answer
        carries no diagnostic at all: SNES exits converged after its single
        linear solve at the initial state.
        """
        with pytest.raises(ValueError, match="ksponly"):
            build(meshes,
                  approximation_kwargs={"exponent": 3.0,
                                        "transition_stress": 1e-3},
                  solver_parameters_extra={"snes_type": "ksponly"})

    def test_power_law_is_accepted_on_the_lowrank_representation(self, meshes):
        """A power law builds and solves on the low-rank representation.

        `test_gia_gravity_adjoint_lowrank.py::test_taylor_with_a_power_law`
        takes the gradient at exponent 3 on both presets. This one pins the
        acceptance itself, and that the Newton solve converges: `B` does not
        depend on the rheology and is reinstalled after every Jacobian
        assembly, so nothing in the low-rank path reads the exponent.
        """
        solver, u, _ = build(
            meshes, rotation=False, dtn_representation="lowrank",
            approximation_kwargs={"exponent": 3.0, "transition_stress": 1e-3})
        solver.solve()
        assert solver.solver.snes.getConvergedReason() > 0
        assert fd.norm(u) > 0.0

    def test_it_solves(self, solved):
        solver, u, psi = solved
        assert fd.norm(u) > 0.0
        assert fd.norm(psi) > 0.0
        assert np.isfinite(fd.norm(psi))

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


class TestNewtonianLimit:
    """A power-law configuration that is Newtonian must solve the Newtonian problem.

    Two statements, and they are different. The first is exact: at
    `exponent = 1` the `0^0` guard in `power_law_factor` makes the factor
    identically 1, whatever the transition and background stresses are, so the
    two residuals are the same form up to UFL simplification and the two
    solutions agree to roundoff. The second is a limit: at `exponent = 1 + eps`
    with a positive background stress the factor is `1 + O(eps)` and the
    solutions agree to that order.

    Both are route tests on a 2-D annulus, so neither says anything about
    power-law physics: in 2-D the factor is evaluated on a stress carrying the
    spurious trace of the fixed 2/3 in the deviatoric operator.
    """

    @staticmethod
    def _fields(solver, u, psi):
        """The three states to compare: displacement, history, potential.

        The history is the DG field of the mixed solution; with one Maxwell
        element it holds one `(d, d)` slice.

        Args:
          solver: the solved `SelfGravitatingGIASolver`.
          u: the displacement Function the solver writes.
          psi: the potential Function the solver writes.

        Returns:
          A tuple `(u, m, psi)`.
        """
        m, = history_slices(
            solver.solution.subfunctions[solver.field_index["history"]])
        return u, m, psi

    def test_exponent_one_with_power_law_parameters_is_the_newtonian_solve(
            self, meshes):
        """Exponent 1 makes the transition and background stresses inert.

        `power_law_factor` guards the `0^0` that UFL evaluates as 0 by
        conditioning on `eq(n - 1, 0)`, so at `n = 1` both powers are 1 and the
        factor is `(1 + 1) / (1 + 1) = 1` for every transition stress and every
        background stress, the zero background included. The two residuals are
        then the same form up to UFL simplification, and 1e-12 relative is the
        roundoff that simplification can cost.
        """
        plain, u_plain, psi_plain = build(meshes, solver_parameters="direct")
        power, u_power, psi_power = build(
            meshes, solver_parameters="direct",
            approximation_kwargs={"exponent": 1, "transition_stress": 1e-4,
                                  "background_stress": 0.5})
        plain.solve()
        power.solve()

        u_p, m_p, psi_p = self._fields(plain, u_plain, psi_plain)
        u_q, m_q, psi_q = self._fields(power, u_power, psi_power)
        assert fd.norm(u_p) > 0.0
        assert fd.norm(m_p) > 0.0
        # Measured on this annulus, uncondensed: 7.0e-17 in `u`, 0 in `m`,
        # 6.9e-17 in `psi` (plan review, table under B1). The gate is five
        # orders above that, so a factor that is not exactly 1 fails it.
        differences = {"u": relative_difference(u_q, u_p),
                       "m": relative_difference(m_q, m_p),
                       "psi": relative_difference(psi_q, psi_p)}
        summary = "  ".join(f"{k}={v:.2e}" for k, v in differences.items())
        print(f"\n    [exponent 1] {summary}")
        for key, value in differences.items():
            assert value < 1e-12, f"{key} differs by {value:.3e}"

    def test_exponent_near_one_is_the_newtonian_solve_to_first_order(
            self, meshes):
        r"""At `n = 1 + eps` with a positive background stress, `f = 1 + O(eps)`.

        With `a = sigma_bg / sigma*` and `b = (sigma + sigma_bg) / sigma*`, the
        factor is `f = (1 + a^eps) / (1 + b^eps)`. Choosing
        `background_stress = transition_stress` gives `a = 1` exactly, so
        `f = 2 / (1 + b^eps)` and

            |f - 1| <= (eps / 2) ln(1 + sigma_max / sigma*),

        of order 1e-7 on this annulus. The displacement gate is 1e-6 relative;
        the measured difference is 5.8e-10 (plan review, table under B1).

        **The limit needs a positive background stress, and the third
        assertion pins why.** With `background_stress = 0` the model is
        discontinuous in the exponent at `n = 1`: UFL evaluates a zero base to
        a positive power as 0, so `a^(n-1)` is 0 for every `n > 1`, while the
        guard sets it to 1 at `n = 1` exactly. The factor therefore tends to
        1/2 rather than to 1 as `n` approaches 1 from above, the Maxwell times
        halve, and the state differs from the Newtonian one by 25 percent in
        `u` (measured). Asserting that gap keeps the property visible: a future
        change to the guard that makes the zero-background case continuous
        fails here, and so gets read rather than absorbed.
        """
        newtonian, u_newtonian, _ = build(meshes, solver_parameters="direct")
        near, u_near, _ = build(
            meshes, solver_parameters="direct",
            approximation_kwargs={"exponent": 1 + 1e-6,
                                  "transition_stress": 1e-3,
                                  "background_stress": 1e-3})
        # Newton is the outer method, which is the whole reason this
        # configuration is reachable: the residual is nonlinear for any
        # exponent other than 1.
        assert near.solver_parameters["snes_type"] == "newtonls"
        newtonian.solve()
        near.solve()
        assert near.solver.snes.getConvergedReason() > 0

        assert fd.norm(u_newtonian) > 0.0
        near_difference = relative_difference(u_near, u_newtonian)
        print(f"\n    [exponent 1+1e-6] background 1e-3: "
              f"u={near_difference:.2e}")
        assert near_difference < 1e-6

        # The zero-background control: same exponent, same transition stress,
        # a model that is discontinuous at n = 1 and a state that says so.
        zero_background, u_zero, _ = build(
            meshes, solver_parameters="direct",
            approximation_kwargs={"exponent": 1 + 1e-6,
                                  "transition_stress": 1e-3,
                                  "background_stress": 0.0})
        zero_background.solve()
        zero_difference = relative_difference(u_zero, u_newtonian)
        print(f"    [exponent 1+1e-6] background 0: u={zero_difference:.2e}")
        assert zero_difference > 1e-1


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


def real_block_index_sets(solver):
    """The block-1 index sets, taken the way `DtNTwoBlockSchurPC` takes them.

    `Z.field_ises[i_R:]`, i.e. Firedrake's own authority on where each
    sub-field sits in the monolithic row space, sliced at the first `Real`
    sub-field. `DtNTwoBlockSchurPC.initialize` merges exactly this slice into
    its "1" split, so a probe that reads through it exercises the
    preconditioner's own indexing rather than a second, parallel-untested one.
    """
    return solver.solution_space.field_ises[solver.real_indices[0]:]


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


def real_block(solver, F=None):
    """The full `n x n` `Real` block of the Jacobian, by residual differencing.

    `F(e_j) - F(0)` read on the `Real` rows is the `j`th column exactly - the
    coupled residual is linear in every `Real` unknown, Newtonian rheology
    making the whole Jacobian constant - so this is a *measurement* of the
    block and not a finite difference with a step size to argue about. Measured
    against `mat_type="nest"` it agrees to 2.234e-16 relative.

    Differencing rather than nest assembly, and the reason is not stylistic:
    the shipped presets run `mat_type: matfree`, and asking for a nest
    Jacobian materialises every block including `(u,u)` and `(psi,psi)`, which
    is the assembled operator the configuration exists to avoid. At production
    size that is the most likely explanation for the 180 GB that killed the job
    this gate was first attempted in. Here it costs `n + 1` residual
    assemblies, and nothing else.

    Args:
      solver: the solver; its mixed solution is overwritten (left at zero).
      F: the residual to difference, `solver.F` by default.

    Returns:
      The block as a dense array, in `solver.real_indices` order.
    """
    F = solver.F if F is None else F
    z = solver.solution
    comm = solver.solution_space.mesh().comm
    ises = real_block_index_sets(solver)
    real = solver.real_indices

    z.assign(0.0)
    base = read_real_rows(fd.assemble(F), ises, comm)
    block = np.zeros((len(real), len(real)))
    for column, field in enumerate(real):
        z.assign(0.0)
        z.subfunctions[field].assign(1.0)
        block[:, column] = read_real_rows(fd.assemble(F), ises, comm) - base
    z.assign(0.0)
    return block


def predicted_real_diagonal(solver):
    """The diagonal of the `Real` block as the derivation gives it, per field.

    Written from the constants of the form, never assembled:

    * a DtN multiplier: `theta_psi * (-scale_k * A_h)`, from
      `DtNGravityForm.multiplier_diagonal`, which derives the sign and uses the
      *discrete* boundary measure `A_h`;
    * a rotation row: `theta_rot_i * K_i = s_i c B_mu Omega^2 K_i`, with
      `K = (C - A, C - A, C)`; the row divides `K_i m_i` by the volume of the
      measure it is integrated on, so the volume cancels from the entry;
    * the core pressure and the centre-of-mass multipliers: exactly zero,
      because they are Lagrange multipliers.

    `theta_psi` is read as the solver's number, never recomputed, so that the
    floor of `B_mu` at zero is the one the residual carries.

    Args:
      solver: the solver.

    Returns:
      The diagonal as an array in `solver.real_indices` order.
    """
    by_name = {}
    names = solver.field_names
    multipliers = [names[i] for i in solver.multiplier_indices]
    if multipliers:
        values = solver.theta_psi_value * solver.form.multiplier_diagonal()
        assert len(values) == len(multipliers)
        by_name.update(zip(multipliers, values))
    rotations = [c for c in solver.couplings if isinstance(c, Rotation)]
    if rotations:
        rotation, = rotations
        ctx = solver.coupling_context
        for k, name in enumerate(rotation.slots(ctx)):
            if name is None:
                continue
            by_name[name] = (rotation.CLOSURE_SIGNS[k]
                             * scalar_value(solver.scaling_factor)
                             * scalar_value(solver.row_scale_B_mu)
                             * scalar_value(rotation.Omega_sq)
                             * scalar_value(rotation.closure_constants[k]))
    for name in names:
        if name == "core_pressure" or name.startswith("centre_of_mass_"):
            by_name[name] = 0.0
    missing = [names[i] for i in solver.real_indices if names[i] not in by_name]
    assert not missing, f"no predicted diagonal entry for {missing}"
    return np.array([by_name[names[i]] for i in solver.real_indices])


def assembly_floor(solver):
    """`max |F(0) - F(0)|` over the `Real` rows, from two identical assemblies.

    The reproducibility of the assembly itself, measured rather than assumed.

    **`F(0)` on the `Real` rows is not zero, and it has a plain cause.** It
    lives *entirely* on the rotation row **at the rank counts a laptop can
    reach**: measured at 1, 2, 3 and 4 ranks, every multiplier row is exactly
    0.0 and only the `m_3` row is not. That ceiling matters - at 48 ranks the
    multiplier block acquires a spurious coupling of 5.7199e-05 where serial
    reads 0.0 (a different quantity from this one, but the same class of claim),
    so "the multiplier rows are clean" must not be stated without a scale. The
    rotation row carries the load sheet's `int nu sigma p_3 dS`, and for this
    configuration's `cos 2phi` load `int cos(2 phi) r^2 ds` is **analytically
    zero** - the same quantity `test_inertia_of_a_cos2_sheet_vanishes` asserts
    vanishes. What survives is that integral's discrete remainder, scaled by
    `theta_rot`. So the residual is not structurally zero; it is one
    physically-zero integral failing to be discretely zero, in the one row
    carrying a prescribed sheet, and it cancels in the differencing.

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
    solver.solution.assign(0.0)
    ises = real_block_index_sets(solver)
    comm = solver.solution_space.mesh().comm
    first = read_real_rows(fd.assemble(solver.F), ises, comm)
    second = read_real_rows(fd.assemble(solver.F), ises, comm)
    return float(np.abs(first - second).max())


def check_multiplier_alignment(solver, *, silent_tol=1e-6, predicted_tol=1e-12):
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

    Args:
      solver: a solver on the multiplier representation; its mixed solution is
        overwritten (left at zero).
      silent_tol: the bound on a row that the driven mode must not move,
        relative to the weakest fired row.
      predicted_tol: the bound on a fired row against its discrete prediction.

    Returns:
      `{"discrete", "analytic", "silent"}`: the worst relative deviations.
    """
    z = solver.solution
    comm = solver.solution_space.mesh().comm
    ises = real_block_index_sets(solver)
    form = solver.form
    theta = float(solver.theta_psi)
    psi_h = z.subfunctions[solver.field_index["potential"]]
    multipliers = solver.multiplier_indices

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
                f"sub-field {multipliers[i]} claims to be {keys[i]} but "
                "does not respond to it as one")
            worst_discrete = max(
                worst_discrete, abs(row[i] - discrete) / abs(discrete))
            analytic = theta * float(mode.norm)
            worst_analytic = max(
                worst_analytic, abs(row[i] - analytic) / abs(analytic))
        weakest = min(abs(row[i]) for i in fired)
        for i in silent:
            assert abs(row[i]) < silent_tol * weakest, (
                f"driving {driven} moved sub-field {multipliers[i]}, "
                f"which claims to be {keys[i]}")
            worst_silent = max(worst_silent, abs(row[i]) / weakest)
        # And the rotation rows never see the potential at all.
        for i in range(len(keys), len(solver.real_indices)):
            assert abs(row[i]) < silent_tol * weakest
    z.assign(0.0)
    return {"discrete": worst_discrete, "analytic": worst_analytic,
            "silent": worst_silent}


def offdiagonal_ratios(block):
    """Per row, `max_j |A[i,j]| / |A[i,i]|` over `j != i`."""
    off = block.copy()
    np.fill_diagonal(off, 0.0)
    return np.abs(off).max(axis=1) / np.abs(np.diag(block))


def block_asymmetry(F, z, J=None):
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

    Args:
      F: the residual.
      z: the mixed unknown of `F`.
      J: the Jacobian to measure; `None` takes `derivative(F, z)`.
    """
    Z = z.function_space()
    J = fd.derivative(F, z) if J is None else J
    A = fd.assemble(J, mat_type="nest").petscmat
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
    """The DG family's `(u, M, psi)` block 0 is not symmetric.

    The internal-variable rows are the asymmetric ones: the driver's own
    instrument reads `diff[0,1] = 4.768e-01` against `size[0,1] = 3.576e-01`
    on this annulus, a ratio of 1.333, i.e. the `(u, sigma)` pair is not even
    close to transposing. That is not a defect - the free-slip Nitsche terms
    carry `mu` against `mu0`, recorded in
    `NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md` - but it does mean a
    symmetry statement belongs to a system with the history eliminated.

    Serial, like every per-block measurement in this project: `block_asymmetry`
    builds a global dense transpose. The block-1 probe below has no such
    restriction and runs under MPI.
    """

    pytestmark = pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs a global dense transpose")

    def test_the_uncondensed_block_is_not(self, meshes):
        """Measured **2.528e-03** relative on this annulus.

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
        asymmetry is fixed at source, likewise: delete, do not loosen.
        """
        solver, _, _ = build(meshes)
        z = solver.solution
        block0 = [i for i in range(len(solver.solution_space))
                  if i not in solver.real_indices]
        assert len(block0) == 3
        diff, size = block_asymmetry(solver.F, z)
        window = np.ix_(block0, block0)
        assert diff[window].max() / size[window].max() > 1e-6


class TestEliminatedBlockSymmetry:
    """Block 0 with the history eliminated is symmetric; with it, it is not."""

    def test_the_eliminated_block_is_symmetric(self, meshes):
        """The Quadrature family's `(u, psi)` block 0 is symmetric to round-off.

        On the Quadrature family the history is eliminated at the quadrature
        points, and the operator of the step is the Jacobian of the step
        backend with that elimination in it (`DirectConstitutiveBackend.J`),
        plus the low-rank DtN update. The presets solve the eliminated
        displacement block with CG because it is symmetric; this measures
        that, for a linear Maxwell rheology of two elements with a fluid core
        and rotation.

        The instrument is the bilinear form of the operator: for three
        coordinate-defined directions `x_k` with zero `Real` components,
        `|x_i . J x_j - x_j . J x_i|` relative to the largest `|x_i . J x_j|`.
        It runs on any number of ranks, unlike a dense block transpose (and
        the nest assembly that one needs does not accept the backend's
        Jacobian). The same instrument on the DG family, whose block 0 also
        holds the history rows, reads far above round-off, so it is not
        vacuous there.
        """
        import gia_solver_cases as cases

        def asymmetry(family):
            solver, _, _ = TestFluidCore.build_fluid(
                meshes, rotation=True, internal_variable_family=family,
                approximation_kwargs={"viscosity": [1.0, 3.0],
                                      "shear_modulus": [1.0, 0.5]})
            directions = []
            for k in range(3):
                x = cases.coordinate_mixed_function(solver, f"symmetry{k}", 1.0)
                for index in solver.real_indices:
                    x.subfunctions[index].assign(0.0)
                directions.append(x)
            # The actions of the operator the step solves, at the solver's
            # state (the backend's Jacobian on the Quadrature family).
            _, actions = cases.residual_and_actions(solver, directions)
            pairing = np.zeros((3, 3))
            for i, x in enumerate(directions):
                for j, action in enumerate(actions):
                    with action.dat.vec_ro as a, x.dat.vec_ro as v:
                        pairing[i, j] = a.dot(v)
            assert np.abs(pairing).max() > 0.0
            return np.abs(pairing - pairing.T).max() / np.abs(pairing).max()

        quadrature = asymmetry("Quadrature")
        dg = asymmetry("DG")
        print(f"bilinear asymmetry: Quadrature {quadrature:.3e}, DG {dg:.3e}")
        assert quadrature <= 1e-12
        assert dg > 1e-6


#: Bounds on the relative L2 gap between the DG and Quadrature families after
#: two steps on this annulus, about ten times the measured gaps (1.5e-4 in
#: `u` and 7.9e-4 in `psi`).
FAMILY_GAP_U = 2e-3
FAMILY_GAP_PSI = 1e-2


class TestFamiliesAgree:
    """The DG and Quadrature history families describe one physical system.

    The DG family carries the history as a DG1 unknown in the mixed space;
    the Quadrature family eliminates it at the quadrature points. Both are
    built here with a fluid core, because the Quadrature family supports
    natural displacement boundaries only.
    """

    MAXWELL2 = {"viscosity": [1.0, 3.0], "shear_modulus": [1.0, 0.5]}

    def test_dg_and_quadrature_agree_after_two_steps(self, meshes):
        """Two steps of each family agree in `u` and `psi` to the discretisation gap.

        The two families differ in how the history is represented (DG1
        against values at the quadrature points), so they agree to a
        discretisation error of this resolution and not to round-off. The
        bound is about ten times the measured gap; a sign or factor error
        in either family moves the fields by order one.
        """
        fields = {}
        for family in ("DG", "Quadrature"):
            solver, u, psi = TestFluidCore.build_fluid(
                meshes, rotation=True, internal_variable_family=family,
                approximation_kwargs=self.MAXWELL2, nullspace="rigid_rotation",
                solver_parameters="direct")
            solver.solve()
            solver.solve()
            fields[family] = (u, psi)
        (u_dg, psi_dg), (u_q, psi_q) = fields["DG"], fields["Quadrature"]
        assert fd.norm(u_q) > 0.0
        gap_u = relative_difference(u_dg, u_q)
        gap_psi = relative_difference(psi_dg, psi_q)
        print(f"DG against Quadrature: u {gap_u:.3e}, psi {gap_psi:.3e}")
        assert gap_u < FAMILY_GAP_U
        assert gap_psi < FAMILY_GAP_PSI

    def test_the_real_block_does_not_depend_on_the_family(self, meshes):
        """The `Real` block of the Jacobian is the same matrix in both families.

        Eliminating the history touches no constraint row, so the block of
        the DtN multipliers, the core pressure and the rotation row is the
        same to round-off. Both use the multiplier representation, which has
        one `Real` row per DtN mode.
        """
        blocks = []
        for family in ("DG", "Quadrature"):
            solver, _, _ = TestFluidCore.build_fluid(
                meshes, rotation=True, internal_variable_family=family,
                dtn_representation="multiplier")
            blocks.append(real_block(solver))
        dg, quadrature = blocks
        assert np.abs(dg).max() > 0.0
        assert np.abs(dg - quadrature).max() <= 1e-14 * np.abs(dg).max()


class TestBlockOneIsTheDiagonalItClaims:
    """The assembled `Real` block against its derived diagonal, entry by entry.

    The whole block is recovered, not just its diagonal, because a claim that
    a matrix "is" a diagonal is two claims and the second one - that nothing
    else is there - is the one nobody had checked. The block is that of the
    multiplier representation, which has one `Real` row per DtN mode; it does
    not depend on the history family, because eliminating the history touches
    no constraint row.

    Three things are asserted and they are not equally strong:

    1. **The diagonal.** `theta_psi * (-scale_k * A_h)` per multiplier and
       `theta_rot_i * K_i` per rotation row (`predicted_real_diagonal`),
       against the assembled entry. Measured 2.673e-15 relative with rotation,
       2.045e-16 without.

       **The rotation row is already the worst row here, by an order**
       (2.664535e-15 against 2.0447e-16 for the multipliers), and that is not
       noise: the rotation row divides by a separately assembled mantle
       volume, so its diagonal carries the ratio of two assemblies of the same
       integral under different quadrature rules. At production size the ratio
       is much further from 1: the rotation row measured **8.8775e-11** on
       the coarse Spada mesh, against 1.2735e-15 for the 72 multiplier rows
       there - so **`DIAG_TOL` at 1e-12 holds on these meshes and does not
       hold at scale.** A bound on the rotation rows at production size must
       be looser than `DIAG_TOL`.
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
       zips `form.multiplier_keys` against the multiplier fields to attribute
       each solved trace coefficient to a mode, so a shuffle gives a silently
       wrong geoid spectrum with a perfectly plausible magnitude.

    What none of it establishes is recorded in
    `NOTES/measurements/PHASE1-DIAGONAL-GATE.md`; the short version is that a
    flip of `CLOSURE_SIGNS` is invisible here, because the rotation entry is
    `theta_rot_i * K_i` and both factors carry `s_i`, so a flip moves the
    prediction and the assembled row together.
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
        solver, _, _ = build(meshes, rotation=rotation, truncation=truncation,
                             dtn_representation="multiplier")
        block = real_block(solver)
        claimed = predicted_real_diagonal(solver)

        assert len(claimed) == len(solver.real_indices)
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
        that the production residual does carry `Real x Real` terms - the
        multiplier rows and the rotation closure row are both such terms, and
        they are the diagonal this class gates.
        """
        solver, _, _ = build(meshes, rotation=True,
                             dtn_representation="multiplier")
        z = solver.solution
        Z = solver.solution_space
        real = solver.real_indices
        n_mult = len(solver.multiplier_indices)
        dx_parent = fd.dx(domain=solver.potential_mesh)
        volume = fd.assemble(fd.Constant(1.0) * dx_parent)

        trials, tests = fd.split(z), fd.TestFunctions(Z)
        source = trials[real[0]] / volume
        planted = (solver.F
                   + PLANTED_EPS * source * tests[real[1]] * dx_parent
                   + PLANTED_EPS * source * tests[real[n_mult]] * dx_parent)

        clean = real_block(solver)
        assert np.all(np.abs(np.diag(clean)) > 0.0), (
            "a zero diagonal would make offdiagonal_ratios inf or nan rather "
            "than fail, so it is checked before the ratio is taken")
        assert offdiagonal_ratios(clean).max() <= OFFDIAG_TOL
        # Against the assembly's own reproducibility, measured here rather than
        # asserted as bit zero: `F(0)` is 3.388e-21 on these rows even in
        # serial, it merely reproduces, and in parallel it does not.
        floor = assembly_floor(solver)
        assert abs(clean[1, 0]) <= floor and abs(clean[n_mult, 0]) <= floor
        assert floor < 1e-3 * PLANTED_EPS, (
            f"the assembly floor {floor:.3e} is not comfortably below the "
            f"plant {PLANTED_EPS:.3e}, so the partner proves nothing")

        block = real_block(solver, F=planted)
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
        assert np.max(np.abs(predicted_real_diagonal(solver) - np.diag(block))
                      / np.abs(np.diag(block))) <= DIAG_TOL

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
        solver, _, _ = build(meshes, rotation=True)
        Z = solver.solution_space
        comm = Z.mesh().comm
        i_R = solver.real_indices[0]
        block1 = real_block_index_sets(solver)
        block0 = Z.field_ises[:i_R]

        assert len(block1) == len(solver.real_indices)
        merged = np.concatenate([iset.getIndices() for iset in block1]) \
            if block1 else np.array([], dtype=int)
        assert np.all(np.diff(merged) > 0), (
            "the merged block-1 index set is not increasing, so it is not the "
            "in-order concatenation the matrix-free submatrix extraction needs")
        # One global degree of freedom per Real sub-field, summed over ranks.
        assert comm.allreduce(len(merged)) == len(solver.real_indices)

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

        **The sequence is written out as a literal, deliberately.** An
        expectation built by iterating `dtn_boundaries` and `dtn.modes(...)`
        would re-execute the loop `boundary_bilinear` uses and compare it with
        itself. That catches a post-hoc mutation of `multiplier_keys` and
        cannot catch a reordering of either loop, because the expectation
        reorders with it. For an *ordering*, a literal is a specification and
        not a golden number, so it does not fall foul of this file's rule
        against stored values: exterior boundary first, `cos` before `sin`
        within an order, orders ascending, then the interior boundary with its
        `mean` mode leading.
        """
        solver, _, _ = build(meshes, rotation=True,
                             dtn_representation="multiplier")
        assert solver.form.multiplier_keys == [
            (CURVE_OUTER, "cos1"), (CURVE_OUTER, "sin1"),
            (CURVE_OUTER, "cos2"), (CURVE_OUTER, "sin2"),
            (CURVE_OUTER, "cos3"), (CURVE_OUTER, "sin3"),
            (CURVE_INNER, "mean"),
            (CURVE_INNER, "cos1"), (CURVE_INNER, "sin1"),
            (CURVE_INNER, "cos2"), (CURVE_INNER, "sin2"),
            (CURVE_INNER, "cos3"), (CURVE_INNER, "sin3"),
        ]
        assert len(solver.form.multiplier_keys) == len(solver.multiplier_indices)

    @pytest.mark.parametrize("truncation", [3, 5])
    def test_each_multiplier_row_is_the_mode_its_key_names(self, meshes,
                                                           truncation):
        """The alignment assertion proper, and the one the diagonal cannot make.

        `check_multiplier_alignment` carries the argument and the measurements.
        Both truncations, because that is the axis a zip-length error lives on,
        because 5 is production's `L`, and because the analytic deviation this
        test deliberately does *not* gate on grows 7.6x between them - which is
        how an earlier version of this test came to sit 6% inside its own
        threshold at truncation 5 without anyone noticing.
        """
        solver, _, _ = build(meshes, rotation=True, truncation=truncation,
                             dtn_representation="multiplier")
        worst = check_multiplier_alignment(solver)
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
    no mesh file, and the same mesh carries the displacement and the potential
    rather than a `Submesh`. That makes the mechanics and the potential share a
    domain, which the production geometry does not - the stand-off buffer is the
    whole point there - but nothing this class asserts is about the buffer.

    It warns, and the warnings are expected: `L h_max/R = 0.72` on a level-1
    cubed sphere is too coarse for the boundary rule to resolve `L = 1`, so
    `DtNGravityForm` caps the quadrature degree and says so; and the top
    traction is a `normal_stress` with no mass, which the solver accepts with a
    warning. Nothing here is a statement about the boundary integrals'
    accuracy - the diagonal is `-scale_k A_h` with `A_h` the *discrete*
    boundary measure, whatever that measure happens to be on this mesh.
    """

    @staticmethod
    def build(L=1, refinement_level=1, fluid_core=False):
        """The shell solver: DG history, multiplier DtN, rotation with `C - A = 0.1 C`.

        Args:
          L: the `SphericalDtN` truncation on both shell surfaces.
          refinement_level: the refinement of the `CubedSphereMesh` base.
          fluid_core: replace the rigid core (`un = 0` on the bottom) by a
            `FluidCore` there.

        Returns:
          `(solver, u, psi)`.
        """
        from gadopt import SphericalDtN

        base = fd.CubedSphereMesh(radius=1.0,
                                  refinement_level=refinement_level, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False

        u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
        X = fd.SpatialCoordinate(mesh)
        C = fd.assemble(fd.dot(X, X) * fd.dx(domain=mesh))
        bcs = {"top": {"normal_stress": SIGMA_HAT * X[2]}}
        couplings = [
            # Both moments, and it has to be both: the closure reads `C_minus_A`
            # for the polar-wander pair and `C` for `m_3`. 0.1 C is not a
            # physical dynamical ellipticity - it is a number that makes the
            # two closure constants distinguishable in the diagonal, which is
            # the point.
            Rotation(C=C, C_minus_A=0.1 * C, Omega_sq=OMEGA_SQ_EARTH)]
        if fluid_core:
            couplings.append(FluidCore(boundary="bottom", rho_core=2.0))
        else:
            bcs["bottom"] = {"un": 0.0}
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(), dt=1.0,
            gravity_bcs={"top": {"dtn": SphericalDtN(L)},
                         "bottom": {"dtn": SphericalDtN(L)}},
            bcs=bcs, couplings=couplings, dtn_representation="multiplier")
        return solver, u, psi

    def test_b1_near_nullspace_composes_with_core_pressure(self):
        """B1's per-field basis loop includes the trailing core-pressure field."""
        solver, _, _ = self.build(fluid_core=True)
        Z = solver.solution_space
        index = solver.field_index
        V = Z.sub(index["displacement"])
        rbm = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])
        basis = fd.MixedVectorSpaceBasis(
            Z, [rbm if i == index["displacement"] else Z.sub(i)
                for i in range(len(Z))])
        entries = list(basis)
        assert len(entries) == len(Z)
        assert entries[index["displacement"]] is rbm
        assert entries[index["core_pressure"]] == Z.sub(index["core_pressure"])

    def test_the_case_is_the_small_one(self):
        """A tripwire on the cost, not a statement about the discretisation.

        24 columns of two layers (48 cells) and 1971 degrees of freedom outside
        the DG history field (the displacement, the potential and the 11
        `Real` fields), and 2304 in the history: one symmetric `3 x 3` tensor
        (6 stored components) on the 8 nodes of a DQ1 hexahedron, in each of
        the 48 cells. If a future edit makes this a refinement level larger
        it stops being a unit test, and the rule this class is written under
        stops holding.

        The cell count is summed over `cells.owned.local_size`, the *owned*
        count, and not `num_cells()`, which is rank-local and includes the
        halo, so the obvious spelling is a test that passes only in serial.
        On an extruded mesh `cells` counts every cell of every layer.
        """
        solver, _, _ = self.build()
        history = solver.solution_space[solver.field_index["history"]].dim()
        assert solver.solution_space.dim() - history == 1971
        assert history == 6 * 8 * 48
        assert solver.mesh.comm.allreduce(solver.mesh.cells.owned.local_size) == 48
        assert len(solver.real_indices) == 11
        # (L+1)^2 = 4 modes on each of two boundaries, then m_1, m_2, m_3.
        assert len(solver.multiplier_indices) == 8
        assert set(coupling(solver, Rotation).values()) == {"m1", "m2", "m3"}

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
        solver, _, _ = self.build()
        block = real_block(solver)
        claimed = predicted_real_diagonal(solver)
        assert np.max(np.abs(claimed - np.diag(block))
                      / np.abs(np.diag(block))) <= DIAG_TOL
        assert offdiagonal_ratios(block).max() <= OFFDIAG_TOL
        # As in 2-D: bit zero, because the form contains no such term.
        off = block.copy()
        np.fill_diagonal(off, 0.0)
        if SERIAL:
            assert np.abs(off).max() == 0.0
        # Two distinct multiplier values, one per boundary, from 8 rows.
        n_mult = len(solver.multiplier_indices)
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
        solver, _, _ = self.build(L=L, refinement_level=refinement)
        assert len(solver.multiplier_indices) == 2 * (L + 1) ** 2
        block = real_block(solver)
        n_mult = len(solver.multiplier_indices)
        assert len(set(np.round(np.diag(block)[:n_mult], 12))) == 2, (
            "the premise of this test: the diagonal cannot tell the modes "
            "apart, so if it suddenly can, read this docstring again")
        check_multiplier_alignment(solver, silent_tol=silent_tol)

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
        solver, _, _ = self.build()
        block = real_block(solver)
        n_mult = len(solver.multiplier_indices)
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

    The string was once accepted, validated against the allowed set, and then
    ignored: every branch handed back the direct dictionary, whose own
    docstring says "2-D ONLY. Do not use this in 3-D". The failure is
    invisible in 2-D (where direct is right anyway) and in any driver that
    passes an explicit dictionary, so nothing caught it. These tests exist so
    it cannot recur silently: each asserts a key that only one of the two
    presets carries.
    """

    def test_iterative_selects_the_iterative_preset(self, meshes):
        solver, _, _ = build(meshes, solver_parameters="iterative")
        p = solver.solver_parameters
        # LU on block 0 is the direct preset's signature and appears nowhere
        # in the iterative one, whose block 0 is a python preconditioner.
        assert p["dtn_fieldsplit_0_pc_type"] == "python"
        assert "dtn_fieldsplit_0_assembled_pc_type" not in p

    def test_direct_still_selects_the_direct_preset(self, meshes):
        solver, _, _ = build(meshes, solver_parameters="direct")
        p = solver.solver_parameters
        assert p["dtn_fieldsplit_0_assembled_pc_type"] == "lu"

    def test_the_two_presets_actually_differ(self, meshes):
        """The rejecting partner: a test that passes on broken wiring is useless.

        With the wiring broken, both strings produced byte-identical
        dictionaries and every assertion above would have held for the wrong
        reason.
        """
        a, _, _ = build(meshes, solver_parameters="direct")
        b, _, _ = build(meshes, solver_parameters="iterative")
        assert a.solver_parameters != b.solver_parameters

    def test_the_default_is_direct_in_two_dimensions(self, meshes):
        """Dimension-aware, mirroring `GravitySolver` and `StokesSolverBase`.

        These meshes are 2-D, so the default must still be the direct preset
        and no existing 2-D result moves. The 3-D half of the rule cannot be
        exercised here - solving a 3-D coupled system is not a unit test.
        """
        solver, _, _ = build(meshes)
        assert solver.mesh.topological_dimension == 2
        assert solver.solver_parameters["dtn_fieldsplit_0_assembled_pc_type"] \
            == "lu"

    def test_the_block_zero_sweep_follows_the_layout_not_the_caller(
            self, meshes):
        """The block-0 split of a dictionary matches the space's block-0 fields.

        The preset's `condensed` and the space's history family are independent
        arguments that nothing reconciles when the caller gives a dictionary.
        For the DG family's `block0="pair"` route, the split count must match
        the space's block-0 field count: the displacement, the history and the
        potential.

        The route with a block-0 fieldsplit is `block0="pair"`; the default
        route names no `_fields` key at all and is guarded by the class name
        instead (`tests/unit/test_gia_condensed_block0.py`).
        """
        solver, _, _ = build(
            meshes,
            dtn_representation="multiplier",
            solver_parameters=selfgrav_dtn_iterative_solver_parameters(
                condensed=False, block0="pair",
                dtn_representation="multiplier"))
        # A split may name a comma-separated pair (the condensed `(u, M)`
        # split), so count the fields named, not the splits.
        n_fields = sum(
            len(str(v).split(","))
            for k, v in solver.solver_parameters.items()
            if k.startswith("dtn_fieldsplit_0_pc_fieldsplit_")
            and k.endswith("_fields"))
        assert n_fields == 2 + ("history" in solver.field_index)

    def test_the_displacement_split_gets_the_near_incompressible_pc(
            self, meshes):
        """Naming `firedrake.AssembledPC` here is the defect, not a milder choice.

        A `near_nullspace` declared on the outer mixed space never reaches
        GAMG underneath `DtNTwoBlockSchurPC`; the modes are dropped with no
        error and no warning. `b4_polar_motion` shipped exactly that. With the
        history eliminated at quadrature points, the displacement split
        carries the class that builds the modes itself.

        The class it carries is the near-incompressible one, which adds the
        low-degree divergence-free fields to the six rigid-body modes. The
        volumetric penalty of the internal-variable stress puts the slow modes
        of the displacement block in the divergence-free space, and GAMG
        coarsens onto the modes it is handed and no others. Measured on Gadi
        at 500 yr steps, 96 ranks, bulk/shear 100: rigid modes 33 699 GAMG
        V-cycles with 64 of 124 block-0 solves at the cap and 565 s for the
        warm step (job 178765557); this class 16 853 V-cycles, 8 of 102 at the
        cap, 270 s (job 178765558).

        The split runs one GAMG V-cycle, written as `ksp_type preonly` with
        no tolerance and no cap attached. This is the solver's own path, the
        one a caller reaches by asking for `"iterative"` and naming nothing
        else, so this is where the preset's `u_ksp_max_it` default is pinned.
        Measured end to end on Gadi against four iterations of CG on the same
        split, both Spada cases over the full 138-step ladder from the preset
        alone: 1 h 18 32 against 2 h 32 15 on the polar-motion case (job
        179496714 against 179423870) and 1 h 08 54 against 1 h 39 01 on the
        cap case (179483713 against 179423871), with every printed benchmark
        number unchanged. A cap left behind on a `preonly` KSP would be inert
        and would read as a truncated solve to anyone auditing the dictionary,
        so the three keys of the CG must be absent and not merely unused.

        The Quadrature family supports natural displacement boundaries only,
        so this configuration has a fluid core instead of the rigid one; the
        multiplier representation gives it the `(u, psi)` block-0 sweep.
        """
        solver, _, _ = TestFluidCore.build_fluid(
            meshes, internal_variable_family="Quadrature",
            dtn_representation="multiplier", solver_parameters="iterative")
        p = solver.solver_parameters
        u_split = next(
            s for s in range(3)
            if p.get(f"dtn_fieldsplit_0_pc_fieldsplit_{s}_fields") == "0")
        prefix = f"dtn_fieldsplit_0_fieldsplit_{u_split}_"
        assert p[prefix + "pc_python_type"] \
            == "gadopt.NearlyIncompressibleAssembledPC"
        assert p[prefix + "ksp_type"] == "preonly"
        assert prefix + "ksp_max_it" not in p
        assert prefix + "ksp_rtol" not in p
        assert prefix + "ksp_converged_maxits" not in p

    def test_the_uncondensed_pair_split_condenses_and_seeds_near_incompressible_modes(
            self, meshes):
        """With `block0="pair"` the DG family's `(u, M)` is one split.

        The condensed displacement operator is an assembled matrix, so no
        `AssembledPC` wraps it; GAMG is seeded through the near-nullspace
        provider the solver publishes, whose modes come from
        `near_nullspace`.

        Both routes get the same modes and the same displacement split, so
        that the wall-clock ratio between them measures the cost of the
        coupled residual and nothing else: 670 s against 271 s for a 500 yr
        step at matched settings (Gadi jobs 178765563 and 178765560, both with
        the four-iteration CG this preset no longer writes by default).

        The split is `u_ksp_max_it`, one argument for both block-0 routes, so
        the default reaches this route too and the assertions below are the
        same three absences as on the condensed route. A second default for
        this route would break the comparison the route exists for.
        """
        solver, _, _ = build(
            meshes,
            dtn_representation="multiplier",
            solver_parameters=selfgrav_dtn_iterative_solver_parameters(
                condensed=False, block0="pair",
                dtn_representation="multiplier"))
        p = solver.solver_parameters
        prefix = "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"
        assert p["dtn_fieldsplit_0_pc_fieldsplit_0_fields"] == "0,1"
        assert p["dtn_fieldsplit_0_pc_fieldsplit_1_fields"] == "2"
        assert p["dtn_fieldsplit_0_fieldsplit_0_pc_python_type"] \
            == "gadopt.InternalVariableSCPC"
        assert p["dtn_fieldsplit_0_fieldsplit_0_pc_sc_eliminate_fields"] == "1"
        assert p[prefix + "ksp_type"] == "preonly"
        assert prefix + "ksp_max_it" not in p
        assert prefix + "ksp_rtol" not in p
        assert prefix + "ksp_converged_maxits" not in p
        assert solver.near_nullspace_modes == "incompressible"
        assert callable(solver.appctx["condensed_field_near_nullspace"])
        assert solver.appctx["operator_version"] == 0

    def test_u_pc_is_refused_on_the_uncondensed_layout(self):
        with pytest.raises(ValueError, match="u_pc has no meaning"):
            selfgrav_dtn_iterative_solver_parameters(
                condensed=False, u_pc="firedrake.AssembledPC")

    # The displacement split sits at a different depth on each route: split 0
    # of the block-0 sweep with the history condensed, split 0 of
    # `gadopt.CondensedBlockPC`'s own `(u, psi)` sweep on the full layout with
    # `block0="condensed"`, and split 0 of the `(u, M)` pair's condensed field
    # on the full layout with `block0="pair"`.
    U_SPLIT_PREFIXES = (
        ({"condensed": True}, "dtn_fieldsplit_0_fieldsplit_0_"),
        ({"condensed": False}, "dtn_fieldsplit_0_condensed_fieldsplit_0_"),
        ({"condensed": False, "block0": "pair",
          "dtn_representation": "multiplier"},
         "dtn_fieldsplit_0_fieldsplit_0_condensed_field_"),
    )

    def test_the_displacement_split_defaults_to_one_v_cycle(self):
        """The sixth preset default, on all three block-0 routes.

        The split is one GAMG V-cycle per block-0 iteration, written as
        `ksp_type preonly`. The three keys of the truncated CG must be absent
        and not merely unused: a cap or a tolerance left behind on a `preonly`
        KSP is inert and would read as a truncated solve to anyone auditing
        the dictionary.

        Measured on Gadi from the preset alone, both Spada cases over the full
        138-step ladder, against four iterations of CG on the same split:
        1 h 18 32 against 2 h 32 15 (job 179496714 against 179423870) and
        1 h 08 54 against 1 h 39 01 (179483713 against 179423871), with every
        printed benchmark number unchanged.
        """
        for kwargs, prefix in self.U_SPLIT_PREFIXES:
            p = selfgrav_dtn_iterative_solver_parameters(**kwargs)
            assert p[prefix + "ksp_type"] == "preonly", kwargs
            assert prefix + "ksp_max_it" not in p, kwargs
            assert prefix + "ksp_rtol" not in p, kwargs
            assert prefix + "ksp_converged_maxits" not in p, kwargs

    def test_the_truncated_cg_is_selectable(self):
        """`u_ksp_max_it=4` is the alternative, on all three routes.

        It runs four iterations of CG at `u_ksp_rtol` and counts the cap as
        convergence, which is what `ksp_converged_maxits` says. It was the
        preset's default from commit 77af62b to c48564a2, and the campaign of
        2026-09-20 measured most of its arms on it, so a caller reproducing one
        of those runs asks for it by name. Older runs, the P3 march and job
        178765558 among them, ran the single V-cycle this preset now defaults
        to.
        """
        for kwargs, prefix in self.U_SPLIT_PREFIXES:
            p = selfgrav_dtn_iterative_solver_parameters(
                u_ksp_max_it=4, **kwargs)
            assert p[prefix + "ksp_type"] == "cg", kwargs
            assert p[prefix + "ksp_max_it"] == 4, kwargs
            assert p[prefix + "ksp_rtol"] == 1e-2, kwargs
            assert prefix + "ksp_converged_maxits" in p, kwargs

    def test_the_default_is_the_dictionary_u_ksp_max_it_zero_writes(self):
        """Asking for `0` by hand and asking for nothing give one dictionary.

        This is what makes every measurement taken with `u_ksp_max_it=0`
        passed by hand, the two full runs of the Spada benchmark included, a
        measurement of the default.
        """
        for kwargs, _ in self.U_SPLIT_PREFIXES:
            assert (selfgrav_dtn_iterative_solver_parameters(**kwargs)
                    == selfgrav_dtn_iterative_solver_parameters(
                        u_ksp_max_it=0, **kwargs)), kwargs

    def test_the_iterative_preset_solves_the_same_system(self, meshes, solved):
        """The one that is not about dictionaries: does the iterative path solve?

        Wiring a preset up is worth nothing if the path it reaches does not
        converge to the same answer. `solved` is the module fixture, which
        takes the default, which in 2-D is direct.

        The tolerance is the iterative preset's own `outer_rtol` of 1e-6 and
        not a fudge: measured on this system the disagreement is 1.9e-06 at
        1e-6, 1.5e-08 at 1e-8 and 1.0e-10 at 1e-10, i.e. it tracks the Krylov
        tolerance four orders down. That is what says the two presets solve one
        discrete system rather than two close ones - a formulation difference
        would sit on a floor and refuse to move.
        """
        reference, u_ref, psi_ref = solved
        solver, u, psi = build(meshes, solver_parameters="iterative")
        solver.solve()

        for a_field, b_field in ((u_ref, u), (psi_ref, psi)):
            a = a_field.dat.data_ro
            b = b_field.dat.data_ro
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
        # On the default block-0 route the block-0 KSP is `preonly` and the
        # solve that does the work is `gadopt.CondensedBlockPC`'s own
        # `(u, psi)` Krylov solve, so that is the line the Gadi counters read
        # one block-0 application from.
        for key in ("ksp_converged_reason",
                    "snes_converged_reason",
                    "dtn_fieldsplit_0_condensed_ksp_converged_reason"):
            assert key in p
        # Block 1 is the exception, and only while the cached apply owns it:
        # `gadopt.DtNTwoBlockSchurPC` then applies the dense factors of the
        # exact complement itself and the block-1 KSP is never entered, so a
        # converged reason there would describe a solve that never runs. The
        # preset writes the key wherever that KSP is real, which is every
        # configuration that leaves block 1 to PETSc.
        assert p["dtn_schur_ainvb"] is True
        assert "dtn_fieldsplit_1_ksp_converged_reason" not in p
        delegating, _, _ = build(
            meshes, solver_parameters="iterative",
            preset_options={"ainvb": False})
        assert "dtn_fieldsplit_1_ksp_converged_reason" \
            in delegating.solver_parameters


def source_mass_form(solver):
    """The volume source of the solver's potential row, against `v = 1`.

    The potential row is `GravityPotential.residual`; its volume source is
    `-Lambda int rho_0 u . grad(v) dx_m`, the only integral of the row on the
    mantle's cells. The row is built here at the solution's displacement with
    `v` a unit Function on the potential space, and the mantle cell integrals
    are kept.

    The constant is a `Function` assigned 1 and not a `Constant`: `grad` of a
    `Constant` folds to symbolic zero and the test would then be a statement
    about UFL rather than about the discretisation. This way the gradient is
    the real one, evaluated from a full set of unit degrees of freedom, and
    the zero it returns is the discrete statement that `v = 1` lies in the
    test space.

    Args:
      solver: the solver.

    Returns:
      `(form, one)`: the source form and the unit Function in it.
    """
    space = solver.psi.function_space()
    one = fd.Function(space).assign(1.0)
    zero = fd.Function(space)
    row = solver.gravity.residual(
        zero, one, solver.field("displacement"),
        density=solver.approximation.density, theta_psi=1.0)
    source = [integral for integral in row.integrals()
              if integral.integral_type() == "cell"
              and integral.ufl_domain() == solver.mesh]
    return fd.Form(source), one


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
        form, _ = source_mass_form(solver)
        value = fd.assemble(form)
        u = solver.field("displacement")
        scale = float(solver.Lambda) * fd.assemble(
            solver.approximation.density * fd.sqrt(fd.dot(u, u))
            * solver.gravity.dx_m)
        assert scale > 0.0, "the test is vacuous on a zero displacement"
        assert abs(value) < 1e-16 * scale

    def test_the_constant_is_a_real_function_not_a_ufl_constant(self, solved):
        """Otherwise the test is a statement about UFL's constant folding.

        `grad(Constant(1))` folds to symbolic zero and the integral disappears
        before anything is discretised. With a `Function` of unit degrees of
        freedom the source integral stays in the row, and the unit function is
        one of its coefficients.
        """
        solver, _, _ = solved
        form, one = source_mass_form(solver)
        assert form.integrals(), "the source-mass form folded away entirely"
        # No family filter: on a quadrilateral mesh the continuous element is
        # reported as "Q", not "Lagrange".
        unit = [c for c in extract_coefficients(form)
                if isinstance(c, fd.Function)
                and c.function_space() is solver.psi.function_space()
                and np.allclose(c.dat.data_ro, 1.0)]
        assert unit and unit[0] is one, \
            "no unit function on the potential space in the form"

    def test_net_enclosed_mass_is_the_sheets_alone(self, solved):
        """A `cos 2phi` sheet carries no net mass, so the datum is idle.

        The volume source contributes nothing by construction, which is the
        whole difference from `GravitySolver`'s mass bookkeeping. The enclosed
        mass is the datum's `source_mass` plus the cavity flux in mass units.
        """
        solver, _, _ = solved
        gravity = solver.gravity
        enclosed = (scalar_value(gravity.source_mass)
                    + scalar_value(gravity.cavity_flux)
                    / (4.0 * np.pi * scalar_value(gravity.form.G)))
        assert abs(enclosed) < 1e-14


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
        rotation = coupling(solver, Rotation)
        theta_rot = float(rotation.theta_rot(solver.coupling_context, 2))
        assert float(solver.theta_psi) == pytest.approx(
            NULL_COUPLING_ROW_SCALE / LAMBDA, rel=1e-14)
        assert theta_rot == pytest.approx(
            -NULL_COUPLING_ROW_SCALE * OMEGA_SQ_EARTH, rel=1e-14)
        assert float(solver.theta_psi) != 0.0
        assert theta_rot != 0.0

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
        solver, _, _ = build(meshes, approximation_kwargs={"B_mu": 0.0},
                             dtn_representation="multiplier")
        z = solver.solution
        index = solver.field_index
        A = fd.assemble(fd.derivative(solver.F, z), mat_type="nest").petscmat

        def amax(i, j):
            M = A.getNestSubMatrix(i, j)
            return 0.0 if M is None else float(
                np.abs(M.convert("dense").getDenseArray()).max())

        assert amax(index["potential"], index["potential"]) > 0.0
        assert amax(solver.multiplier_indices[0], index["potential"]) > 0.0
        # The coupling itself *is* off: the body force is gone.
        assert amax(index["displacement"], index["potential"]) == 0.0
        # ... while the one-way leg survives, so psi is still driven by u.
        assert amax(index["potential"], index["displacement"]) > 0.0

    def test_it_reproduces_the_uncoupled_solver(self, meshes):
        """V2, run as specified rather than around the defect.

        Both sides declare the rigid-rotation kernel (defect D-2); without it
        the displacements agree only to ~1e-07, the difference being entirely
        that zero-energy mode.

        The load is the traction of `mechanics_bcs` on both sides
        (`sheet=False`). A `SurfaceLoad` weighs its sheet with the
        approximation's `B_mu`, which is zero here, so it would apply no
        traction at all; the mechanics-only reference has the traction
        `B_MU g sigma`.
        """
        _, sub = meshes
        solver, u, psi = build(
            meshes, sheet=False, approximation_kwargs={"B_mu": 0.0},
            declare_nullspace=True)
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
        # `apply_kernel_projection` makes the PETSc call of
        # `basis.orthogonalize` with a read-write access to the vector.
        apply_kernel_projection(basis, zm.subfunctions[0])

        ur = zm.subfunctions[0]
        assert fd.norm(ur) > 0.0
        assert fd.norm(fd.assemble(u - ur)) / fd.norm(ur) < 1e-11
        # The gravity space stores the history as one (n, d, d) field with
        # symmetric storage; the reference holds a (d, d) field. Compare
        # slice by slice as expressions.
        stored = history_slices(
            solver.solution.subfunctions[solver.field_index["history"]])
        for k, m in enumerate(stored):
            mr = zm.subfunctions[1 + k]
            assert fd.norm(m - mr) / fd.norm(mr) < 1e-11
        # ... and psi is still driven, so the test is not vacuous.
        assert fd.norm(psi) > 0.0


class TestAcceptedStateAfterAStep:
    """The accepted state is the solved state when `solve` returns.

    Code that reads the state between two steps (a checkpoint, a diagnostic
    that assembles the residual) reads the accepted history level
    `solver.internal_variables` and the Functions `u` and `psi`. After a step
    each must hold exactly what the step solved, with and without a declared
    nullspace (the projection path and the path without it).
    """

    @pytest.mark.parametrize("declare_nullspace", [False, True])
    def test_the_accepted_state_is_the_solved_state(self, meshes,
                                                    declare_nullspace):
        solver, u, psi = build(meshes, dtn_representation="multiplier",
                               declare_nullspace=declare_nullspace)
        solver.solve()
        z = solver.solution
        # The solve moved away from the zero initial state, so equality is
        # not the trivial one of two zero vectors.
        with z.dat.vec_ro as current:
            assert current.norm() > 0.0
        index = solver.field_index
        pairs = ((solver.internal_variables, z.subfunctions[index["history"]]),
                 (u, z.subfunctions[index["displacement"]]),
                 (psi, z.subfunctions[index["potential"]]))
        for accepted, solved_field in pairs:
            with accepted.dat.vec_ro as old, solved_field.dat.vec_ro as current:
                assert current.norm() > 0.0
                difference = old.copy()
                difference.axpy(-1.0, current)
                assert difference.norm() == 0.0


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
        solver, _, _ = build(meshes)
        z = solver.solution
        X = fd.SpatialCoordinate(sub)
        probe = fd.Function(solver.solution_space)
        probe.subfunctions[solver.field_index["displacement"]].interpolate(
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
        """The declared kernel is one vector on the displacement and nothing else."""
        solver, _, _ = build(meshes, rotation=True, declare_nullspace=True)
        bases = list(solver.nullspace)
        displacement = solver.field_index["displacement"]
        assert len(bases) == len(solver.solution_space)
        assert isinstance(bases[displacement], fd.VectorSpaceBasis)
        assert len(bases[displacement]._petsc_vecs) == 1
        for i, entry in enumerate(bases):
            if i != displacement:
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

        def rotation_content(u):
            V = u.function_space()
            r0 = fd.Function(V).interpolate(fd.as_vector([-X[1], X[0]]))
            dxm = fd.Measure("dx", domain=sub)
            c = (fd.assemble(fd.dot(u, r0) * dxm)
                 / fd.assemble(fd.dot(r0, r0) * dxm))
            return abs(c) * fd.norm(r0) / fd.norm(u)

        plain, u_plain, _ = build(meshes)
        plain.solve()
        declared, u_declared, _ = build(meshes, declare_nullspace=True)
        declared.solve()

        assert rotation_content(u_declared) < 1e-14
        assert rotation_content(u_declared) < 1e-3 * rotation_content(u_plain)


class TestFluidCore:
    """`FluidCore`: the CMB condition that replaces the rigid `un = 0`.

    Structural only, as everything in this file is.
    """

    #: The keywords of `build_fluid` that go to `FluidCore`; the rest go to
    #: the solver.
    FLUID_CORE_KEYS = ("boundary", "rho_core", "rho_mantle", "g",
                       "buoyancy_density")

    @staticmethod
    def build_fluid(meshes, *, rotation=False, sheet=True, extra_bcs=None,
                    approximation_kwargs=None, **kwargs):
        """The annulus with a fluid core on Rc and the `cos 2 phi` load on Re.

        Args:
          meshes: the `(parent, sub)` pair of the module fixture.
          rotation: add the `Rotation` coupling with the mantle's polar moment.
          sheet: `True` gives the load as a `SurfaceLoad`; `False` gives no
            load, for a test that writes its own traction in `extra_bcs`.
          extra_bcs: mechanics conditions merged into `bcs`.
          approximation_kwargs: settings merged into `approximation`.
          kwargs: `boundary`, `rho_core`, `rho_mantle`, `g` and
            `buoyancy_density` go to `FluidCore` (defaults `CURVE_RC` and
            2.0); everything else goes to `SelfGravitatingGIASolver`
            (`dtn_representation`, `internal_variable_family`,
            `solver_parameters`, `nullspace`, ...).

        Returns:
          `(solver, u, psi)`.
        """
        parent, sub = meshes
        settings = {"boundary": CURVE_RC, "rho_core": 2.0}
        settings.update({k: kwargs.pop(k)
                         for k in TestFluidCore.FLUID_CORE_KEYS if k in kwargs})
        u, psi = new_fields(meshes)
        couplings = [FluidCore(**settings)]
        if sheet:
            couplings.append(SurfaceLoad(boundary=CURVE_RE,
                                         sigma=load_density(sub),
                                         g_surface=1.0))
        if rotation:
            couplings.append(Rotation(C=polar_moment(meshes),
                                      Omega_sq=OMEGA_SQ_EARTH))
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(**(approximation_kwargs or {})), dt=1.0,
            gravity_bcs=gravity_bcs(parent, sheet=False),
            bcs=dict(extra_bcs or {}), couplings=couplings, **kwargs)
        return solver, u, psi

    def test_refuses_un_on_the_same_boundary(self, meshes):
        """The two treatments are alternatives, not layers: `un` pins `dot(u, n)`."""
        with pytest.raises(ValueError, match="both a FluidCore and"):
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
        """No `FluidCore` means no coupling and no core-pressure field."""
        solver, _, _ = build(meshes, rotation=False)
        assert not [c for c in solver.couplings if isinstance(c, FluidCore)]
        assert "core_pressure" not in solver.field_index
        assert "core_pressure" not in solver.state()

    def test_core_pressure_is_a_parent_real_field(self, meshes):
        parent, _ = meshes
        solver, _, _ = self.build_fluid(meshes)
        pressure = coupling(solver, FluidCore).pressure()
        assert pressure is solver.solution.subfunctions[
            solver.field_index["core_pressure"]]
        assert pressure.function_space().mesh() is parent
        assert pressure.function_space().ufl_element().family() == "Real"

    @pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs global dense blocks")
    @REAL_BLOCK_ON_PARENT_FACETS
    def test_core_pressure_blocks_form_an_exact_saddle_pair(self, meshes):
        solver, _, _ = self.build_fluid(meshes)
        core = coupling(solver, FluidCore)
        A = fd.assemble(fd.derivative(core.residual(solver.coupling_context),
                                      solver.solution),
                        mat_type="nest").petscmat
        iu, ic = solver.field_index["displacement"], solver.field_index["core_pressure"]
        up = A.getNestSubMatrix(iu, ic).convert("dense").getDenseArray()
        pu = A.getNestSubMatrix(ic, iu).convert("dense").getDenseArray()
        pp = A.getNestSubMatrix(ic, ic)
        scale = max(np.abs(up).max(), np.abs(pu).max())
        assert scale > 0.0
        assert np.abs(up - pu.T).max() == 0.0
        if pp is not None:
            x, y = pp.createVecRight(), pp.createVecLeft()
            x.set(1.0)
            pp.mult(x, y)
            assert y.norm() == 0.0

    def test_constraint_row_has_the_documented_scale(self, meshes):
        solver, _, _ = self.build_fluid(meshes)
        core = coupling(solver, FluidCore)
        _, sub = meshes
        X = fd.SpatialCoordinate(sub)
        displacement = solver.solution.subfunctions[
            solver.field_index["displacement"]]
        displacement.interpolate(X / fd.sqrt(fd.dot(X, X)))
        residual = fd.assemble(core.residual(solver.coupling_context))
        row = float(residual.subfunctions[
            solver.field_index["core_pressure"]].dat.data_ro[0])
        flux = fd.assemble(
            fd.dot(displacement, fd.FacetNormal(solver.mesh)) * core.measure())
        row_scale = (float(solver.scaling_factor)
                     * float(solver.row_scale_B_mu))
        assert row / row_scale == pytest.approx(flux, rel=1e-13)

    @REAL_BLOCK_ON_PARENT_FACETS
    def test_constraint_survives_at_zero_B_mu(self, meshes):
        solver, _, _ = self.build_fluid(
            meshes, approximation_kwargs={"B_mu": 0.0})
        core = coupling(solver, FluidCore)
        A = fd.assemble(fd.derivative(core.residual(solver.coupling_context),
                                      solver.solution),
                        mat_type="nest").petscmat
        index = solver.field_index
        iu, ip, ic = (index["displacement"], index["potential"],
                      index["core_pressure"])

        def amax(i, j):
            block = A.getNestSubMatrix(i, j)
            if block is None:
                return 0.0
            return float(np.abs(
                block.convert("dense").getDenseArray()).max())

        assert amax(iu, ic) > 0.0
        assert amax(ic, iu) > 0.0
        assert amax(iu, ip) == 0.0
        assert amax(iu, iu) == 0.0

    @pytest.mark.xfail(
        # Not strict: the outcome depends on round-off in a singular block-0
        # LU, so a pass is as uninformative as a failure. With the dense Schur
        # complement on block 1 of the direct preset (2026-09-15) the test
        # passes on phoenix; with `pc_type none` there it failed. Neither
        # result shows that block 0 is non-singular.
        strict=False,
        reason=(
            "Fragile by construction: with a fluid core and B_mu = 0 nothing "
            "pins a rigid motion of the mantle, so the block-0 LU of the "
            "direct preset factors a singular matrix and the outer GMRES "
            "reaches 1e-14 or stalls at 1e-9 depending on round-off. A "
            "mathematically identical rewrite of deviatoric_strain on main "
            "(PR #540) flipped it from 11 iterations to a stall. Declaring "
            "the kernel and enabling MUMPS null-pivot detection makes the "
            "solve deterministic but leaves the flux at 3.7e-10 and the "
            "pressure 2.6e-7 relative from the reference, outside these "
            "tolerances. Needs a non-singular block 0 (pin the rigid modes "
            "in the residual, or compare at B_mu > 0). Recorded 2026-09-10; "
            "read NOTES/HANDOVER.md 4.3."
        ),
    )
    def test_zero_B_mu_matches_independently_constrained_mechanics(
            self, meshes):
        """The zero-coupling solve keeps the physical core constraint.

        The load is a traction alone (`sheet=False` and a `normal_stress`),
        because a `SurfaceLoad` weighs its sheet with the approximation's
        `B_mu`, which is zero here.
        """
        _, sub = meshes
        X = fd.SpatialCoordinate(sub)
        load = B_MU * SIGMA_HAT * (
            fd.cos(2 * fd.atan2(X[1], X[0])) + fd.Constant(0.25))
        surface_bcs = {CURVE_RE: {"normal_stress": load}}
        solver, u, _ = self.build_fluid(
            meshes, sheet=False, approximation_kwargs={"B_mu": 0.0},
            dtn_representation="multiplier", extra_bcs=surface_bcs)
        solver.solve()

        V = fd.VectorFunctionSpace(sub, "CG", 2)
        S = fd.TensorFunctionSpace(sub, "DG", 1)
        Zm = fd.MixedFunctionSpace([V, S])
        zm = fd.Function(Zm)
        pressure = fd.Constant(0.0)
        w = fd.TestFunctions(Zm)[0]
        n = fd.FacetNormal(sub)
        dss = fd.Measure("ds", domain=sub)(CURVE_RC)
        reference = CoupledInternalVariableSolver(
            zm, approximation(B_mu=0.0), dt=1.0, bcs=surface_bcs,
            additional_forcing_term=pressure * fd.dot(w, n) * dss,
            solver_parameters="direct")

        def solve_at(value):
            pressure.assign(value)
            zm.assign(0.0)
            reference.solution_old.assign(0.0)
            reference.solve()
            return zm.copy(deepcopy=True)

        base = solve_at(0.0)
        direction = solve_at(1.0)
        direction -= base
        base_flux = fd.assemble(fd.dot(base.subfunctions[0], n) * dss)
        response_flux = fd.assemble(
            fd.dot(direction.subfunctions[0], n) * dss)
        assert abs(response_flux) > 1e-12
        reference_pressure = -base_flux / response_flux
        constrained = base.copy(deepcopy=True)
        for target, base_field, direction_field in zip(
                constrained.subfunctions, base.subfunctions,
                direction.subfunctions):
            with target.dat.vec_wo as target_vec, \
                    base_field.dat.vec_ro as base_vec, \
                    direction_field.dat.vec_ro as direction_vec:
                target_vec.waxpy(
                    reference_pressure, direction_vec, base_vec)

        flux = fd.assemble(fd.dot(u, n) * dss)
        flux_ref = fd.assemble(
            fd.dot(constrained.subfunctions[0], n) * dss)
        assert abs(flux) < 1e-12
        assert abs(flux_ref) < 1e-12
        assert float(coupling(solver, FluidCore).pressure()) == pytest.approx(
            reference_pressure, rel=1e-11, abs=1e-13)

        dxm = fd.Measure("dx", domain=sub)
        displacement_error = u - constrained.subfunctions[0]
        error_strain = fd.sym(fd.grad(displacement_error))
        reference_strain = fd.sym(fd.grad(constrained.subfunctions[0]))
        relative_strain_error = np.sqrt(fd.assemble(
            fd.inner(error_strain, error_strain) * dxm) / fd.assemble(
                fd.inner(reference_strain, reference_strain) * dxm))
        assert relative_strain_error < 1e-10
        m, = history_slices(
            solver.solution.subfunctions[solver.field_index["history"]])
        m_ref = constrained.subfunctions[1]
        assert fd.norm(m - m_ref) / fd.norm(m_ref) < 1e-10

    def test_rigid_rotation_nullspace_composes_with_core_pressure(self, meshes):
        solver, _, _ = self.build_fluid(meshes, nullspace="rigid_rotation")
        basis = list(solver.nullspace)
        index = solver.field_index
        assert len(basis) == len(solver.solution_space)
        assert isinstance(basis[index["displacement"]], fd.VectorSpaceBasis)
        assert not isinstance(basis[index["core_pressure"]], fd.VectorSpaceBasis)

    def test_solve_enforces_zero_integrated_cmb_displacement(self, meshes):
        solver, u, _ = self.build_fluid(meshes)
        solver.solve()
        dss = coupling(solver, FluidCore).measure()
        un = fd.dot(u, fd.FacetNormal(solver.mesh))
        flux = fd.assemble(un * dss)
        scale = np.sqrt(fd.assemble(un * un * dss)
                        * fd.assemble(fd.Constant(1.0) * dss))
        assert fd.norm(u) > 0.0
        assert abs(flux) <= 1e-10 * scale + 1e-13

    @REAL_BLOCK_ON_PARENT_FACETS
    def test_the_pair_transposes_exactly(self, meshes):
        """One energy, two variations: the blocks are transposes by construction.

        Exactly, and not to a tolerance - `derivative` of a single scalar form
        is the only spelling in which this cannot be got wrong, which is why
        the condition is written that way.
        """
        solver, _, _ = self.build_fluid(meshes)
        core = coupling(solver, FluidCore)
        A = fd.assemble(fd.derivative(core.residual(solver.coupling_context),
                                      solver.solution),
                        mat_type="nest").petscmat
        iu, ip = solver.field_index["displacement"], solver.field_index["potential"]
        upsi = A.getNestSubMatrix(iu, ip).convert("dense").getDenseArray()
        psiu = A.getNestSubMatrix(ip, iu).convert("dense").getDenseArray()
        assert np.abs(upsi).max() > 0.0
        assert np.abs(upsi - psiu.T).max() == 0.0

    def test_the_spring_carries_rho_core_not_the_contrast(self, meshes):
        """The buoyancy (u,u) block scales as `rho_core`, not the contrast.

        The prestress volume term supplies the mantle's `-0.5 rho_0 g u_r^2`
        half at the CMB (measured to 2.9e-15,
        `NOTES/measurements/cmb_prestress_check.py`), so the fluid-core
        energy's spring must carry `rho_core` ALONE for the assembled net to be
        the physical contrast. `buoyancy_density="contrast"` is the 7.3x-too-soft
        coefficient, kept only for baselines. With `rho_core=3`, `rho_mantle=1`
        the isolated block's core:contrast ratio is exactly `3 : (3-1)` = 1.5;
        every geometric and `B_mu`/`g_0` factor cancels in the ratio.
        """
        _, sub = meshes

        def spring_quadratic(mode):
            solver, _, _ = self.build_fluid(
                meshes, rho_core=3.0, rho_mantle=1.0, buoyancy_density=mode)
            z = solver.solution
            X = fd.SpatialCoordinate(sub)
            z.subfunctions[solver.field_index["displacement"]].interpolate(
                X / fd.sqrt(fd.dot(X, X)))         # radial u; psi stays zero
            fc = coupling(solver, FluidCore).residual(solver.coupling_context)
            # z^T J z with psi(z)=0 isolates the (u,u) buoyancy block.
            return fd.assemble(fd.action(
                fd.action(fd.derivative(fc, z), z), z))

        core = spring_quadratic("core")
        contrast = spring_quadratic("contrast")
        assert core / contrast == pytest.approx(3.0 / 2.0, rel=1e-4)

    def test_buoyancy_density_rejects_an_unknown_mode(self, meshes):
        """A typo in the switch must fail loudly, not silently pick a default.

        `FluidCore` checks the switch when it is built, before any form.
        """
        with pytest.raises(ValueError, match="'core' or 'contrast'"):
            self.build_fluid(meshes, buoyancy_density="nonsense")

    def test_the_measure_is_the_cmb_circle(self, meshes):
        """`2 pi Rc`, measured on the facets the condition actually integrates.

        The measure is the mantle's own `ds` intersected with the parent's
        `dS`: a facet integral must be paired with a *facet* measure, and
        pairing it with the parent's cell measure evaluates the parent's field
        at the wrong points with no warning at all.
        """
        solver, _, _ = self.build_fluid(meshes)
        length = fd.assemble(
            fd.Constant(1.0) * coupling(solver, FluidCore).measure())
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
        against that, and against the same form with the core sheet taken out
        of the context's list of sheets, which is what the rotation row reads.
        """
        rho_core = 2.0
        _, sub = meshes
        solver, _, _ = self.build_fluid(meshes, rho_core=rho_core,
                                        rotation=True)
        core = coupling(solver, FluidCore)
        rotation = coupling(solver, Rotation)
        ctx = solver.coupling_context
        X = fd.SpatialCoordinate(sub)
        solver.solution.subfunctions[solver.field_index["displacement"]] \
            .interpolate(X / fd.sqrt(fd.dot(X, X)))

        predicted = rho_core * RC ** 2 * 2 * np.pi * RC
        with_core = fd.assemble(rotation.inertia_form(ctx, 2))
        saved = list(ctx.sheets)
        ctx.sheets[:] = [s for s in saved if s.name != "fluid core"]
        try:
            assert len(ctx.sheets) == len(saved) - 1
            without = fd.assemble(rotation.inertia_form(ctx, 2))
        finally:
            ctx.sheets[:] = saved

        assert with_core - without == pytest.approx(predicted, rel=1e-4)
        sheet_integral = (Rotation.inertia_polynomial(2, X)
                          * core.sheet_density(ctx.fields["displacement"])
                          * core.measure())
        assert fd.assemble(sheet_integral) == pytest.approx(predicted, rel=1e-4)

    def test_the_sheet_reaches_the_monopole_datum(self, meshes):
        """The 2-D enclosed-mass datum must see the core's volume change.

        The 2-D exterior DtN needs a monopole datum carrying *all* the enclosed
        mass; the divergence-form volume source contributes identically zero
        (V4), so the datum is the sheets alone - and the CMB sheet must reach
        it as one of the couplings' sheets. Nothing else looks: the potential
        stays smooth, the geoid is merely wrong.

        Exercised where the term is *nonzero*, which needs saying. With the
        prototype's `cos 2 phi` load the net CMB sheet mass is 8.3e-16, because
        the load has no degree-0 content and the CMB does not breathe - so a
        test on a solved state would pass with the term missing. Here the
        displacement is set to `u = rhat` by hand, a pure degree-0 expansion,
        and the datum must then read `rho_core 2 pi Rc` (the `cos 2 phi` load
        sheet contributing nothing to a net mass).
        """
        rho_core = 2.0
        _, sub = meshes
        solver, _, _ = self.build_fluid(meshes, rho_core=rho_core)
        X = fd.SpatialCoordinate(sub)
        solver.solution.subfunctions[solver.field_index["displacement"]] \
            .interpolate(X / fd.sqrt(fd.dot(X, X)))

        solver.update_total_mass()
        assert float(solver.gravity.source_mass) == pytest.approx(
            rho_core * 2 * np.pi * RC, rel=1e-4)

    def test_the_datum_is_unmoved_without_a_fluid_core(self, meshes):
        """The same state, rigid core: the datum sees the load sheet alone.

        The guard against the core sheet having changed the configuration
        everything else in this project was measured on.
        """
        _, sub = meshes
        solver, _, _ = build(meshes, rotation=False)
        X = fd.SpatialCoordinate(sub)
        solver.solution.subfunctions[solver.field_index["displacement"]] \
            .interpolate(X / fd.sqrt(fd.dot(X, X)))
        solver.update_total_mass()
        assert abs(float(solver.gravity.source_mass)) < 1e-12

    @REAL_BLOCK_ON_PARENT_FACETS
    def test_the_rotation_pair_transposes(self, meshes):
        """The centrifugal traction on the CMB, without which the pair is 97 % off.

        The fluid core's sheet enters `dI`, so it enters the `(m_i, u)` block;
        its transpose partner is a CMB traction proportional to `psi_rot`, and
        `rotational_potential_term` is a *volume* term with no such piece.
        Measured without that traction: `max|A(u,m3) - A(m3,u)^T| = 9.07e-04`
        against a block maximum of 9.32e-04, where the rigid core gives
        2.6e-15.

        It went unseen because every transpose gate in the fluid core's suite
        ran without rotation. This test exists so that cannot recur.
        """
        solver, _, _ = self.build_fluid(meshes, rotation=True)
        A = fd.assemble(fd.derivative(solver.F, solver.solution),
                        mat_type="nest").petscmat
        iu, im = solver.field_index["displacement"], solver.field_index["m3"]
        um = A.getNestSubMatrix(iu, im).convert("dense").getDenseArray()
        mu = A.getNestSubMatrix(im, iu).convert("dense").getDenseArray()
        scale = max(np.abs(um).max(), np.abs(mu).max())
        assert scale > 0.0
        assert np.abs(um - mu.T).max() / scale < 1e-14


class TestCentreOfMassFrame:
    """The centre-of-mass multipliers: layout, refusals, symmetry and the constraint.

    Structural checks, on the coarse annulus. The physics measurements (the
    translation near-kernel under a consistent reference gravity, the
    frame independence of `N - u_r`, the convergence of the multiplier) are
    in `NOTES/frame/`.

    The reference gravity here is the 2-D Gauss law for the test's own
    density, `g(r) = Lambda M(<r) / (2 pi r)`. With the constant `g = 1` of the
    other tests a rigid translation costs energy and the multiplier would
    carry a real force; the constraint would still hold, but the test would
    not describe the configuration the frame exists for.
    """

    RHO_CORE = 2.0

    @classmethod
    def gravity_of_r(cls, r):
        """`g(r)` for a uniform core of `RHO_CORE` and a unit-density mantle."""
        mass = cls.RHO_CORE * np.pi * RC ** 2 + np.pi * (r ** 2 - RC ** 2)
        return LAMBDA * mass / (2 * np.pi * r)

    def build(self, meshes, *, load_degree=1, rotation=False,
              approximation_kwargs=None, dtn_representation="multiplier",
              **solver_kwargs):
        """The frame configuration: fluid core, degree-`load_degree` sheet on Re.

        Args:
          meshes: the module fixture `(parent, sub)`.
          load_degree: the azimuthal degree of the load sheet on Re. Degree 1
            has a mass dipole, which the frame multipliers must cancel.
          rotation: carry the rotation closure and its moment of inertia.
          approximation_kwargs: extra keywords for the approximation.
          dtn_representation: the DtN representation. `"multiplier"` by
            default, the representation the frame tests have always used.
          solver_kwargs: passed to `SelfGravitatingGIASolver`.

        Returns:
          `(solver, u, psi)`.
        """
        parent, sub = meshes
        Xm = fd.SpatialCoordinate(sub)
        rm = fd.sqrt(fd.dot(Xm, Xm))
        u, psi = new_fields(meshes)
        couplings = [
            FluidCore(boundary=CURVE_RC, rho_core=self.RHO_CORE,
                      g=fd.Constant(self.gravity_of_r(RC))),
            CentreOfMass(),
            # The weight of the sheet with the surface gravity of this
            # reference state, so that the load exerts no net force.
            SurfaceLoad(boundary=CURVE_RE,
                        sigma=load_density(sub, load_degree),
                        g_surface=self.gravity_of_r(RE))]
        if rotation:
            couplings.append(Rotation(C=polar_moment(meshes),
                                      Omega_sq=OMEGA_SQ_EARTH))
        solver = SelfGravitatingGIASolver(
            u, psi,
            approximation(g=self.gravity_of_r(rm), **(approximation_kwargs or {})),
            dt=1.0, gravity_bcs=gravity_bcs(parent, sheet=False), bcs={},
            couplings=couplings, dtn_representation=dtn_representation,
            **solver_kwargs)
        return solver, u, psi

    @pytest.mark.parametrize("rotation", [False, True])
    def test_fields_are_last_and_one_per_direction(self, meshes, rotation):
        solver, _, _ = self.build(meshes, rotation=rotation)
        Z = solver.solution_space
        frame = [solver.field_index[f"centre_of_mass_{i}"] for i in range(2)]
        assert not {"centre_of_mass_2"} & set(solver.field_names)
        assert frame == list(range(len(Z) - 2, len(Z)))
        assert len(Z) == len(solver.field_names)
        real = tuple(i for i, V in enumerate(Z)
                     if V.ufl_element().family() == "Real")
        assert real == solver.real_indices
        assert real == tuple(range(real[0], len(Z)))

    def test_absent_by_default(self, meshes):
        solver, _, _ = TestFluidCore.build_fluid(meshes)
        assert not [name for name in solver.field_names
                    if name.startswith("centre_of_mass_")]

    def test_refuses_a_rigid_core(self, meshes):
        with pytest.raises(ValueError, match="requires a FluidCore"):
            build(meshes, rotation=False, couplings=[CentreOfMass()])

    @pytest.mark.parametrize("rotation", [False, True])
    def test_the_lowrank_space_carries_the_frame_fields(self, meshes, rotation):
        """On the low-rank path the frame fields are last, after core and rotation.

        The low-rank space has no DtN multiplier unknowns, so the `Real` run is
        the core pressure, the rotation scalar if present, and one frame
        multiplier per direction, contiguous and last. `DtNTwoBlockSchurPC`
        needs that run contiguous and last.
        """
        solver, _, _ = self.build(meshes, rotation=rotation,
                                  dtn_representation="lowrank")
        Z = solver.solution_space
        index = solver.field_index
        assert solver.dtn_representation == "lowrank"
        assert solver.multiplier_indices == ()
        assert len(Z) == len(solver.field_names)
        real = tuple(i for i, V in enumerate(Z)
                     if V.ufl_element().family() == "Real")
        # Core pressure, then the rotation scalar, then the two directions.
        expected = (index["core_pressure"],
                    *([index["m3"]] if rotation else []),
                    index["centre_of_mass_0"], index["centre_of_mass_1"])
        assert real == expected
        assert real == tuple(range(real[0], len(Z)))
        assert real == solver.real_indices

    def test_translation_moment_is_the_total_mass(self, meshes):
        """`D[u = e] = (rho_0 V_mantle + rho_core V_core) e`, per direction.

        The load sheet is taken out of the context's list of sheets, because
        its moment is a constant and would hide an error in the two
        displacement terms.
        """
        solver, _, _ = self.build(meshes, load_degree=1)
        _, sub = meshes
        centre = coupling(solver, CentreOfMass)
        ctx = solver.coupling_context
        mantle = fd.assemble(fd.Constant(1.0) * fd.dx(domain=sub))
        # The core volume from the discrete CMB circle, the same geometry the
        # core sheet is integrated on.
        Xm = fd.SpatialCoordinate(sub)
        ds_c = fd.Measure("ds", domain=sub)(CURVE_RC)
        radius = fd.assemble(fd.sqrt(fd.dot(Xm, Xm)) * ds_c) / fd.assemble(
            fd.Constant(1.0) * ds_c)
        mass = mantle + self.RHO_CORE * np.pi * radius ** 2
        saved = list(ctx.sheets)
        ctx.sheets[:] = [s for s in saved if s.name != "surface load"]
        try:
            assert len(ctx.sheets) == len(saved) - 1
            displacement = solver.solution.subfunctions[
                solver.field_index["displacement"]]
            for i, e in enumerate(([1.0, 0.0], [0.0, 1.0])):
                displacement.interpolate(fd.as_vector(e))
                D = [fd.assemble(centre.dipole_form(ctx, j)) for j in range(2)]
                assert abs(D[i] - mass) < 1e-6 * mass
                assert abs(D[1 - i]) < 1e-10 * mass
        finally:
            ctx.sheets[:] = saved

    @pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs global dense blocks")
    @pytest.mark.parametrize("rotation", [False, True])
    @REAL_BLOCK_ON_PARENT_FACETS
    def test_blocks_form_an_exact_saddle_pair(self, meshes, rotation):
        solver, _, _ = self.build(meshes, rotation=rotation)
        A = fd.assemble(fd.derivative(solver.F, solver.solution),
                        mat_type="nest").petscmat
        iu = solver.field_index["displacement"]
        frame = [solver.field_index[f"centre_of_mass_{i}"] for i in range(2)]
        for il in frame:
            ul = A.getNestSubMatrix(iu, il).convert("dense").getDenseArray()
            lu = A.getNestSubMatrix(il, iu).convert("dense").getDenseArray()
            scale = max(np.abs(ul).max(), np.abs(lu).max())
            assert scale > 0.0
            assert np.abs(ul - lu.T).max() / scale < 1e-14
            for jl in frame:
                ll = A.getNestSubMatrix(il, jl)
                if ll is not None:
                    assert np.abs(ll.convert("dense").getDenseArray()).max() \
                        == 0.0

    @pytest.mark.parametrize("dtn_representation", ["multiplier", "lowrank"])
    def test_solve_holds_the_centre_of_mass(self, meshes, dtn_representation):
        """After a solve with a degree-1 load, `D = 0` to the solver tolerance.

        The scale is the moment of the load sheet alone, which the displacement
        has to cancel.

        The test runs on both DtN representations. It is the only unit test
        that solves the frame without sea level on the low-rank path, and that
        is the configuration of the 3-D frame run G0-on. The multiplier arm
        uses the default solver parameters of the solver. The low-rank arm
        uses the iterative preset with the dense Schur complement on the
        `Real` block, as `test_sea_level.solve_settings` does, with an outer
        relative tolerance of 1e-12 and a Newton relative tolerance of 1e-10,
        so that the bound on `D` below is set by the discretisation and not by
        an inexact linear solve.
        """
        solver_kwargs = {}
        if dtn_representation == "lowrank":
            solver_kwargs = dict(
                solver_parameters="iterative",
                preset_options=dict(
                    multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
                    outer_rtol=1e-12, block0_rtol=1e-4, snes_rtol=1e-10))
        solver, _, _ = self.build(
            meshes, load_degree=1, dtn_representation=dtn_representation,
            **solver_kwargs)
        # The solver runs on the named representation.
        assert solver.dtn_representation == dtn_representation
        centre = coupling(solver, CentreOfMass)
        load_moment = SIGMA_HAT * np.pi * RE ** 2
        solver.solve()
        D = [fd.assemble(form) for form in centre.dipole()]
        assert abs(D[0]) < 1e-8 * load_moment
        assert abs(D[1]) < 1e-8 * load_moment
        assert all(np.isfinite([float(lam) for lam in centre.multipliers()]))

    def test_residual_is_the_variation_of_the_energy(self, meshes):
        """The written-out residual equals `derivative` of the energy.

        The residual is written term by term for a Slate reason (see
        `CentreOfMass.residual`). This pins that it is still the variation of
        the energy `c beta sum_i lambda_i D_i`, on a state where every unknown
        is nonzero. The energy is built from `CentreOfMass.dipole_form` with
        the multiplier as the weight and the row scaling as the scale, so the
        sheets and their restrictions are those of the residual.
        """
        solver, _, _ = self.build(meshes, rotation=False)
        centre = coupling(solver, CentreOfMass)
        ctx = solver.coupling_context
        z = solver.solution
        rng = np.random.default_rng(3)
        for sub_z in z.subfunctions:
            sub_z.dat.data[:] = rng.standard_normal(sub_z.dat.data.shape)
        beta = ctx.scaling_factor * ctx.row_scale_B_mu
        energy = sum(
            centre.dipole_form(ctx, i, weight=ctx.fields[f"centre_of_mass_{i}"],
                               scale=beta)
            for i in range(2))
        written = fd.assemble(centre.residual(ctx))
        derived = fd.assemble(fd.derivative(energy, z, fd.TestFunction(ctx.space)))
        for a, b in zip(written.subfunctions, derived.subfunctions):
            scale = max(np.abs(b.dat.data_ro).max(), 1e-300)
            assert np.abs(a.dat.data_ro - b.dat.data_ro).max() / scale < 1e-13

    @pytest.mark.skipif(
        fd.COMM_WORLD.size > 1,
        reason="the per-block instrument needs global dense blocks")
    def test_extruded_sphere_assembles_the_saddle_pair(self):
        """3-D, assembly only: three multipliers, exact transposes, `D[e] = M e`.

        The production geometry passes one extruded mesh for both roles and
        reaches the core boundary through `"bottom"`. No solve, by this file's
        rule for 3-D. The mesh is the small `CubedSphereMesh` of the Real-block
        diagonal tests. The top traction carries no mass (a `normal_stress`,
        which the solver accepts with a warning), so the only sheet is the
        core's.
        """
        from gadopt import SphericalDtN

        base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
        mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5,
                               extrusion_type="radial")
        mesh.cartesian = False
        u = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2), name="displacement")
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2), name="potential")
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(), dt=1.0,
            gravity_bcs={"top": {"dtn": SphericalDtN(1)},
                         "bottom": {"dtn": SphericalDtN(1)}},
            bcs={"top": {"normal_stress": B_MU * SIGMA_HAT}},
            couplings=[FluidCore(boundary="bottom", rho_core=self.RHO_CORE),
                       CentreOfMass()],
            dtn_representation="multiplier")
        Z = solver.solution_space
        frame = [solver.field_index[f"centre_of_mass_{i}"] for i in range(3)]
        assert frame == list(range(len(Z) - 3, len(Z)))

        A = fd.assemble(fd.derivative(solver.F, solver.solution),
                        mat_type="nest").petscmat
        iu = solver.field_index["displacement"]
        for il in frame:
            ul = A.getNestSubMatrix(iu, il).convert("dense").getDenseArray()
            lu = A.getNestSubMatrix(il, iu).convert("dense").getDenseArray()
            scale = max(np.abs(ul).max(), np.abs(lu).max())
            assert scale > 0.0
            assert np.abs(ul - lu.T).max() / scale < 1e-14

        # The whole mass of the body: the unit-density shell plus the core
        # inside the discrete bottom sphere, whose volume is taken from its
        # own area and mean radius (V = A r / 3 for a sphere).
        centre = coupling(solver, CentreOfMass)
        ctx = solver.coupling_context
        shell = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        ds_b = coupling(solver, FluidCore).measure()
        X = fd.SpatialCoordinate(mesh)
        area = fd.assemble(fd.Constant(1.0) * ds_b)
        radius = fd.assemble(fd.sqrt(fd.dot(X, X)) * ds_b) / area
        mass = shell + self.RHO_CORE * area * radius / 3
        displacement = solver.solution.subfunctions[iu]
        for i in range(3):
            e = [0.0, 0.0, 0.0]
            e[i] = 1.0
            displacement.interpolate(fd.as_vector(e))
            D = [fd.assemble(centre.dipole_form(ctx, j)) for j in range(3)]
            assert abs(D[i] - mass) < 1e-2 * mass
            assert max(abs(D[j]) for j in range(3) if j != i) < 1e-10 * mass
