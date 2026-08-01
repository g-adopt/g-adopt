"""`DtNGravityForm` on its own, and the shim `GravitySolver` keeps over it.

The boundary treatment used to be half of `GravitySolver`; it is now
`gadopt.dtn_form.DtNGravityForm`, so that a coupled self-gravitating residual
can write the same boundary terms into its own mixed space without
instantiating a solver it does not want. Every existing gravity test still
passes, but that is a weak statement on its own: it is satisfied by a shim that
forwards attribute lookups, and it says nothing about whether the extracted
class works when it is the thing being driven. This file adds the two
assertions that are not implied by the absence of a regression.

The first is **identity**, not existence. `solver.ds is solver.form.ds` and so
on for every forwarded name, so that the day someone rebuilds a measure or a
dictionary on one side of the seam, the two cannot silently disagree. Three
names deliberately do not forward and are pinned here for the same reason.

The second is a **second consumer with an independent answer**. The mixed
space, the multiplier pairs, the volume term and the solve below are written by
hand, exactly as the coupled solver will write them, and the result is compared
against the closed form for a mass sheet on an interior circle - the
configuration `test_gravity_interior_sheet.py` runs through `GravitySolver`.
Running it through a directly constructed form is what tests the extraction:
the shipped solver could keep working while `DtNGravityForm` was unusable by
anybody else.
"""

import tempfile
from pathlib import Path

import gadopt  # noqa: F401 - before firedrake, for the python PC name below
import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, DtNGravityForm, GravitySolver
from gadopt.dtn_form import DtNGravityForm as _DtNGravityForm
from test_gravity_interior_sheet import (
    INNER_ID, N_AZIMUTHAL, OUTER_ID, RIN, ROUT, RSHEET, SHEET_ID,
    _write_annulus, relative_l2_error)

#: The direct 2-D preset, spelled flat because this is a hand-built problem
#: with no `GravitySolver` to nest the options for it.
MULTIPLIER_PARAMETERS = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-11,
    "pc_type": "python",
    "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
    "dtn_pc_fieldsplit_schur_fact_type": "full",
    "dtn_fieldsplit_0_ksp_type": "preonly",
    "dtn_fieldsplit_0_pc_type": "python",
    "dtn_fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "dtn_fieldsplit_0_assembled_ksp_type": "preonly",
    "dtn_fieldsplit_0_assembled_pc_type": "lu",
    "dtn_fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
    "dtn_fieldsplit_1_ksp_type": "gmres",
    "dtn_fieldsplit_1_ksp_rtol": 1e-6,
    "dtn_fieldsplit_1_pc_type": "none",
}

#: Every name `GravitySolver` forwards to its form. Written out rather than
#: read off the class, so that dropping one from the shim fails here instead
#: of failing wherever a demo happens to use it.
FORWARDED = (
    "ds", "dS", "quad_degree", "quad_rule_report", "dtn_boundaries",
    "sigma_bcs", "flux_bcs", "dirichlet_bcs", "boundary_geometry",
    "boundary_area", "mode_rows", "element_degree", "surface_measure",
    "geometry_probe_measure", "boundary_facet_scale", "default_quad_degree",
    "quadrature_rule_report", "warn_on_quadrature_rule_limits",
    "sheet_integral", "boundary_quadrature_rule", "check_boundary_quadrature",
    "_mode_norms", "_validate_recovered_rule",
)

#: The form's own build steps. Reachable on `solver.form`, deliberately not
#: given a second name on the solver: re-running one after the residual is
#: built would rebuild state the residual already refers to.
NOT_FORWARDED = ("set_boundary_conditions", "set_measures",
                 "check_sheet_measures", "set_boundary_geometry")


@pytest.fixture(scope="module")
def sheet_mesh():
    """The gmsh annulus with the sheet circle tagged.

    The same mesh `test_gravity_interior_sheet` builds, at the same cached
    path, so the two files share the generation rather than the fixture -
    pytest keys a fixture by the name it is bound to in the module, and
    importing that one as `mesh` would shadow a parameter in every test here.
    A fixed path rather than `tmp_path_factory`, because under MPI every rank
    runs its own pytest and would otherwise be handed a different directory.
    """
    pytest.importorskip("gmsh")
    path = Path(tempfile.gettempdir()) / (
        f"gadopt_interior_sheet_{RIN}_{RSHEET}_{ROUT}_{N_AZIMUTHAL}.msh")
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        _write_annulus(path)
    fd.COMM_WORLD.barrier()
    return fd.Mesh(str(path))


def annulus():
    """A plain extruded annulus; the shim tests need no tagged interior facet."""
    base = fd.CircleManifoldMesh(64, radius=1.0, degree=2)
    return fd.ExtrudedMesh(base, layers=6, layer_height=1.0 / 6,
                           extrusion_type="radial")


def shim_solver(representation="multiplier"):
    psi = fd.Function(fd.FunctionSpace(annulus(), "CG", 2))
    return GravitySolver(
        psi, 1.0,
        bcs={"top": {"dtn": CylindricalDtN(M=3)},
             "bottom": {"dtn": CylindricalDtN(M=3)}},
        solver_parameters="direct", dtn_representation=representation)


class TestTheShimForwardsIdentity:
    """`getattr` succeeding is not the property that matters."""

    @pytest.mark.parametrize("representation", ["multiplier", "lowrank"])
    def test_every_forwarded_name_is_the_form_s_own_object(self, representation):
        solver = shim_solver(representation)
        assert isinstance(solver.form, _DtNGravityForm)
        for name in FORWARDED:
            mine = getattr(solver, name)
            theirs = getattr(solver.form, name)
            if hasattr(mine, "__self__"):
                # A bound method: the binding is what has to agree, since two
                # bindings of the same function are never `is` each other.
                assert mine.__self__ is solver.form, name
                assert mine.__func__ is theirs.__func__, name
            elif isinstance(mine, int):
                # `quad_degree` and `element_degree` are numbers, where `is`
                # would be testing CPython's small-integer cache.
                assert mine == theirs, name
            else:
                assert mine is theirs, name

    def test_the_build_steps_are_reachable_only_on_the_form(self):
        solver = shim_solver()
        for name in NOT_FORWARDED:
            assert hasattr(solver.form, name), name
            assert not hasattr(solver, name), name

    def test_the_multiplier_keys_are_the_form_s_list(self):
        """Read by `test_gravity_adjoint`, and by `dtn_adjoint` positionally."""
        solver = shim_solver()
        assert solver._multiplier_keys is solver.form.multiplier_keys
        assert len(solver._multiplier_keys) == solver.form.n_multipliers

    def test_the_mode_rows_pair_with_the_boundaries_positionally(self):
        """`taped_trace_coefficients` zips the two; a filter would mis-pair."""
        solver = shim_solver("lowrank")
        assert len(solver.mode_rows) == len(solver.dtn_boundaries)
        keys = [(bc_id, key)
                for (bc_id, _), rows in zip(solver.dtn_boundaries,
                                            solver.mode_rows)
                for key in rows.keys]
        assert keys == solver._multiplier_keys


class TestWhatDoesNotForward:
    """Three names stay behind, and each would fail quietly if it moved."""

    def test_alpha_stays_a_GravitySolver_class_attribute(self):
        """The Robin shift is this class's knob, and a test overrides it.

        `test_gravity_solver.test_monopole_gauge_does_not_depend_on_the_robin_
        shift` subclasses `GravitySolver` and sets `alpha` on the subclass for
        three values, then asserts the three solutions agree. Had `alpha`
        become a `DtNGravityForm` class attribute the override would bind to
        nothing, all three solves would run at 1.0, and the three solutions
        would agree *exactly* - a test that passes and certifies nothing. So
        the shift is passed into the form by `__init__`, and this pins that it
        arrives.
        """
        class ShiftedGravitySolver(GravitySolver):
            alpha = 2.5

        psi = fd.Function(fd.FunctionSpace(annulus(), "CG", 1))
        solver = ShiftedGravitySolver(
            psi, 1.0, bcs={"top": {"dtn": CylindricalDtN(M=2)}},
            solver_parameters="direct")
        assert solver.alpha == 2.5
        assert solver.form.alpha == 2.5
        assert _DtNGravityForm.alpha == 1.0  # the class default is untouched

    def test_n_multipliers_is_zero_on_the_lowrank_path_and_not_on_the_form(self):
        """"No fields to enumerate" is a fact about the solver, not the form."""
        solver = shim_solver("lowrank")
        assert solver.n_multipliers == 0
        assert solver.form.n_multipliers == 13  # 2*3 exterior + 2*3+1 interior

    def test_the_multiplier_offset_indexes_the_solver_s_own_space(self):
        """It means nothing to a form that owns no mixed space."""
        solver = shim_solver()
        assert solver._multiplier_offset == 1
        assert not hasattr(solver.form, "_multiplier_offset")


class TestTheFormOnItsOwn:
    """The second consumer: a hand-built residual, and a closed form."""

    @staticmethod
    def solve_through_the_form(sheet_mesh, sigma, M=4):
        """A coupled solver's residual in miniature, written out by hand.

        Volume term, mixed space and multiplier pairing are the caller's - the
        shape Phase 4's monolithic system will have - and everything on the
        boundary comes from `boundary_residual`. Nothing here imports
        `GravitySolver`.
        """
        V = fd.FunctionSpace(sheet_mesh, "CG", 2)
        form = DtNGravityForm(
            V,
            {OUTER_ID: {"dtn": CylindricalDtN(M=M)},
             INNER_ID: {"dtn": CylindricalDtN(M=M)},
             SHEET_ID: {"interior_sigma": sigma}})

        R = fd.FunctionSpace(sheet_mesh, "R", 0)
        W = fd.MixedFunctionSpace([V] + [R] * form.n_multipliers)
        w = fd.Function(W)
        trials, tests = fd.split(w), fd.TestFunctions(W)
        psi, v = trials[0], tests[0]

        F = fd.dot(fd.grad(psi), fd.grad(v)) * fd.dx(domain=sheet_mesh)
        F += form.boundary_residual(psi, v, list(zip(trials[1:], tests[1:])))
        fd.solve(F == 0, w, solver_parameters=MULTIPLIER_PARAMETERS)
        return w, form

    @pytest.mark.parametrize("m_mode", [1, 2])
    def test_interior_sheet_against_its_closed_form(self, sheet_mesh, m_mode):
        """sigma = cos(m phi) at radius a, no volume source at all.

        psi = (2 pi G a / m) (r_< / r_>)^m cos(m phi), the same reference
        `test_gravity_interior_sheet.py` runs through `GravitySolver` and
        `demos/gravity/gravity_poisson_interior_sheet.py` runs against
        `passess.polar.SheetPolar2D`. An absent sheet returns the zero field
        rather than a wrong one, so the tolerance is not what makes this test
        worth its runtime.
        """
        X = fd.SpatialCoordinate(sheet_mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])

        w, form = self.solve_through_the_form(sheet_mesh, fd.cos(m_mode * phi))
        psi = w.subfunctions[0]

        amplitude = 2 * np.pi * RSHEET / m_mode
        ratio = fd.min_value(r, RSHEET) / fd.max_value(r, RSHEET)
        reference = amplitude * ratio**m_mode * fd.cos(m_mode * phi)
        assert relative_l2_error(psi, reference, sheet_mesh) < 5e-3

        # The multipliers the form's constraint rows defined, read out of the
        # caller's own mixed space through `multiplier_keys`. This is the
        # pairing a coupled solver depends on and it is not exercised anywhere
        # else outside `GravitySolver`.
        volume = fd.assemble(1 * fd.dx(domain=sheet_mesh))
        for bc_id, expected in ((OUTER_ID, amplitude * (RSHEET / ROUT) ** m_mode),
                                (INNER_ID, amplitude * (RIN / RSHEET) ** m_mode)):
            index = form.multiplier_keys.index((bc_id, f"cos{m_mode}"))
            c = fd.assemble(w.subfunctions[1 + index] * fd.dx(domain=sheet_mesh))
            assert c / volume == pytest.approx(expected, rel=5e-3)

    def test_the_form_refuses_the_wrong_number_of_multipliers(self, sheet_mesh):
        """The count is the caller's to get right, so it is checked."""
        V = fd.FunctionSpace(sheet_mesh, "CG", 1)
        form = DtNGravityForm(V, {OUTER_ID: {"dtn": CylindricalDtN(M=2)}})
        R = fd.FunctionSpace(sheet_mesh, "R", 0)
        W = fd.MixedFunctionSpace([V, R])
        trials, tests = fd.split(fd.Function(W)), fd.TestFunctions(W)
        with pytest.raises(ValueError, match="multiplier"):
            form.boundary_bilinear(trials[0], tests[0],
                                   list(zip(trials[1:], tests[1:])))

    def test_the_boundary_source_is_empty_when_there_is_nothing_on_it(self, sheet_mesh):
        """An empty form, so a caller needs no `if` around it."""
        V = fd.FunctionSpace(sheet_mesh, "CG", 1)
        form = DtNGravityForm(V, {OUTER_ID: {"dtn": CylindricalDtN(M=2)}})
        v = fd.TestFunction(V)
        source = form.boundary_source(v)
        assert source.empty()
        bilinear = form.boundary_bilinear(fd.TrialFunction(V), v)
        assert len((bilinear - source).integrals()) == len(bilinear.integrals())

    def test_extra_flux_is_the_channel_the_monopole_datum_uses(self, sheet_mesh):
        """`GravitySolver` keeps the mass bookkeeping; only the value crosses.

        Assembled against a constant test function, `extra_flux` has to
        reproduce the boundary integral of the flux it was given - which is the
        whole of the contract, since the form does nothing else with it.
        """
        V = fd.FunctionSpace(sheet_mesh, "CG", 1)
        form = DtNGravityForm(V, {OUTER_ID: {"dtn": CylindricalDtN(M=2)}})
        v = fd.TestFunction(V)
        flux = fd.Constant(-0.375)
        source = form.boundary_source(v, extra_flux={OUTER_ID: flux})
        one = fd.Function(V).assign(1.0)
        perimeter = fd.assemble(1 * form.ds(OUTER_ID))
        assert fd.assemble(fd.action(source, one)) == pytest.approx(
            float(flux) * perimeter, rel=1e-12)
