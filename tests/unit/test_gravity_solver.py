"""Tests for gadopt.gravity_solver on Firedrake utility meshes.

All references are closed forms written out inline (delta-sheet and
volumetric-shell harmonics); the heavier gmsh-based validation against the
passess package lives in demos/gravity. The 2-D meshes are radially
extruded annuli (CircleManifoldMesh + ExtrudedMesh), the production GIA
mesh construction.

The 2-D closed-form solves (boundary sheets, volumetric shell, uniform
disc, mass channels, gauge, Dirichlet and flux), the submesh and preset
comparisons, the index-set split and the net-mass refusals and warnings
are checked by the `gravity` case in `tests/gia` (driver `gia_gravity.py`,
entries `closed-2d-*` and `submesh`). That driver and `gia_parallel.py`
import the mesh, source and comparison helpers of this module.
"""

import warnings

import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.gravity_solver import (
    direct_gravity_solver_parameters, iterative_gravity_solver_parameters)

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


# ---------------------------------------------------------------------------
# Geometry inference and input validation
# ---------------------------------------------------------------------------
class TestSetup:
    def test_orientation_and_radius_inference(self):
        mesh = annulus_mesh()
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=2)},
                           "bottom": {"dtn": CylindricalDtN(M=2)}})
        side_top, radius_top = solver.boundary_geometry["top"]
        side_bottom, radius_bottom = solver.boundary_geometry["bottom"]
        assert side_top == "exterior" and side_bottom == "interior"
        assert abs(radius_top - RMAX) < 1e-3 * RMAX
        assert abs(radius_bottom - RMIN) < 1e-3 * RMIN

    def test_bcs_validation(self):
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))

        with pytest.raises(ValueError, match="mutually exclusive"):
            GravitySolver(
                psi, 0.0,
                bcs={"top": {"dtn": CylindricalDtN(M=2), "psi": 0.0}})
        with pytest.raises(ValueError, match="unknown condition"):
            GravitySolver(psi, 0.0, bcs={"top": {"psl": 0.0}})
        with pytest.raises(ValueError, match="must be a"):
            GravitySolver(psi, 0.0, bcs={"top": {"dtn": 5}})
        with pytest.raises(ValueError, match="does not apply"):
            GravitySolver(psi, 0.0, bcs={"top": {"dtn": SphericalDtN(L=2)}})
        with pytest.raises(ValueError, match="Require M >= 0"):
            CylindricalDtN(M=-1)
        with pytest.raises(ValueError, match="Require L >= 0"):
            SphericalDtN(L=-1)

    def test_quadrature_check(self):
        """A degree too low to resolve the modes is caught by the check.

        `__init__` no longer measures this - it warns for free when the rule is
        capped or extrapolating, but the measurement needs `HDiv Trace` spaces
        over the whole mesh and is not worth a permanent 400 MiB on every
        construction. See `GravitySolver.warn_on_quadrature_rule_limits`. So an
        explicitly requested degree is honoured, and the check is what rejects
        it.
        """
        mesh = annulus_mesh(n_azimuthal=32, dr=0.25)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=10)}},
            quad_degree=1)
        with pytest.raises(ValueError, match="does not resolve"):
            solver.check_boundary_quadrature(rtol=1e-6)

    def test_quadrature_check_sampling(self):
        """`__init__` checks the extreme modes; "all" is still available.

        The constructor used to assemble one boundary form per treated mode,
        which is linear in the truncation. The sampled set is `2L + 2` modes
        rather than `(L+1)^2`, so it is still linear but with a much smaller
        constant, and it is a *proxy* rather than a bound - the worst mode is
        not always at the highest degree. This pins its composition, since the
        thing that must not happen quietly is the sample shrinking further.
        """
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        X = fd.SpatialCoordinate(mesh)

        cylindrical = CylindricalDtN(M=6)
        sampled = [m.key for m in cylindrical.check_modes("interior", 1.0, X)]
        every = [m.key for m in cylindrical.modes("interior", 1.0, X)]
        assert set(sampled) <= set(every)
        assert {"cos6", "sin6", "mean"} <= set(sampled)
        assert len(sampled) < len(every)

        # M = 0 is the degenerate pure-Robin case: no azimuthal mode is
        # treated, so none may be checked. Sampling an m = 1 that the solver
        # never assembles would let the constructor warn about a mode playing
        # no part in the answer.
        degenerate = CylindricalDtN(M=0)
        for side in ("exterior", "interior"):
            sampled = [m.key for m in degenerate.check_modes(side, 1.0, X)]
            assert sampled == [m.key for m in degenerate.modes(side, 1.0, X)]
        assert degenerate.check_modes("exterior", 1.0, X) == []

        spherical = SphericalDtN(L=4)
        X3 = fd.SpatialCoordinate(fd.UnitCubeMesh(1, 1, 1))
        sampled = [m.key for m in spherical.check_modes("exterior", 1.0, X3)]
        every = [m.key for m in spherical.modes("exterior", 1.0, X3)]
        assert set(sampled) <= set(every)
        assert set(sampled) == {"Y0,0"} | {f"Y4,{m}" for m in range(-4, 5)}
        assert len(sampled) < len(every)

        # Both settings agree on a configuration that resolves everything, and
        # an unknown setting is rejected rather than silently treated as "all".
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=3)}})
        assert solver.check_boundary_quadrature(sample="extremes") < 1e-8
        assert solver.check_boundary_quadrature(sample="all") < 1e-8
        with pytest.raises(ValueError, match="sample must be"):
            solver.check_boundary_quadrature(sample="some")

    def test_monolithic_override_guard(self):
        """A full solver_parameters replacement requesting monolithic
        assembly is rejected: R-space blocks cannot assemble aij."""
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        with pytest.raises(ValueError, match="Monolithic"):
            GravitySolver(
                psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=2)}},
                solver_parameters={"mat_type": "aij", "ksp_type": "preonly",
                                   "pc_type": "lu"})

    @pytest.mark.longtest
    def test_monopole_absent_in_3d(self):
        """3-D has no undetermined monopole, so none of this machinery engages."""
        mesh = shell_mesh_3d(refinement_level=1, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 1.0, bcs={"top": {"dtn": SphericalDtN(L=1)}})
        assert solver.monopole_boundaries == []
        solver.solve()
        assert solver.total_enclosed_mass() == 0.0

    def test_two_exterior_dtn_boundaries_refused(self):
        """"The enclosed mass" is not defined with two exterior boundaries."""
        mesh = fd.UnitDiskMesh(refinement_level=1)
        X = fd.SpatialCoordinate(mesh)
        marker = fd.Function(
            fd.FunctionSpace(mesh, "HDiv Trace", 0)
        ).interpolate(fd.conditional(X[1] > 0, 1.0, 0.0))
        relabelled = fd.RelabeledMesh(mesh, [marker], [77])
        psi = fd.Function(fd.FunctionSpace(relabelled, "CG", 1))
        with pytest.raises(ValueError, match="exterior DtN boundaries"):
            GravitySolver(psi, 1.0,
                          bcs={1: {"dtn": CylindricalDtN(M=1)},
                               77: {"dtn": CylindricalDtN(M=1)}})

    def test_flux_on_the_monopole_boundary_refused(self):
        """A flux datum and the DtN monopole both set d psi/d n at m = 0."""
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        with pytest.raises(ValueError, match="double specification"):
            GravitySolver(
                psi, 1.0,
                bcs={"top": {"dtn": CylindricalDtN(M=1),
                             "flux": fd.Constant(0.1)}})

    def _fieldsplit_0(self, solver):
        """The psi block sits under the 'dtn' subtree of DtNTwoBlockSchurPC."""
        return solver.solver_parameters["dtn"]["fieldsplit_0"]

    def test_default_preset_is_dimension_dependent(self):
        """Omitting solver_parameters selects direct in 2-D, iterative in 3-D,
        mirroring the Stokes solver - the psi block is MUMPS vs CG+GAMG."""
        mesh2 = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi2 = fd.Function(fd.FunctionSpace(mesh2, "CG", 1))
        solver2 = GravitySolver(
            psi2, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=2)}})
        # 2-D default -> direct: fieldsplit_0 is preonly + assembled LU.
        assert self._fieldsplit_0(solver2)["assembled"]["pc_type"] == "lu"

        mesh3 = shell_mesh_3d(refinement_level=0, dr=0.5)
        psi3 = fd.Function(fd.FunctionSpace(mesh3, "CG", 1))
        solver3 = GravitySolver(
            psi3, 0.0, bcs={"top": {"dtn": SphericalDtN(L=1)}})
        # 3-D default -> iterative: fieldsplit_0 is CG + SPDAssembledPC/GAMG.
        assert self._fieldsplit_0(solver3)["pc_python_type"] == "gadopt.SPDAssembledPC"
        assert self._fieldsplit_0(solver3)["assembled_pc_type"] == "gamg"

    def test_string_presets(self):
        """"direct"/"iterative" strings select the psi solver regardless of
        dimension; anything else is rejected."""
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        bcs = {"top": {"dtn": CylindricalDtN(M=2)}}

        # "iterative" accepted in 2-D (overrides the dimension default).
        solver_it = GravitySolver(psi, 0.0, bcs=bcs, solver_parameters="iterative")
        assert self._fieldsplit_0(solver_it)["pc_python_type"] == "gadopt.SPDAssembledPC"

        # "direct" gives the MUMPS psi block.
        solver_di = GravitySolver(psi, 0.0, bcs=bcs, solver_parameters="direct")
        assert self._fieldsplit_0(solver_di)["assembled"]["pc_type"] == "lu"

        # An unknown string is rejected rather than silently ignored.
        with pytest.raises(ValueError, match="must be a dictionary or one of"):
            GravitySolver(psi, 0.0, bcs=bcs, solver_parameters="banana")

    def test_preset_describes_the_split_by_index_set(self):
        """The preset must select DtNTwoBlockSchurPC, with the two sub-solvers
        under `dtn` and no field lists left at the top level.

        DO NOT DELETE THIS AS REDUNDANT WITH THE PARITY TESTS BELOW. It is one
        of only two things in this suite that can detect a return to the old
        field-number description of the split, and the parity tests are not the
        other one - quite the opposite. Both descriptions produce bit-identical
        answers below PETSc's 128-field limit (measured: trace coefficients
        agreeing exactly, identical outer iteration counts), which is precisely
        what makes the change safe and also what makes it invisible to every
        numerical test in this file. Revert `set_solver_options` and the
        potentials, the coefficients and the iteration counts all still match.
        The only detectors are this assertion on the shape of the options, and
        `test_solves_past_the_field_enumeration_wall`, which uses a truncation
        the old description physically cannot set up.
        """
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=2)}},
            solver_parameters="iterative")
        parameters = solver.solver_parameters

        assert parameters["pc_type"] == "python"
        assert parameters["pc_python_type"] == "gadopt.DtNTwoBlockSchurPC"
        assert parameters["mat_type"] == "matfree"
        assert parameters["dtn"]["pc_fieldsplit_schur_fact_type"] == "full"
        assert set(parameters["dtn"]) >= {"fieldsplit_0", "fieldsplit_1"}

        # Field lists are what carried the 128-field limit; none may survive,
        # at either spelling, or PETSc would enumerate the sub-fields again.
        assert not [key for key in parameters
                    if key.startswith(("fieldsplit_", "pc_fieldsplit_"))]

    def test_stale_top_level_fieldsplit_options_warn(self):
        """A `fieldsplit_0` override at the top level no longer reaches PETSc.

        Moving the sub-solvers under `dtn` silently orphans any override
        written against the old layout, so the class says so. The three silent
        cases matter as much as the two noisy ones, and the last of them most:
        a full dictionary is passed through verbatim, which is what keeps the
        old field-number split drivable - it is the control arm of every parity
        measurement behind this change. A warning there would be noise; a
        failure there would remove the only means of comparison.
        """
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        dtn = CylindricalDtN(M=2)
        bcs = {"top": {"dtn": dtn}}

        def stale_warnings(**kwargs):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                GravitySolver(psi, 0.0, bcs=bcs, **kwargs)
            return [str(w.message) for w in caught
                    if "sit at the top level" in str(w.message)]

        # Both spellings of an override written against the old layout warn.
        assert stale_warnings(
            solver_parameters="iterative",
            solver_parameters_extra={"fieldsplit_0": {"ksp_type": "cg"}})
        assert stale_warnings(
            solver_parameters="iterative",
            solver_parameters_extra={"fieldsplit_0_ksp_type": "cg"})

        # The correct nesting, a bare preset, and a full user dictionary do not.
        assert not stale_warnings(
            solver_parameters="iterative",
            solver_parameters_extra={"dtn": {"fieldsplit_0": {"ksp_rtol": 1e-9}}})
        assert not stale_warnings(solver_parameters="iterative")
        assert not stale_warnings(
            solver_parameters=field_number_parameters(
                "iterative", 1, dtn.n_multipliers("exterior")))


# ---------------------------------------------------------------------------
# 2-D: sheets and volumetric shell against inline closed forms
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The index-set two-block split: parity with the field-number split it
# replaces, and the truncations only it can reach
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3-D: spherical sheet against the inline closed form
# ---------------------------------------------------------------------------
@pytest.mark.longtest
class TestSpherical:
    def test_boundary_sheet(self):
        """Sheet sigma Y_lm on the outer DtN boundary of a spherical shell.

        The interior potential of a sheet at radius a is
        (4 pi G sigma a / (2l + 1)) (r/a)^l Y_lm; this also exercises
        R-space multipliers on the production extruded cubed sphere.
        """
        from gadopt.spherical_harmonics import real_spherical_harmonic

        l_mode, m_order = 2, 1
        mesh = shell_mesh_3d()
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        Y = real_spherical_harmonic(l_mode, m_order, X)

        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": SphericalDtN(L=3), "sigma": Y},
                 "bottom": {"dtn": SphericalDtN(L=3)}})
        assert solver.boundary_geometry["top"][0] == "exterior"
        assert solver.boundary_geometry["bottom"][0] == "interior"
        solver.solve()

        amplitude = 4 * np.pi * RMAX / (2 * l_mode + 1)
        reference = amplitude * (r / RMAX) ** l_mode * Y
        assert relative_l2_error(psi, reference, mesh) < 5e-3

        coefficients = solver.coefficients()
        mode_key = f"Y{l_mode},{m_order}"
        assert abs(coefficients["top"][mode_key] / amplitude - 1) < 5e-4
        inactive = max(
            abs(value) for bc in coefficients.values()
            for key, value in bc.items() if key != mode_key)
        assert inactive < 1e-9

    def test_interior_monopole(self):
        """Constant sheet on the inner boundary: the l = 0 machinery.

        A uniform sheet at r = a has nonzero total mass - legitimate in 3-D,
        where the l = 0 exterior decays as 1/r with no monopole exception.
        The domain-side potential is psi = 4 pi G sigma a^2 / r, exciting the
        interior l = 0 mean multiplier at the bottom boundary and the
        exterior l = 0 trace map at the top.
        """
        sigma = 1.3
        mesh = shell_mesh_3d()
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))

        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": SphericalDtN(L=1)},
                 "bottom": {"dtn": SphericalDtN(L=1), "sigma": sigma}})
        solver.solve()

        reference = 4 * np.pi * sigma * RMIN**2 / r
        assert relative_l2_error(psi, reference, mesh) < 5e-3

        # The Y0,0 trace coefficients: psi(R) = c_00 Y_00 = c_00 / sqrt(4 pi).
        coefficients = solver.coefficients()
        for marker, radius in (("top", RMAX), ("bottom", RMIN)):
            trace = 4 * np.pi * sigma * RMIN**2 / radius
            c_00 = coefficients[marker]["Y0,0"]
            assert abs(c_00 / (np.sqrt(4 * np.pi) * trace) - 1) < 5e-3
