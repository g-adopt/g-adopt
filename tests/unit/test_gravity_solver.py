"""Tests for gadopt.gravity_solver on Firedrake utility meshes.

All references are closed forms written out inline (delta-sheet and
volumetric-shell harmonics); the heavier gmsh-based validation against the
passess package lives in demos/gravity. The 2-D tests run on a radially
extruded annulus (CircleManifoldMesh + ExtrudedMesh) - the production GIA
mesh construction - which also exercises R-space multipliers and the
CombinedSurfaceMeasure boundary handling on extruded meshes.
"""

import warnings

import firedrake as fd
import numpy as np
import pytest
from firedrake.petsc import PETSc

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
        mesh = annulus_mesh(n_azimuthal=32, dr=0.25)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        with pytest.warns(UserWarning, match="does not resolve"):
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

    def test_net_mass_guard(self):
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 1.0, bcs={"top": {"dtn": CylindricalDtN(M=1)}})
        with pytest.raises(NotImplementedError, match="Net mass"):
            solver.solve()

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
class TestCylindrical:
    @pytest.mark.parametrize("side,m_mode", [("top", 2), ("top", 3),
                                             ("bottom", 2)])
    def test_boundary_sheet(self, side, m_mode):
        """Sheet sigma cos(m phi) on a DtN boundary.

        The domain-side potential of a sheet at radius a is the single
        harmonic (2 pi G sigma a / m) (r_</r_>)^m cos(m phi).
        """
        mesh = annulus_mesh()
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])
        a = RMAX if side == "top" else RMIN

        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"dtn": CylindricalDtN(M=4)},
                 "bottom": {"dtn": CylindricalDtN(M=4)},
                 side: {"dtn": CylindricalDtN(M=4),
                        "sigma": fd.cos(m_mode * phi)}})
        solver.solve()

        amplitude = 2 * np.pi * a / m_mode
        radial = (r / a) ** m_mode if side == "top" else (a / r) ** m_mode
        reference = amplitude * radial * fd.cos(m_mode * phi)
        assert relative_l2_error(psi, reference, mesh) < 1e-4

        coefficients = solver.coefficients()
        assert abs(coefficients[side][f"cos{m_mode}"] / amplitude - 1) < 1e-4
        inactive = max(
            abs(value) for bc in coefficients.values()
            for key, value in bc.items() if key != f"cos{m_mode}")
        assert inactive < 1e-10

    def test_volumetric_shell(self):
        """Blind modal DtN with an exactly integrated shell density.

        For rho = cos(m phi) on [r1, r2], the potential mode is (m != 2)
        psi_m(r) = (2 pi G / m) [r^-m I_in(r) + r^m I_out(r)] with
        I_in = int r'^(m+1) dr', I_out = int r'^(1-m) dr'.
        """
        m_mode = 3
        mesh = annulus_mesh(n_azimuthal=256, dr=0.05,
                            interfaces=(R1_SHELL, R2_SHELL))
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])
        shell = fd.conditional(fd.And(r >= R1_SHELL, r <= R2_SHELL), 1.0, 0.0)

        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, fd.cos(m_mode * phi) * shell,
            bcs={"top": {"dtn": CylindricalDtN(M=5)},
                 "bottom": {"dtn": CylindricalDtN(M=5)}},
            source_quad_degree=2 * m_mode + 8)
        solver.solve()

        def inner_integral(lower, upper):
            return (upper ** (m_mode + 2) - lower ** (m_mode + 2)) / (m_mode + 2)

        def outer_integral(lower, upper):
            return (upper ** (2 - m_mode) - lower ** (2 - m_mode)) / (2 - m_mode)

        r_in = fd.min_value(fd.max_value(r, R1_SHELL), R2_SHELL)
        radial = (r ** -m_mode * inner_integral(R1_SHELL, r_in)
                  + r**m_mode * outer_integral(r_in, R2_SHELL))
        reference = (2 * np.pi / m_mode) * radial * fd.cos(m_mode * phi)
        assert relative_l2_error(psi, reference, mesh, degree=10) < 1e-4

    def test_dirichlet_and_flux(self):
        """The 'psi' and 'flux' conditions reproduce a known harmonic.

        Manufactured solution psi = (r/RMAX)^m cos(m phi): Dirichlet trace at
        the outer boundary, prescribed normal derivative (n = -r_hat) at the
        inner one.
        """
        m_mode = 2
        mesh = annulus_mesh()
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])
        reference = (r / RMAX) ** m_mode * fd.cos(m_mode * phi)

        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"psi": reference},
                 "bottom": {"flux": -(m_mode / r) * reference}},
            quad_degree=12)
        solver.solve()
        assert relative_l2_error(psi, reference, mesh) < 2e-5

    def test_cross_mesh_consistency(self):
        """A Submesh-hosted density reproduces the same-mesh solution."""
        mesh = fd.UnitDiskMesh(refinement_level=3)
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])

        DG0 = fd.FunctionSpace(mesh, "DG", 0)
        indicator = fd.Function(DG0).interpolate(
            fd.conditional(r < 0.55, 1.0, 0.0))
        marked = fd.RelabeledMesh(mesh, [indicator], [99])
        submesh = fd.Submesh(marked, marked.topological_dimension, 99)

        rho_expression = fd.cos(2 * phi)
        X_sub = fd.SpatialCoordinate(submesh)
        phi_sub = fd.atan2(X_sub[1], X_sub[0])
        rho_sub = fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(
            fd.cos(2 * phi_sub))
        rho_same = fd.Function(fd.FunctionSpace(marked, "DG", 0)).interpolate(
            rho_expression * fd.conditional(r < 0.55, 1.0, 0.0))

        solutions = []
        for rho in (rho_same, rho_sub):
            psi = fd.Function(fd.FunctionSpace(marked, "CG", 2))
            solver = GravitySolver(
                psi, rho, bcs={1: {"dtn": CylindricalDtN(M=3)}})
            assert solver.cross_mesh == (rho is rho_sub)
            solver.solve()
            solutions.append(psi)

        difference = np.max(np.abs(
            solutions[0].dat.data_ro - solutions[1].dat.data_ro))
        scale = np.max(np.abs(solutions[0].dat.data_ro))
        assert difference < 1e-8 * scale

    def test_cross_mesh_iterative_preset(self):
        """The iterative (CG+GAMG) preset matches direct on the cross-mesh
        config, where fieldsplit_0 is blockdiag(A, DG0 dummy mass) spanning two
        meshes - the one block with no Stokes precedent for the AMG solve."""
        mesh = fd.UnitDiskMesh(refinement_level=3)
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        DG0 = fd.FunctionSpace(mesh, "DG", 0)
        indicator = fd.Function(DG0).interpolate(fd.conditional(r < 0.55, 1.0, 0.0))
        marked = fd.RelabeledMesh(mesh, [indicator], [99])
        submesh = fd.Submesh(marked, marked.topological_dimension, 99)
        X_sub = fd.SpatialCoordinate(submesh)
        phi_sub = fd.atan2(X_sub[1], X_sub[0])
        rho_sub = fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(
            fd.cos(2 * phi_sub))

        solutions = []
        for preset in ("direct", "iterative"):
            psi = fd.Function(fd.FunctionSpace(marked, "CG", 2))
            solver = GravitySolver(
                psi, rho_sub, bcs={1: {"dtn": CylindricalDtN(M=3)}},
                solver_parameters=preset)
            assert solver.cross_mesh and solver._multiplier_offset == 2
            solver.solve()
            solutions.append(psi.dat.data_ro.copy())

        difference = np.max(np.abs(solutions[0] - solutions[1]))
        scale = np.max(np.abs(solutions[0]))
        assert difference < 1e-6 * scale


# ---------------------------------------------------------------------------
# The index-set two-block split: parity with the field-number split it
# replaces, and the truncations only it can reach
# ---------------------------------------------------------------------------
class TestIndexSetSplit:
    """`DtNTwoBlockSchurPC` describes the two Schur blocks to PETSc as index
    sets instead of lists of field numbers. Nothing about the weak form, the
    mixed space or the taped problem changes, so below PETSc's 128-field
    enumeration limit the two descriptions must agree to round-off; above it,
    only the index-set description exists.
    """

    def test_matches_the_field_number_split(self):
        """Same answers as the description it replaces, on both presets.

        Not a tolerance check dressed up as one: the two paths run the same
        Krylov method on the same operator with the same sub-solvers, so the
        iterates coincide and the difference is round-off in the norms, not in
        the solve.
        """
        mesh = annulus_mesh(n_azimuthal=64, dr=0.25)
        rho = annulus_source(mesh)

        for preset in ("iterative", "direct"):
            psi_new, solver_new = solve_annulus(mesh, rho, 5, preset)
            psi_old, solver_old = solve_annulus(
                mesh, rho, 5,
                field_number_parameters(preset, solver_new._multiplier_offset,
                                        solver_new.n_multipliers))

            assert relative_l2_error(psi_new, psi_old, mesh) < 1e-12
            assert max_coefficient_difference(
                solver_old.coefficients(), solver_new.coefficients()) < 1e-12
            assert (solver_new.solver.snes.ksp.getIterationNumber()
                    == solver_old.solver.snes.ksp.getIterationNumber())

    def test_solves_past_the_field_enumeration_wall(self):
        """A truncation the field-number description cannot set up at all.

        PETSc refuses to enumerate more than 128 sub-fields, and this mixed
        space carries one scalar R space per treated mode, so the shipped
        solver used to stop here. The physical check is that raising the
        truncation leaves the low modes alone: the annulus is symmetric, so
        the modes decouple and the coefficients of modes 1..5 must not move.

        Deliberately the same form as
        `test_gravity_adjoint.test_taylor_rho_past_field_enumeration_wall`
        (same mesh, element, truncation and hence quadrature degree), so
        whichever runs second in a pytest process inherits the compiled
        kernels and skips the ~40s of symbolic form processing a 141-term
        boundary mode sum costs. Changing the truncation or the quadrature
        degree here alone silently costs that test ~40s.
        """
        mesh = annulus_mesh(n_azimuthal=96, dr=0.5)
        rho = annulus_source(mesh)

        _, solver_low = solve_annulus(mesh, rho, 5, "iterative")
        _, solver_high = solve_annulus(mesh, rho, 35, "iterative")

        # Counted off the objects, never off a formula for the truncation.
        n_fields = len(solver_high.mixed_space)
        assert n_fields == 1 + solver_high.n_multipliers
        assert n_fields > 128
        assert solver_high.solver.snes.ksp.getConvergedReason() > 0

        scale = max(abs(value) for boundary in solver_low.coefficients().values()
                    for value in boundary.values())
        assert max_coefficient_difference(
            solver_low.coefficients(), solver_high.coefficients()) < 1e-8 * scale

        # And the description it replaces still cannot reach this truncation:
        # PETSc raises (error code 56) while enumerating the sub-fields.
        with pytest.raises(PETSc.Error):
            solve_annulus(mesh, rho, 35,
                          field_number_parameters(
                              "iterative", solver_high._multiplier_offset,
                              solver_high.n_multipliers))

    def test_parallel_matches_the_field_number_split(self):
        """Two-rank parity: the coupled DtN solve had never been regression
        tested on more than one rank, and the Real-space multiplier degrees of
        freedom all live on rank 0, so the index sets are empty on every other
        rank - the one place a parallel off-by-one would hide.

        Skipped rather than passed on a single rank: run it under mpiexec.
        """
        if fd.COMM_WORLD.size < 2:
            pytest.skip("needs >= 2 MPI ranks; run under mpiexec")

        mesh = annulus_mesh(n_azimuthal=64, dr=0.25)
        rho = annulus_source(mesh)

        psi_new, solver_new = solve_annulus(mesh, rho, 5, "iterative")
        psi_old, solver_old = solve_annulus(
            mesh, rho, 5,
            field_number_parameters("iterative", solver_new._multiplier_offset,
                                    solver_new.n_multipliers))
        psi_direct, solver_direct = solve_annulus(mesh, rho, 5, "direct")

        assert relative_l2_error(psi_new, psi_old, mesh) < 1e-12
        assert max_coefficient_difference(
            solver_old.coefficients(), solver_new.coefficients()) < 1e-12
        assert relative_l2_error(psi_new, psi_direct, mesh) < 1e-9
        assert max_coefficient_difference(
            solver_direct.coefficients(), solver_new.coefficients()) < 1e-9

    def test_parallel_repeated_matfree_application_is_stable(self):
        """Applying the matrix-free Jacobian twice must give the same answer.

        Guards a live Firedrake defect rather than a hypothetical one: on more
        than one rank, a mixed operator with R-space blocks assembled matfree
        can be correct on its first application and wrong on every later one,
        gaining an extra copy of the volume contribution in each R row, and the
        corrupted operator is consistent enough that Krylov converges happily
        against it. `set_form` escapes only because every term touching an R
        field is a boundary integral. The day one grows a volume term - a
        volume-averaged constraint, say - this fails instead of quietly
        returning a converged wrong answer.
        """
        if fd.COMM_WORLD.size < 2:
            pytest.skip("needs >= 2 MPI ranks; run under mpiexec")

        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, annulus_source(mesh),
            bcs={"top": {"dtn": CylindricalDtN(M=3)},
                 "bottom": {"dtn": CylindricalDtN(M=3)}})

        jacobian = fd.derivative(solver.F, solver.mixed_solution,
                                 fd.TrialFunction(solver.mixed_space))
        matfree = fd.assemble(jacobian, mat_type="matfree").petscmat
        reference = fd.assemble(jacobian, mat_type="nest").petscmat

        x = matfree.createVecRight()
        # Seeded: a failure here has a subtle signature and must be reproducible.
        low, high = x.getOwnershipRange()
        x.setValues(range(low, high),
                    np.random.default_rng(42).standard_normal(x.getSize())[low:high])
        x.assemble()
        expected = reference.createVecLeft()
        reference.mult(x, expected)

        for application in range(3):
            y = matfree.createVecLeft()
            matfree.mult(x, y)
            y.axpy(-1.0, expected)
            assert y.norm() <= 1e-12 * expected.norm(), (
                f"matrix-free application {application} disagrees with the "
                "assembled operator")


# ---------------------------------------------------------------------------
# 3-D: spherical sheet against the inline closed form
# ---------------------------------------------------------------------------
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
