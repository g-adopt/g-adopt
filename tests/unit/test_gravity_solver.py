"""Tests for gadopt.gravity_solver on Firedrake utility meshes.

All references are closed forms written out inline (delta-sheet and
volumetric-shell harmonics); the heavier gmsh-based validation against the
passess package lives in demos/gravity. The 2-D tests run on a radially
extruded annulus (CircleManifoldMesh + ExtrudedMesh) - the production GIA
mesh construction - which also exercises R-space multipliers and the
CombinedSurfaceMeasure boundary handling on extruded meshes.
"""

import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN

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

    def test_net_mass_guard(self):
        mesh = annulus_mesh(n_azimuthal=32, dr=0.5)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 1.0, bcs={"top": {"dtn": CylindricalDtN(M=1)}})
        with pytest.raises(NotImplementedError, match="Net mass"):
            solver.solve()


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
