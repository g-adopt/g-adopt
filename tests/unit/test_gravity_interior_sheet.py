"""Mass sheets on a tagged interior facet (`interior_sigma`).

The coupled self-gravitating problem stands its DtN boundaries off from the
sources, which puts the Earth's surface inside the domain: the surface load is
then a sheet on an *interior* facet, and the shipped `sigma` spelling - an
exterior-facet measure - integrates over nothing. It does so silently, with a
`WARNING Subdomain (2,) is empty` and a converged solve, which is why these
tests come with a constructor-time refusal rather than only with a test.

Three levels, in increasing order of what they would catch:

  * the measures themselves, against `2 pi R`, which is where a zero shows;
  * a `cos(2 phi)` sheet against its closed form, which is where a factor or a
    wrong restriction shows;
  * a constant sheet against the 2-D monopole datum, which is the only path
    that runs the sheet through `enclosed_mass_forms` and `check_net_mass`.

The reference potentials are the same closed forms the rest of this suite
writes out inline (and the same ones `passess.polar.SheetPolar2D` evaluates -
`demos/gravity/gravity_poisson_interior_sheet.py` runs the comparison against
the package itself, on this same configuration).

The mesh is gmsh rather than a Firedrake utility mesh, because an interior
facet can only carry a subdomain id if something tags it, and because the
circle it sits on must conform to element edges for the closed form to mean
anything. `AnnulusMesh` is a tensor-product cell and cannot take a plain
`exterior_facet` integral at all, so it is not an alternative here.
"""

import tempfile
from pathlib import Path

import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, GravitySolver

RIN, RSHEET, ROUT = 1.0, 1.6, 2.4
OUTER_ID, INNER_ID, SHEET_ID = 1, 2, 7
N_AZIMUTHAL = 128


def _write_annulus(path):
    """Two conforming rings with the circle between them tagged as a curve.

    Transfinite, so every azimuthal position is a mesh line and the tagged
    circle is exactly the interface between the two rings - which is what
    makes it an interior facet with a physical group of its own.
    """
    import gmsh

    radii = [RIN, RSHEET, ROUT]
    curve_tags = [INNER_ID, SHEET_ID, OUTER_ID]
    layers = [4, 6]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("interior_sheet_annulus")
    geo = gmsh.model.geo

    centre = geo.addPoint(0, 0, 0)
    quadrants = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    points = [[geo.addPoint(R * cx, R * cy, 0) for cx, cy in quadrants]
              for R in radii]
    arcs = [[geo.addCircleArc(pts[j], centre, pts[(j + 1) % 4])
             for j in range(4)] for pts in points]
    radial = [[geo.addLine(points[i][j], points[i + 1][j]) for j in range(4)]
              for i in range(len(radii) - 1)]

    surfaces = []
    for i in range(len(radii) - 1):
        ring = []
        for j in range(4):
            loop = geo.addCurveLoop([arcs[i][j], radial[i][(j + 1) % 4],
                                     -arcs[i + 1][j], -radial[i][j]])
            ring.append(geo.addPlaneSurface([loop]))
        surfaces.append(ring)

    for ring in arcs:
        for arc in ring:
            geo.mesh.setTransfiniteCurve(arc, N_AZIMUTHAL // 4 + 1)
    for i, n_layers in enumerate(layers):
        for line in radial[i]:
            geo.mesh.setTransfiniteCurve(line, n_layers + 1)
    for ring in surfaces:
        for surface in ring:
            geo.mesh.setTransfiniteSurface(surface)

    geo.synchronize()
    for tag, ring in zip(curve_tags, arcs):
        gmsh.model.addPhysicalGroup(1, ring, tag)
    gmsh.model.addPhysicalGroup(2, [s for ring in surfaces for s in ring], 101)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(path))
    gmsh.finalize()


@pytest.fixture(scope="module")
def mesh():
    """The annulus, generated once on rank 0 and read on all of them.

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


def relative_l2_error(psi, reference, mesh, degree=8):
    dxq = fd.dx(domain=mesh, degree=degree)
    error = fd.assemble((psi - reference) ** 2 * dxq)
    norm = fd.assemble(reference**2 * dxq)
    return np.sqrt(error / norm)


def solve_with_sheet(mesh, sigma, M=4):
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
    solver = GravitySolver(
        psi, 0.0,
        bcs={OUTER_ID: {"dtn": CylindricalDtN(M=M)},
             INNER_ID: {"dtn": CylindricalDtN(M=M)},
             SHEET_ID: {"interior_sigma": sigma}},
        solver_parameters="direct")
    solver.solve()
    return psi, solver


class TestMeasures:
    """V10-assembly: the measures, before anything is solved on them."""

    def test_tagged_interior_circle_has_its_circumference(self, mesh):
        """Zero is the failure mode; 2 pi R low by the straight-chord error."""
        length = fd.assemble(fd.avg(fd.Constant(1.0)) * fd.dS(SHEET_ID,
                                                              domain=mesh))
        assert length == pytest.approx(2 * np.pi * RSHEET, rel=1e-3)
        assert abs(length / (2 * np.pi * RSHEET) - 1) < 1e-3

    def test_the_shipped_exterior_spelling_finds_nothing(self, mesh):
        """The silent failure this whole condition exists to prevent.

        Not an exception and not a diagnostic: `0.0` and a warning, which is
        why the solver has to refuse it rather than rely on the user reading
        the log.
        """
        assert fd.assemble(fd.Constant(1.0) * fd.ds(SHEET_ID,
                                                    domain=mesh)) == 0.0

    @pytest.mark.longtest
    def test_sheet_integral_agrees_with_the_measure(self, mesh):
        """The solver's own helper, not a hand-written form beside it."""
        _, solver = solve_with_sheet(mesh, fd.Constant(1.0))
        length = fd.assemble(solver.sheet_integral(
            fd.Constant(1.0), SHEET_ID, "interior_facet"))
        assert length == pytest.approx(2 * np.pi * RSHEET, rel=1e-3)


class TestRefusals:
    """The structural guard, which matters more than any of the tests above."""

    def test_sheet_on_an_empty_exterior_measure_is_refused(self, mesh):
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        with pytest.raises(ValueError, match="sheet measure is empty"):
            GravitySolver(
                psi, 0.0,
                bcs={OUTER_ID: {"dtn": CylindricalDtN(M=2)},
                     INNER_ID: {"dtn": CylindricalDtN(M=2)},
                     SHEET_ID: {"sigma": fd.Constant(1.0)}})

    def test_interior_sheet_on_a_boundary_tag_is_refused(self, mesh):
        """The same guard the other way round, since the tag cannot say."""
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        with pytest.raises(ValueError, match="sheet measure is empty"):
            GravitySolver(
                psi, 0.0,
                bcs={OUTER_ID: {"dtn": CylindricalDtN(M=2),
                                "interior_sigma": fd.Constant(1.0)},
                     INNER_ID: {"dtn": CylindricalDtN(M=2)}})

    def test_interior_sheet_refused_on_an_extruded_mesh(self):
        """dS_h carries no subdomain ids, so there is nothing to tag."""
        base = fd.CircleManifoldMesh(32, radius=1.0, degree=2)
        extruded = fd.ExtrudedMesh(base, layers=4, layer_height=0.25,
                                   extrusion_type="radial")
        psi = fd.Function(fd.FunctionSpace(extruded, "CG", 1))
        with pytest.raises(NotImplementedError, match="extruded mesh"):
            GravitySolver(
                psi, 0.0,
                bcs={"top": {"dtn": CylindricalDtN(M=2),
                             "interior_sigma": fd.Constant(1.0)}})


@pytest.mark.longtest
class TestAgainstClosedForms:
    """V10-end-to-end: the sheet is present, and present with the right size."""

    @pytest.mark.parametrize("m_mode", [1, 2])
    def test_single_mode_sheet(self, mesh, m_mode):
        """sigma = cos(m phi) at radius a, no volume source.

        psi = (2 pi G a / m) (r_< / r_>)^m cos(m phi) on both sides of the
        sheet, so this is the same closed form the exterior-facet sheet test
        uses, evaluated on a circle the solve does not have a boundary on.
        """
        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        phi = fd.atan2(X[1], X[0])

        psi, solver = solve_with_sheet(mesh, fd.cos(m_mode * phi))

        amplitude = 2 * np.pi * RSHEET / m_mode
        ratio = fd.min_value(r, RSHEET) / fd.max_value(r, RSHEET)
        reference = amplitude * ratio**m_mode * fd.cos(m_mode * phi)
        assert relative_l2_error(psi, reference, mesh) < 5e-3

        # The traces the DtN multipliers recovered, against the same form
        # evaluated on the two boundaries. An empty facet integral leaves
        # these at zero rather than merely wrong.
        coefficients = solver.coefficients()
        outer = amplitude * (RSHEET / ROUT) ** m_mode
        inner = amplitude * (RIN / RSHEET) ** m_mode
        assert coefficients[OUTER_ID][f"cos{m_mode}"] == pytest.approx(
            outer, rel=5e-3)
        assert coefficients[INNER_ID][f"cos{m_mode}"] == pytest.approx(
            inner, rel=5e-3)

    def test_constant_sheet_drives_the_monopole_datum(self, mesh):
        """The only path that runs a sheet through the mass bookkeeping.

        A constant sheet carries net mass 2 pi a sigma, which in 2-D the
        exterior trace cannot see, so it reaches the far field only through
        `enclosed_mass_forms` - the third of the four sheet sites, and the one
        whose test function lives on a `Real` space. Outside the sheet the
        potential is the solver's gauge-fixed log, inside it is the constant
        that log takes at the sheet.
        """
        sigma = 0.7
        psi, solver = solve_with_sheet(mesh, fd.Constant(sigma), M=2)

        mass = 2 * np.pi * RSHEET * sigma
        assert solver.total_enclosed_mass() == pytest.approx(mass, rel=1e-3)

        X = fd.SpatialCoordinate(mesh)
        r = fd.sqrt(fd.dot(X, X))
        reference = -2 * mass * fd.ln(fd.max_value(r, RSHEET) / ROUT)
        # The gauge is `mean psi over the exterior boundary = 0`, and the
        # discrete boundary is a polygon rather than a circle, so the
        # reference is re-centred the same way before comparison.
        perimeter = fd.assemble(1 * solver.ds(OUTER_ID))
        reference -= fd.assemble(reference * solver.ds(OUTER_ID)) / perimeter
        assert relative_l2_error(psi, reference, mesh) < 5e-3

    def test_the_sheet_is_not_silently_absent(self, mesh):
        """The magnitude, stated against the failure it is guarding.

        If the facet integral were empty the solve would still converge and
        return the zero field, so what makes this test worth its runtime is
        the ratio between the two outcomes rather than the tolerance.
        """
        X = fd.SpatialCoordinate(mesh)
        phi = fd.atan2(X[1], X[0])
        psi, _ = solve_with_sheet(mesh, fd.cos(2 * phi))
        norm = np.sqrt(fd.assemble(psi**2 * fd.dx(domain=mesh)))
        assert norm > 1.0


@pytest.mark.longtest
class TestRealControlsOnTheSheet:
    """Gradients with respect to `Real` controls that scale an interior sheet.

    The sheet term is `4 pi G int avg(sigma v) dS`. A `Real` coefficient
    outside the `avg` assembles, because UFL gives a coefficient of a
    continuous element a default side, but the derivative with respect to it
    is a `Real` `Argument`, which UFL accepts on an interior facet only with a
    restriction (`Discontinuous type Argument must be restricted`). The
    sheet integral (`DtNGravityForm.sheet_integral`) therefore puts the whole
    integrand, `G` and the sheet scale included, under one restriction with
    `gadopt.utility.interior_facet_integrand`. Both gradients are checked
    against a central difference of fresh solves.
    """

    G0, S0, STEP = 1.3, 0.7, 1e-5

    def functional(self, mesh, g_value, s_value):
        """`int psi^2 dx` after one solve with `Real` `G` and sheet scale."""
        from firedrake.adjoint import stop_annotating

        R = fd.FunctionSpace(mesh, "R", 0)
        with stop_annotating():
            G = fd.Function(R).assign(g_value)
            scale = fd.Function(R).assign(s_value)
        X = fd.SpatialCoordinate(mesh)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        bcs = {OUTER_ID: {"dtn": CylindricalDtN(M=4)},
               INNER_ID: {"dtn": CylindricalDtN(M=4)},
               SHEET_ID: {"interior_sigma":
                          scale * (1 + fd.cos(2 * fd.atan2(X[1], X[0])))}}
        GravitySolver(psi, 0.0, gravitational_constant=G, bcs=bcs,
                      solver_parameters="direct").solve()
        return fd.assemble(psi ** 2 * fd.dx), G, scale

    @pytest.mark.parametrize("control", ["G", "sheet scale"])
    def test_gradient_matches_a_central_difference(self, mesh, control):
        from firedrake.adjoint import (Control, ReducedFunctional,
                                       continue_annotation, get_working_tape,
                                       pause_annotation, stop_annotating)

        tape = get_working_tape()
        tape.clear_tape()
        continue_annotation()
        try:
            J, G, scale = self.functional(mesh, self.G0, self.S0)
        finally:
            pause_annotation()
        m = G if control == "G" else scale
        gradient = float(ReducedFunctional(J, Control(m)).derivative())
        tape.clear_tape()
        values = []
        with stop_annotating():
            for sign in (1, -1):
                g_value, s_value = self.G0, self.S0
                if control == "G":
                    g_value += sign * self.STEP
                else:
                    s_value += sign * self.STEP
                values.append(float(self.functional(mesh, g_value, s_value)[0]))
        difference = (values[0] - values[1]) / (2 * self.STEP)
        # The functional is quadratic in either control, so the central
        # difference is exact up to the round-off of the two solves.
        assert abs(gradient - difference) <= 1e-7 * abs(difference)


class TestInteriorFacetIntegrand:
    """`interior_facet_integrand` applied to a `Real` function on one small mesh.

    The helper restricts a quantity on interior facets with `avg`. For a
    `Real` function the value is unchanged, and the derivative with respect to
    it is a restricted `Real` `Argument`, which UFL accepts on `dS`.
    """

    @pytest.fixture(scope="class")
    def square(self):
        """A unit square with a `Real` function of value 1."""
        mesh = fd.UnitSquareMesh(4, 4)
        r = fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(1.0)
        return mesh, r

    def test_the_derivative_exists_and_has_the_facet_value(self, square):
        """`derivative(avg(r) dS, r)` is the interior facet length, once."""
        from gadopt.utility import interior_facet_integrand
        mesh, r = square
        form = interior_facet_integrand(r) * fd.dS
        exact = fd.assemble(fd.Constant(1.0) * fd.dS(domain=mesh))
        # `avg(r) = r` for a field with one value on both sides.
        assert fd.assemble(form) == pytest.approx(exact, rel=1e-14)
        # Each side contributes half of the facet integral to the one global
        # entry, so the sum is the facet length once and not twice.
        gradient = fd.assemble(fd.derivative(form, r)).dat.data_ro_with_halos[0]
        assert gradient == pytest.approx(exact, rel=1e-14)

    def test_an_unrestricted_real_function_has_no_derivative_on_ds(self, square):
        """Negative control: without the helper, UFL refuses the derivative."""
        mesh, r = square
        with pytest.raises(ValueError, match="must be restricted"):
            fd.assemble(fd.derivative(r * fd.dS, r))
