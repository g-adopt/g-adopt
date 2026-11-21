"""`GravityPotential` on its own: the checks of its constructor.

`GravityPotential` is the potential part that both `GravitySolver` and the
coupled self-gravitating GIA solver compose. These tests construct it
directly, so that what they check holds for every caller and not only for
the solver that happens to run it.

The first group of tests covers the constructor arguments: exactly one of
`Lambda` and `gravitational_constant`, and a volume density with its own
quadrature degree. The residual is checked against `operator - source`,
the two spellings of the same row.

The checks of the measures of a cross-mesh system (a potential on a parent
mesh, a density on a `Submesh`) run in the constructor. The check that every
DtN boundary is a complete circle or sphere is a policy of the caller: the
GIA solver runs it, and `GravitySolver` accepts such a boundary.
"""

import gadopt  # noqa: F401 - before firedrake
import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, GravitySolver
from gadopt.gravity_potential import GravityPotential


def disk_pair():
    """The unit disc with an inner disc of radius 0.55 as a `Submesh` (tag 99).

    The parent's boundary (tag 1) is the DtN boundary, so the parent holds a
    buffer between the submesh and the boundary.
    """
    mesh = fd.UnitDiskMesh(refinement_level=2)
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    indicator = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(
        fd.conditional(r < 0.55, 1.0, 0.0))
    marked = fd.RelabeledMesh(mesh, [indicator], [99])
    return marked, fd.Submesh(marked, marked.topological_dimension, 99)


@pytest.fixture
def parent_measure_restricted_to_the_submesh(monkeypatch):
    """Make `set_measures` build a parent measure that covers the submesh only.

    The intersected parent measure `dx_g` then misses the buffer between the
    submesh and the DtN boundary. This is the failure the constructor's check
    exists for: the Laplacian would be integrated over part of the parent,
    and the solve would converge to a wrong potential with no error.
    """
    original = GravityPotential.set_measures

    def restricted(self, source_quad_degree=None):
        original(self, source_quad_degree)
        if self.cross_mesh:
            self.dx_g = fd.Measure(
                "dx", domain=self.potential_mesh, subdomain_id=99,
                intersect_measures=(fd.Measure("dx", domain=self.density_mesh),))

    monkeypatch.setattr(GravityPotential, "set_measures", restricted)


class TestCrossMeshMeasures:
    """The constructor refuses an intersected measure that is wrong."""

    def test_correct_measures_pass(self):
        """The plain pair passes, through the part and through the solver."""
        marked, sub = disk_pair()
        V = fd.FunctionSpace(marked, "CG", 1)
        GravityPotential(V, sub, {1: {"dtn": CylindricalDtN(M=2)}},
                         gravitational_constant=1.0)
        rho = fd.Function(fd.FunctionSpace(sub, "DG", 0)).assign(1.0)
        GravitySolver(fd.Function(V), rho, bcs={1: {"dtn": CylindricalDtN(M=2)}})

    def test_the_part_refuses_a_parent_measure_that_misses_the_buffer(
            self, parent_measure_restricted_to_the_submesh):
        marked, sub = disk_pair()
        V = fd.FunctionSpace(marked, "CG", 1)
        with pytest.raises(ValueError, match="does not cover the parent"):
            GravityPotential(V, sub, {1: {"dtn": CylindricalDtN(M=2)}},
                             gravitational_constant=1.0)

    def test_gravity_solver_refuses_it_too(
            self, parent_measure_restricted_to_the_submesh):
        """`GravitySolver` builds the part, so the same check runs for it."""
        marked, sub = disk_pair()
        V = fd.FunctionSpace(marked, "CG", 1)
        rho = fd.Function(fd.FunctionSpace(sub, "DG", 0)).assign(1.0)
        with pytest.raises(ValueError, match="does not cover the parent"):
            GravitySolver(fd.Function(V), rho,
                          bcs={1: {"dtn": CylindricalDtN(M=2)}})


class TestCompleteDtNBoundaries:
    """A DtN tag that matches half of the circle."""

    @staticmethod
    def half_circle_space():
        """CG1 on the unit disc, with the upper half of its boundary as tag 77."""
        mesh = fd.UnitDiskMesh(refinement_level=1)
        X = fd.SpatialCoordinate(mesh)
        marker = fd.Function(
            fd.FunctionSpace(mesh, "HDiv Trace", 0)
        ).interpolate(fd.conditional(X[1] > 0, 1.0, 0.0))
        relabelled = fd.RelabeledMesh(mesh, [marker], [77])
        return fd.FunctionSpace(relabelled, "CG", 1)

    def test_the_check_refuses_a_half_circle(self):
        V = self.half_circle_space()
        part = GravityPotential(V, V.mesh(), {77: {"dtn": CylindricalDtN(M=1)}},
                                gravitational_constant=1.0)
        with pytest.raises(ValueError, match="complete circle or sphere"):
            part.check_dtn_boundaries_complete()

    @pytest.mark.longtest
    def test_the_constructor_does_not_run_it(self):
        """`GravitySolver` keeps its policy: it warns, then builds and solves."""
        V = self.half_circle_space()
        with pytest.warns(UserWarning, match="deviates from a coordinate circle"):
            solver = GravitySolver(fd.Function(V), 0.0,
                                   bcs={77: {"dtn": CylindricalDtN(M=1)}})
        solver.solve()

    def test_a_complete_circle_passes(self):
        marked, sub = disk_pair()
        V = fd.FunctionSpace(marked, "CG", 1)
        part = GravityPotential(V, sub, {1: {"dtn": CylindricalDtN(M=2)}},
                                gravitational_constant=1.0)
        part.check_dtn_boundaries_complete()


def annulus_space():
    """CG2 on a radially extruded annulus, radii 1 and 2, with two DtN boundaries.

    The outer boundary is a 2-D exterior DtN boundary, so the part builds
    the monopole datum and the volume density's net mass enters it.
    """
    base = fd.CircleManifoldMesh(64, radius=1.0, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=6, layer_height=1.0 / 6,
                           extrusion_type="radial")
    return fd.FunctionSpace(mesh, "CG", 2)


def angular_density(mesh):
    """A density with net mass whose integrand is not a polynomial.

    The angular part integrates to zero by symmetry at any degree. The
    exponential is not a polynomial, so its net mass depends on the
    quadrature degree.
    """
    X = fd.SpatialCoordinate(mesh)
    phi = fd.atan2(X[1], X[0])
    return fd.cos(2 * phi) + 0.3 * fd.sin(phi) + fd.exp(0.5 * X[0])


BCS = {"top": {"dtn": CylindricalDtN(M=3)}, "bottom": {"dtn": CylindricalDtN(M=3)}}


class TestConstructorArguments:
    """`Lambda` or `G`, and the volume density."""

    @pytest.mark.parametrize("given", [{}, {"Lambda": 1.0, "gravitational_constant": 1.0}])
    def test_exactly_one_of_lambda_and_g(self, given):
        V = annulus_space()
        with pytest.raises(ValueError, match="exactly one of Lambda and"):
            GravityPotential(V, V.mesh(), BCS, **given)

    @pytest.mark.longtest
    def test_gravitational_constant_gives_lambda_4_pi_g(self):
        """With `G` the part is the part with `Lambda = 4 pi G`.

        The boundary form takes `G` itself, the volume source and the
        displacement source take `4 pi G`, and the residual at a nonzero
        potential is the same vector to round-off.
        """
        V = annulus_space()
        mesh = V.mesh()
        G = 0.7
        rho = angular_density(mesh)
        with_G = GravityPotential(V, mesh, BCS, gravitational_constant=G,
                                  volume_density=rho)
        with_Lambda = GravityPotential(V, mesh, BCS, Lambda=4 * np.pi * G,
                                       volume_density=rho)
        assert float(with_G.form.G) == pytest.approx(G, rel=1e-15)
        assert float(with_G.Lambda) == pytest.approx(4 * np.pi * G, rel=1e-15)
        assert float(with_G.four_pi_G) == pytest.approx(
            float(with_Lambda.four_pi_G), rel=1e-15)

        X = fd.SpatialCoordinate(mesh)
        psi = fd.Function(V).interpolate(X[0] ** 2 - 0.5 * X[1])
        v = fd.TestFunction(V)
        residuals = []
        for part in (with_G, with_Lambda):
            part.update_total_mass()
            residuals.append(fd.assemble(part.residual(psi, v)).dat.data_ro.copy())
        scale = np.max(np.abs(residuals[0]))
        assert scale > 0.0
        assert np.max(np.abs(residuals[0] - residuals[1])) <= 1e-14 * scale

    def test_volume_density_with_its_own_quadrature_degree(self):
        """`source_quad_degree` reaches the volume source and the datum.

        Every integral of the volume source carries the degree, the assembled
        source equals `Lambda int rho v` at that degree, and the datum's
        enclosed mass equals `int rho` at that degree. A third value of the
        mass, at UFL's own estimate, differs, so the degree is not without
        effect on this density.
        """
        V = annulus_space()
        mesh = V.mesh()
        # A low degree, so that the difference from UFL's estimate is far
        # above round-off.
        degree = 2
        Lambda = fd.Constant(2.5)
        rho = angular_density(mesh)
        part = GravityPotential(V, mesh, BCS, Lambda=Lambda, volume_density=rho,
                                source_quad_degree=degree)
        v = fd.TestFunction(V)

        source = part.volume_source(v)
        assert source.integrals()
        for integral in source.integrals():
            assert integral.metadata()["quadrature_degree"] == degree

        dx_q = fd.dx(domain=mesh, degree=degree)
        expected = fd.assemble(Lambda * rho * v * dx_q).dat.data_ro
        assembled = fd.assemble(source).dat.data_ro
        assert np.max(np.abs(assembled - expected)) <= 1e-14 * np.max(np.abs(expected))

        part.update_total_mass()
        mass = float(part.source_mass)
        assert mass == pytest.approx(fd.assemble(rho * dx_q), rel=1e-13)
        assert abs(mass - fd.assemble(rho * fd.dx(domain=mesh))) > 1e-12 * abs(mass)


class TestOperatorAndSource:
    """`residual` is `operator - source` on the low-rank path."""

    def test_residual_equals_operator_minus_source(self):
        V = annulus_space()
        mesh = V.mesh()
        part = GravityPotential(V, mesh, BCS, gravitational_constant=1.0,
                                volume_density=angular_density(mesh))
        part.update_total_mass()
        X = fd.SpatialCoordinate(mesh)
        psi = fd.Function(V).interpolate(X[0] ** 2 - 0.5 * X[1])
        v = fd.TestFunction(V)
        residual = fd.assemble(part.residual(psi, v)).dat.data_ro
        split = fd.assemble(part.operator(psi, v) - part.source(v)).dat.data_ro
        scale = np.max(np.abs(residual))
        assert scale > 0.0
        assert np.max(np.abs(residual - split)) <= 1e-14 * scale
