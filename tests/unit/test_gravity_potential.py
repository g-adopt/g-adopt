"""`GravityPotential` on its own: the checks of its constructor.

`GravityPotential` is the potential part that both `GravitySolver` and the
coupled self-gravitating GIA solver compose. These tests construct it
directly, so that what they check holds for every caller and not only for
the solver that happens to run it.

The checks of the measures of a cross-mesh system (a potential on a parent
mesh, a density on a `Submesh`) run in the constructor. The check that every
DtN boundary is a complete circle or sphere is a policy of the caller: the
GIA solver runs it, and `GravitySolver` accepts such a boundary.
"""

import gadopt  # noqa: F401 - before firedrake
import firedrake as fd
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
