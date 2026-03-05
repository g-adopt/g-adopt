"""Tests for adjoint tape recording of GplatesVelocityFunction and GplatesScalarFunction.

Verifies that externally-loaded plate reconstruction data is correctly
recorded on the pyadjoint tape via create_block_variable(), that tape
replay reproduces forward results, and that adjoint derivatives are correct.

Uses mock connectors to avoid requiring actual plate reconstruction data.
All tests run on a spherical shell mesh to match the intended geometry.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import firedrake as fd
from firedrake.adjoint import *
from pyadjoint.tape import (
    get_working_tape, set_working_tape, Tape,
    continue_annotation, pause_annotation,
)

from gadopt.gplates import (
    GplatesVelocityFunction,
    GplatesScalarFunction,
    IndicatorConnector,
)


@pytest.fixture(autouse=True)
def fresh_tape():
    """Ensure each test gets a fresh tape with annotation enabled."""
    continue_annotation()
    set_working_tape(Tape())
    yield
    pause_annotation()


@pytest.fixture
def spherical_shell_mesh():
    """Spherical shell mesh from r=1.22 to r=2.22 (typical G-ADOPT geometry)."""
    mesh2d = fd.IcosahedralSphereMesh(
        radius=1.22, refinement_level=1, degree=1
    )
    return fd.ExtrudedMesh(
        mesh2d, layers=2, layer_height=0.5, extrusion_type="radial"
    )


# ---------------------------------------------------------------------------
# Mock connectors
# ---------------------------------------------------------------------------

class MockGplatesConnector:
    """Minimal mock for pyGplatesConnector providing time conversion and
    synthetic plate velocities for testing."""

    def __init__(self, oldest_age=200.0, delta_t=2.0, kappa=1e-6):
        self.oldest_age = oldest_age
        self.delta_t = delta_t
        self.kappa = kappa
        self.reconstruction_age = None
        # Simple linear time mapping: 100 Ma per ndtime unit
        self._time_factor = 100.0

    def ndtime2age(self, ndtime):
        return self.oldest_age - float(ndtime) * self._time_factor

    def age2ndtime(self, age):
        return (self.oldest_age - age) / self._time_factor

    def get_plate_velocities(self, target_coords, ndtime):
        """Return synthetic tangential velocities (rigid rotation about z-axis).

        This produces a velocity field that is tangential to the sphere
        everywhere, so it survives the radial component removal step.
        """
        self.reconstruction_age = self.ndtime2age(ndtime)
        z_axis = np.array([0.0, 0.0, 1.0])
        return np.cross(target_coords, z_axis) * 0.01


class MockIndicatorConnector(IndicatorConnector):
    """Indicator connector returning a controllable constant field.

    Tracks call count so tests can verify caching behaviour.
    """

    def __init__(self, gplates_connector, value=1.0, comm=None):
        self.gplates_connector = gplates_connector
        self.comm = comm
        self.reconstruction_age = None
        self._value = value
        self.get_indicator_call_count = 0

    def set_value(self, value):
        self._value = value

    def get_indicator(self, target_coords, ndtime):
        age = self.ndtime2age(ndtime)
        self.get_indicator_call_count += 1
        self.reconstruction_age = age
        return np.full(len(target_coords), self._value)


# ===========================================================================
# GplatesScalarFunction tests
# ===========================================================================

class TestGplatesScalarFunctionTapeReplay:
    """Tape replay should reproduce forward results after loading external data."""

    def test_replay_at_same_values(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=5.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(sf)
        J = fd.assemble(g * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        sf_replay = GplatesScalarFunction(V)
        sf_replay.assign(5.0)
        J_replayed = rf(sf_replay)
        assert_allclose(float(J), float(J_replayed))

    def test_replay_at_different_values(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=5.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(2.0 * sf)
        J = fd.assemble(g * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        # sf=10 -> g=20 -> J = 20 * volume
        sf_new = GplatesScalarFunction(V)
        sf_new.assign(10.0)
        J_new = rf(sf_new)
        volume = fd.assemble(fd.Constant(1.0) * fd.dx(domain=spherical_shell_mesh))
        assert_allclose(float(J_new), 20.0 * float(volume), rtol=1e-10)


class TestGplatesScalarFunctionDerivative:
    """Taylor tests verifying adjoint derivatives of the scalar function."""

    def test_taylor_linear(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=3.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(sf)
        J = fd.assemble(g ** 2 * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        h = fd.Function(V).assign(1.0)
        assert taylor_test(rf, sf, h) > 1.9

    def test_taylor_nonlinear(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=2.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(sf)
        J = fd.assemble((g ** 3 + g) * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        h = fd.Function(V).assign(1.0)
        assert taylor_test(rf, sf, h) > 1.9

    def test_taylor_to_dict_convergence(self, spherical_shell_mesh):
        """Full R0, R1, R2 convergence analysis."""
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=2.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(sf)
        # Use g**4 so the Taylor remainder after the Hessian term is non-zero.
        # (g**2 is exactly quadratic, giving R2 residuals at machine precision.)
        J = fd.assemble(g ** 4 * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        h = fd.Function(V).assign(1.0)
        r = taylor_to_dict(rf, sf, h)

        assert min(r["R0"]["Rate"]) > 0.95
        assert min(r["R1"]["Rate"]) > 1.95
        assert min(r["R2"]["Rate"]) > 2.95


class TestGplatesScalarFunctionDeltaTCaching:
    """Verify the early-return within delta_t windows prevents redundant
    block variable creation and that adjoint sensitivities accumulate."""

    def test_no_update_within_delta_t(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=5.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )

        # ndtime=0 -> age=200 Ma
        sf.update_plate_reconstruction(ndtime=0.0)
        bv_1 = sf.block_variable
        assert mock_conn.get_indicator_call_count == 1

        # ndtime=0.01 -> age=199 Ma, within delta_t=2.0
        sf.update_plate_reconstruction(ndtime=0.01)
        bv_2 = sf.block_variable
        assert mock_conn.get_indicator_call_count == 1
        assert bv_1 is bv_2

    def test_update_outside_delta_t(self, spherical_shell_mesh):
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=5.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )

        # ndtime=0 -> age=200 Ma
        sf.update_plate_reconstruction(ndtime=0.0)
        bv_1 = sf.block_variable
        assert mock_conn.get_indicator_call_count == 1

        # ndtime=0.05 -> age=195 Ma, outside delta_t=2.0
        mock_conn.set_value(7.0)
        sf.update_plate_reconstruction(ndtime=0.05)
        bv_2 = sf.block_variable
        assert mock_conn.get_indicator_call_count == 2
        assert bv_1 is not bv_2

    def test_no_tape_blocks_within_window(self, spherical_shell_mesh):
        """Repeated calls within delta_t must not add blocks to the tape."""
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=1.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )
        sf.update_plate_reconstruction(ndtime=0.0)

        tape = get_working_tape()
        n_blocks = len(tape.get_blocks())

        sf.update_plate_reconstruction(ndtime=0.005)
        sf.update_plate_reconstruction(ndtime=0.01)
        sf.update_plate_reconstruction(ndtime=0.015)

        assert len(tape.get_blocks()) == n_blocks

    def test_adjoint_accumulates_within_window(self, spherical_shell_mesh):
        """Within delta_t, both uses share a single block variable so
        adjoint contributions from each accumulate correctly."""
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=2.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )

        sf.update_plate_reconstruction(ndtime=0.0)
        u1 = fd.Function(V)
        u1.assign(sf)

        # Within delta_t — early return, same block variable
        sf.update_plate_reconstruction(ndtime=0.01)
        u2 = fd.Function(V)
        u2.assign(3.0 * sf)

        J = fd.assemble((u1 ** 2 + u2 ** 2) * fd.dx)
        # u1 = sf, u2 = 3*sf, same block variable
        # dJ/dsf = 2*sf*1 + 2*(3*sf)*3 = 2*sf + 18*sf = 20*sf = 40

        rf = ReducedFunctional(J, Control(sf))
        dJdf = rf.derivative(apply_riesz=True)
        assert_allclose(dJdf.dat.data_ro, 40.0, rtol=1e-10)

    def test_adjoint_accumulates_within_window_taylor(self, spherical_shell_mesh):
        """Taylor convergence for the accumulation scenario."""
        V = fd.FunctionSpace(spherical_shell_mesh, "CG", 1)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)
        mock_conn = MockIndicatorConnector(gpc, value=2.0)

        sf = GplatesScalarFunction(
            V, indicator_connector=mock_conn, name="indicator"
        )

        sf.update_plate_reconstruction(ndtime=0.0)
        u1 = fd.Function(V)
        u1.assign(sf)

        sf.update_plate_reconstruction(ndtime=0.01)
        u2 = fd.Function(V)
        u2.assign(3.0 * sf)

        J = fd.assemble((u1 ** 2 + u2 ** 2) * fd.dx)

        rf = ReducedFunctional(J, Control(sf))
        h = fd.Function(V).assign(1.0)
        assert taylor_test(rf, sf, h) > 1.9


# ===========================================================================
# GplatesVelocityFunction tests
# ===========================================================================

class TestGplatesVelocityFunctionTapeReplay:
    """Tape replay of the velocity function after loading external data."""

    def test_replay_after_update(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )
        vf.update_plate_reconstruction(ndtime=0.0)

        J = fd.assemble(fd.inner(vf, vf) * fd.ds_t)

        rf = ReducedFunctional(J, Control(vf))
        vf_copy = GplatesVelocityFunction(V)
        vf_copy.assign(vf)
        J_replayed = rf(vf_copy)

        assert_allclose(float(J), float(J_replayed), rtol=1e-10)

    def test_replay_at_different_values(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )
        vf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(2.0 * vf)
        J = fd.assemble(fd.inner(g, g) * fd.ds_t)

        rf = ReducedFunctional(J, Control(vf))

        # Evaluate with a zero velocity -> J should be 0
        J_zero = rf(GplatesVelocityFunction(V))
        assert_allclose(float(J_zero), 0.0, atol=1e-14)


class TestGplatesVelocityFunctionDerivative:
    """Taylor tests for adjoint derivatives of the velocity function."""

    def test_taylor_test(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )
        vf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(vf)
        J = fd.assemble(fd.inner(g, g) * fd.ds_t)

        rf = ReducedFunctional(J, Control(vf))
        h = fd.Function(V).assign(1.0)
        assert taylor_test(rf, vf, h) > 1.9

    def test_taylor_with_scaling(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )
        vf.update_plate_reconstruction(ndtime=0.0)

        g = fd.Function(V)
        g.assign(3.0 * vf)
        J = fd.assemble(fd.inner(g, g) * fd.ds_t)

        rf = ReducedFunctional(J, Control(vf))
        h = fd.Function(V).assign(1.0)
        assert taylor_test(rf, vf, h) > 1.9


class TestGplatesVelocityFunctionDeltaTCaching:
    """Verify the velocity function's early-return within delta_t."""

    def test_no_update_within_delta_t(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )

        # ndtime=0 -> age=200 Ma
        vf.update_plate_reconstruction(ndtime=0.0)
        bv_1 = vf.block_variable

        # ndtime=0.01 -> age=199 Ma, within delta_t=2.0
        vf.update_plate_reconstruction(ndtime=0.01)
        bv_2 = vf.block_variable

        assert bv_1 is bv_2

    def test_update_outside_delta_t(self, spherical_shell_mesh):
        V = fd.VectorFunctionSpace(spherical_shell_mesh, "CG", 2)
        gpc = MockGplatesConnector(oldest_age=200.0, delta_t=2.0)

        vf = GplatesVelocityFunction(
            V, gplates_connector=gpc, name="GplateVelocity"
        )

        # ndtime=0 -> age=200 Ma
        vf.update_plate_reconstruction(ndtime=0.0)
        bv_1 = vf.block_variable

        # ndtime=0.05 -> age=195 Ma, outside delta_t=2.0
        vf.update_plate_reconstruction(ndtime=0.05)
        bv_2 = vf.block_variable

        assert bv_1 is not bv_2
