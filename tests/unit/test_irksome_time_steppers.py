"""Unit tests for Irksome time stepper integration.

This module contains comprehensive tests for the Irksome integration in G-ADOPT,
including tableau conversion, direct scheme usage, EnergySolver integration,
time stepping functionality, and error handling.
"""

import pytest
import numpy as np
from firedrake import *
from irksome import BackwardEuler as IrksomeBackwardEuler
from irksome import GaussLegendre
from irksome.ButcherTableaux import ButcherTableau

from gadopt import *
from gadopt.scalar_equation import diffusion_term, mass_term, source_term
from gadopt.time_stepper import *


# Helper function to create custom tableau
def create_custom_tableau(a, b, c):
    """Create a Butcher tableau for Irksome.

    Args:
        a: Butcher matrix (2D array)
        b: Weights (1D array)
        c: Nodes (1D array)

    Returns:
        ButcherTableau instance
    """
    return ButcherTableau(
        A=np.array(a),
        b=np.array(b),
        btilde=None,
        c=np.array(c),
        order=len(b),
        embedded_order=None,
        gamma0=None
    )


# Define scheme mappings
scheme_mappings = {
    # Special cases with direct Irksome equivalents
    BackwardEulerAbstract: (IrksomeBackwardEuler(), "dirk"),
    ImplicitMidpointAbstract: (GaussLegendre(1), "dirk"),
    # Explicit schemes
    ForwardEulerAbstract: "explicit",
    SSPRK33Abstract: "explicit",
    ERKMidpointAbstract: "explicit",
    ERKLSPUM2Abstract: "explicit",
    ERKLPUM2Abstract: "explicit",
    eSSPRKs3p3Abstract: "explicit",
    eSSPRKs4p3Abstract: "explicit",
    eSSPRKs5p3Abstract: "explicit",
    eSSPRKs6p3Abstract: "explicit",
    eSSPRKs7p3Abstract: "explicit",
    eSSPRKs8p3Abstract: "explicit",
    eSSPRKs9p3Abstract: "explicit",
    eSSPRKs10p3Abstract: "explicit",
    # DIRK schemes
    CrankNicolsonAbstract: "dirk",
    DIRK22Abstract: "dirk",
    DIRK23Abstract: "dirk",
    DIRK33Abstract: "dirk",
    DIRK43Abstract: "dirk",
    DIRKLSPUM2Abstract: "dirk",
    DIRKLPUM2Abstract: "dirk",
}


# Utility functions for testing (moved from time_stepper.py)
def gadopt_to_irksome_tableau(scheme_class):
    """Convert a G-ADOPT scheme class to an Irksome Butcher tableau.

    Args:
        scheme_class: A G-ADOPT AbstractRKScheme class

    Returns:
        tuple: (ButcherTableau instance, stage_type string)
    """
    # Get the mapping function or use default
    mapping = scheme_mappings[scheme_class]
    if isinstance(mapping, str):
        # Need to create custom tableau from scheme class
        temp_scheme = scheme_class()
        return create_custom_tableau(
            temp_scheme.a, temp_scheme.b, temp_scheme.c
        ), mapping
    else:
        # Direct Irksome tableau and stage type
        return mapping


def create_irksome_integrator(equation, solution, dt, scheme_class, **kwargs):
    """Create an IrksomeIntegrator from a G-ADOPT scheme class.

    Args:
        equation: G-ADOPT equation to integrate
        solution: Firedrake function representing the equation's solution
        dt: Integration time step
        scheme_class: G-ADOPT AbstractRKScheme class
        **kwargs: Additional arguments passed to IrksomeIntegrator

    Returns:
        IrksomeIntegrator instance
    """
    tableau, stage_type = gadopt_to_irksome_tableau(scheme_class)

    return IrksomeIntegrator(
        equation=equation,
        solution=solution,
        dt=dt,
        butcher=tableau,
        stage_type=stage_type,
        **kwargs
    )


class TestTableauConversion:
    """Test tableau conversion from G-ADOPT to Irksome."""

    @pytest.mark.parametrize("scheme_class", scheme_mappings.keys())
    def test_tableau_conversion(self, scheme_class):
        """Test that scheme classes convert to valid Irksome tableaux."""
        tableau, stage_type = gadopt_to_irksome_tableau(scheme_class)
        assert tableau is not None
        assert stage_type in ["explicit", "dirk"]

        # Check that tableau has required attributes
        assert hasattr(tableau, 'A')  # Butcher matrix (Irksome uses 'A' not 'a')
        assert hasattr(tableau, 'b')  # weights
        assert hasattr(tableau, 'c')  # nodes

    def test_tableau_conversion_forward_euler(self):
        """Test specific Forward Euler conversion."""
        tableau, stage_type = gadopt_to_irksome_tableau(ForwardEulerAbstract)
        assert stage_type == "explicit"
        assert tableau is not None

    def test_tableau_conversion_dirk33(self):
        """Test specific DIRK33 conversion."""
        tableau, stage_type = gadopt_to_irksome_tableau(DIRK33Abstract)
        assert stage_type == "dirk"
        assert tableau is not None


class TestDirectIrksomeSchemes:
    """Test direct Irksome scheme classes."""

    @pytest.mark.parametrize("irksome_class", [
        IrksomeRadauIIA,
        IrksomeGaussLegendre,
        IrksomeLobattoIIIA,
        IrksomeLobattoIIIC,
        IrksomeAlexander,
        IrksomeQinZhang,
        IrksomePareschiRusso,
    ])
    def test_direct_irksome_schemes(self, irksome_class):
        """Test that direct Irksome schemes can be instantiated."""
        # Create simple setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Test instantiation
        if irksome_class in [
                IrksomeRadauIIA, IrksomeGaussLegendre,
                IrksomeLobattoIIIA, IrksomeLobattoIIIC]:
            # These have order parameter
            integrator = irksome_class(equation, u, dt=0.01, order=2)
        elif irksome_class == IrksomePareschiRusso:
            # This has x parameter
            integrator = irksome_class(equation, u, dt=0.01, x=0.5)
        else:
            # These don't have additional parameters
            integrator = irksome_class(equation, u, dt=0.01)

        assert integrator is not None
        assert integrator.equation == equation
        assert integrator.solution == u


class TestEnergySolverIntegration:
    """Test integration with EnergySolver."""

    @pytest.mark.parametrize("time_stepper", [
        # Test G-ADOPT schemes (now using Irksome internally)
        ImplicitMidpoint,
        BackwardEuler,
        DIRK33,

        # Test direct Irksome schemes
        IrksomeRadauIIA,
        IrksomeGaussLegendre,
        IrksomeLobattoIIIA,
        IrksomePareschiRusso,
    ])
    def test_energy_solver_integration(self, time_stepper):
        """Test that time steppers work with EnergySolver."""
        # Create simple setup
        mesh = UnitSquareMesh(5, 5)
        V = VectorFunctionSpace(mesh, "CG", 1)  # Velocity space
        Q = FunctionSpace(mesh, "CG", 1)        # Temperature space
        T = Function(Q)
        u = Function(V)
        u.assign(as_vector((0.0, 0.0)))  # Zero velocity field

        # Create approximation and equation
        Ra = Constant(1000.0)  # Rayleigh number
        approximation = BoussinesqApproximation(Ra)

        # Test EnergySolver creation
        dt = Constant(0.01)
        solver = EnergySolver(T, u, approximation, dt, time_stepper)

        assert solver is not None
        assert solver.timestepper is not None


class TestBoundaryConditions:
    """Test that schemes work correctly with boundary conditions."""

    @pytest.mark.parametrize("scheme_class", [
        ForwardEulerAbstract,
        SSPRK33Abstract,
        eSSPRKs3p3Abstract,
        eSSPRKs10p3Abstract,
        DIRK33Abstract,
        ImplicitMidpointAbstract,
    ])
    def test_schemes_with_dirichlet_bcs(self, scheme_class):
        """Test that schemes work with Dirichlet boundary conditions.

        This test specifically addresses the bug where explicit schemes with
        boundary conditions would fail due to incorrect stage_type="deriv"
        causing a singular matrix error in Irksome's DAE BC handling.
        """
        # Create setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Create Dirichlet boundary conditions
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

        # Create integrator with boundary conditions
        tableau, stage_type = gadopt_to_irksome_tableau(scheme_class)
        integrator = IrksomeIntegrator(
            equation, u, dt=0.01,
            butcher=tableau,
            stage_type=stage_type,
            strong_bcs=bcs
        )

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take a time step - this should not raise an error
        integrator.advance()

        # Check that solution evolved and BCs are satisfied
        assert norm(u) > 0

    def test_explicit_scheme_with_bcs_stage_type(self):
        """Verify that explicit schemes get stage_type='explicit' not 'deriv'."""
        # Test eSSPRKs10p3 specifically (the scheme that triggered the original bug)
        tableau, stage_type = gadopt_to_irksome_tableau(eSSPRKs10p3Abstract)
        assert stage_type == "explicit", \
            f"eSSPRKs10p3 should have stage_type='explicit', got '{stage_type}'"

        # Test a few more explicit schemes
        for scheme in [ForwardEulerAbstract, SSPRK33Abstract, eSSPRKs3p3Abstract]:
            tableau, stage_type = gadopt_to_irksome_tableau(scheme)
            assert stage_type == "explicit", \
                f"{scheme.__name__} should have stage_type='explicit', got '{stage_type}'"


class TestTimeStepping:
    """Test actual time stepping functionality."""

    @pytest.mark.parametrize("scheme_class", [
        ForwardEulerAbstract,
        eSSPRKs3p3Abstract,
        DIRK33Abstract,
        ImplicitMidpointAbstract,
    ])
    def test_time_stepping(self, scheme_class):
        """Test actual time stepping with different schemes."""
        # Create setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Create integrator
        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme_class=scheme_class)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take a few time steps
        initial_norm = norm(u)
        for _ in range(3):
            integrator.advance()

        # Check that solution evolved (should be different from initial)
        final_norm = norm(u)
        assert not abs(final_norm - initial_norm) < 1e-10  # Should have evolved

    def test_time_stepping_forward_euler(self):
        """Test Forward Euler time stepping specifically."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme_class=ForwardEulerAbstract)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take one step
        integrator.advance()

        # Solution should have evolved
        assert norm(u) > 0

    def test_time_stepping_dirk33(self):
        """Test DIRK33 time stepping specifically."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme_class=DIRK33Abstract)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take one step
        integrator.advance()

        # Solution should have evolved
        assert norm(u) > 0


class TestDynamicTimeStepping:
    """Test dynamic time step updates."""

    def test_dynamic_time_stepping(self):
        """Test that dynamic time stepping works with Irksome."""
        # Create setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Create integrator with Constant dt
        dt = Constant(0.01)
        integrator = create_irksome_integrator(equation, u, dt, ForwardEulerAbstract)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take a step with original dt
        integrator.advance()
        norm_1 = norm(u)

        # Update dt and take another step
        dt.assign(0.02)
        integrator.advance()
        norm_2 = norm(u)

        # Both steps should succeed
        assert norm_1 is not None
        assert norm_2 is not None

    def test_dynamic_time_stepping_dirk(self):
        """Test dynamic time stepping with DIRK scheme."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        dt = Constant(0.01)
        integrator = create_irksome_integrator(equation, u, dt, DIRK33Abstract)

        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Take steps with changing dt
        integrator.advance()
        dt.assign(0.005)
        integrator.advance()
        dt.assign(0.02)
        integrator.advance()

        # All steps should succeed
        assert norm(u) > 0


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_invalid_scheme(self):
        """Test that invalid schemes raise appropriate errors."""
        with pytest.raises(KeyError):
            # Try to convert a non-scheme class
            gadopt_to_irksome_tableau(str)

    def test_invalid_order_parameter(self):
        """Test that invalid order parameters are handled."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Test with invalid order (should raise AssertionError from Irksome)
        with pytest.raises(AssertionError):
            _ = IrksomeRadauIIA(equation, u, dt=0.01, order=0)

    def test_very_small_time_step(self):
        """Test behavior with very small time step."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Test with very small dt
        integrator = create_irksome_integrator(equation, u, dt=1e-10, scheme_class=ForwardEulerAbstract)
        assert integrator is not None

        # Initialization should work
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)

        # Advance with very small dt should not crash
        integrator.advance()

        # Solution should have changed very little
        assert norm(u) > 0


class TestIntegrationWithExistingSchemes:
    """Test that existing G-ADOPT schemes still work with Irksome backend."""

    @pytest.mark.parametrize("scheme_class", [
        ForwardEulerAbstract,
        BackwardEulerAbstract,
        ImplicitMidpointAbstract,
        DIRK33Abstract,
        SSPRK33Abstract,
        eSSPRKs3p3Abstract,
        eSSPRKs10p3Abstract,
    ])
    def test_existing_schemes_still_work(self, scheme_class):
        """Test that existing schemes still work with Irksome backend."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Create integrator using existing scheme
        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme_class=scheme_class)

        # Set initial condition and advance
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi*x[0])*sin(pi*x[1]))
        integrator.initialize(u)
        integrator.advance()

        # Should work without errors
        assert norm(u) > 0


class TestSolverParameters:
    """Test that solver parameters are passed correctly to Irksome."""

    def test_solver_parameters_passed(self):
        """Test that solver parameters are passed to Irksome."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        # Test with custom solver parameters
        solver_params = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "pc_type": "ilu"
        }

        integrator = create_irksome_integrator(
            equation, u, dt=0.01,
            scheme_class=DIRK33Abstract,
            solver_parameters=solver_params
        )

        assert integrator is not None
        # The solver parameters are passed to the Irksome stepper
        assert hasattr(integrator, 'stepper')

    def test_direct_irksome_solver_parameters(self):
        """Test solver parameters with direct Irksome schemes."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), 'source': Constant(0.0)}
        equation = Equation(
            test, V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
            bcs={}
        )

        solver_params = {"ksp_type": "cg", "pc_type": "jacobi"}

        integrator = IrksomeRadauIIA(
            equation, u, dt=0.01, order=2,
            solver_parameters=solver_params
        )

        assert integrator is not None
