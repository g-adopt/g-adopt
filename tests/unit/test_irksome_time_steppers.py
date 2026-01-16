"""Unit tests for Irksome time stepper integration.

This module contains comprehensive tests for the Irksome integration in G-ADOPT,
including tableau conversion, direct scheme usage, EnergySolver integration,
time stepping functionality, and error handling.
"""

import pytest

from gadopt import *
from gadopt.equations import Equation
from gadopt.scalar_equation import diffusion_term, mass_term, source_term
from gadopt.time_stepper import (
    AbstractRKScheme,
    create_custom_tableau,
    rk_schemes_gadopt,
    rk_schemes_irksome,
)


def gadopt_to_irksome_tableau(scheme: AbstractRKScheme):
    """Convert a G-ADOPT scheme class to an Irksome Butcher tableau.

    Args:
        scheme: A G-ADOPT AbstractRKScheme class

    Returns:
        tuple: (ButcherTableau instance, stage_type string)
    """
    butcher_tableau = scheme.butcher_tableau
    if butcher_tableau is None:
        butcher_tableau = create_custom_tableau(scheme.a, scheme.b, scheme.c)

    return butcher_tableau, scheme.stage_type


def create_irksome_integrator(
    equation: Equation,
    solution: Function,
    dt: float,
    scheme: AbstractRKScheme,
    **kwargs,
):
    """Create an IrksomeIntegrator from a G-ADOPT scheme class.

    Args:
        equation: G-ADOPT equation to integrate
        solution: Firedrake function representing the equation's solution
        t: Integration time
        dt: Integration time step
        scheme: G-ADOPT AbstractRKScheme class
        **kwargs: Additional arguments passed to IrksomeIntegrator

    Returns:
        IrksomeIntegrator instance
    """
    butcher_tableau, stage_type = gadopt_to_irksome_tableau(scheme)

    return IrksomeIntegrator(
        equation, solution, dt, butcher_tableau, stage_type=stage_type, **kwargs
    )


class TestTableauConversion:
    """Test tableau conversion from G-ADOPT to Irksome."""

    @pytest.mark.parametrize("scheme", rk_schemes_gadopt)
    def test_tableau_conversion(self, scheme):
        """Test that scheme classes convert to valid Irksome tableaux."""
        tableau, stage_type = gadopt_to_irksome_tableau(scheme)
        assert tableau is not None
        assert stage_type in ["explicit", "dirk"]

        # Check that tableau has required attributes
        assert hasattr(tableau, "A")  # Butcher matrix (Irksome uses "A" not "a")
        assert hasattr(tableau, "b")  # weights
        assert hasattr(tableau, "c")  # nodes

    def test_tableau_conversion_forward_euler(self):
        """Test specific Forward Euler conversion."""
        tableau, stage_type = gadopt_to_irksome_tableau(ForwardEuler)
        assert stage_type == "explicit"
        assert tableau is not None

    def test_tableau_conversion_dirk33(self):
        """Test specific DIRK33 conversion."""
        tableau, stage_type = gadopt_to_irksome_tableau(DIRK33)
        assert stage_type == "dirk"
        assert tableau is not None


class TestDirectIrksomeSchemes:
    """Test direct Irksome scheme classes."""

    @pytest.mark.parametrize("scheme", rk_schemes_irksome)
    def test_direct_irksome_schemes(self, scheme):
        """Test that direct Irksome schemes can be instantiated."""
        # Create simple setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Test instantiation
        integrator = scheme(equation, u, dt=0.01)

        assert integrator is not None
        assert integrator.solution == u

    @pytest.mark.parametrize(
        "scheme_with_param",
        {
            (GaussLegendre, 3),
            (LobattoIIIA, 3),
            (RadauIIA, 4),
            (LobattoIIIC, 3),
            (PareschiRusso, 1.0),
        },
    )
    def test_direct_irksome_schemes_with_alternative_tableau_parameter(
        self, scheme_with_param
    ):
        """Test that direct Irksome schemes can be instantiated."""
        # Create simple setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Test instantiation
        scheme, tableau_parameter = scheme_with_param
        integrator = scheme(equation, u, dt=0.01, tableau_parameter=tableau_parameter)

        assert integrator is not None
        assert integrator.solution == u


class TestEnergySolverIntegration:
    """Test integration with EnergySolver."""

    @pytest.mark.parametrize(
        "time_stepper",
        [BackwardEuler, ImplicitMidpoint, DIRK33] + rk_schemes_irksome,
    )
    def test_energy_solver_integration(self, time_stepper):
        """Test that time steppers work with EnergySolver."""
        # Create simple setup
        mesh = UnitSquareMesh(5, 5)
        V = VectorFunctionSpace(mesh, "CG", 1)  # Velocity space
        Q = FunctionSpace(mesh, "CG", 1)  # Temperature space
        T = Function(Q)
        u = Function(V)
        u.assign(as_vector((0.0, 0.0)))  # Zero velocity field

        # Create approximation and equation
        Ra = Constant(1000.0)  # Rayleigh number
        approximation = BoussinesqApproximation(Ra)

        # Test EnergySolver creation
        dt = 0.01
        solver = EnergySolver(T, u, approximation, dt, time_stepper)

        assert solver is not None
        assert solver.timestepper is not None


class TestBoundaryConditions:
    """Test that schemes work correctly with boundary conditions."""

    @pytest.mark.parametrize(
        "scheme",
        [ForwardEuler, SSPRK33, eSSPRKs3p3, eSSPRKs10p3, DIRK33, ImplicitMidpoint],
    )
    def test_schemes_with_dirichlet_bcs(self, scheme):
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
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Create Dirichlet boundary conditions
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]

        # Create integrator with boundary conditions
        dt = 0.01
        tableau, stage_type = gadopt_to_irksome_tableau(scheme)
        integrator = IrksomeIntegrator(
            equation, u, dt, tableau, stage_type=stage_type, strong_bcs=bcs
        )

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))

        # Take a time step - this should not raise an error
        integrator.advance()

        # Check that solution evolved and BCs are satisfied
        assert norm(u) > 0

    def test_explicit_scheme_with_bcs_stage_type(self):
        """Verify that explicit schemes get stage_type='explicit' not 'deriv'."""
        # Test eSSPRKs10p3 specifically (the scheme that triggered the original bug)
        stage_type = gadopt_to_irksome_tableau(eSSPRKs10p3)[1]
        assert stage_type == "explicit", (
            f"eSSPRKs10p3 should have stage_type='explicit', got '{stage_type}'"
        )

        # Test a few more explicit schemes
        for scheme in [ForwardEuler, SSPRK33, eSSPRKs3p3]:
            stage_type = gadopt_to_irksome_tableau(scheme)[1]
            assert stage_type == "explicit", (
                f"{scheme.__name__} should have stage_type='explicit', got '{stage_type}'"
            )


class TestTimeStepping:
    """Test actual time stepping functionality."""

    @pytest.mark.parametrize(
        "scheme", [ForwardEuler, eSSPRKs3p3, DIRK33, ImplicitMidpoint]
    )
    def test_time_stepping(self, scheme):
        """Test actual time stepping with different schemes."""
        # Create setup
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        # Create equation
        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Create integrator
        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme=scheme)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))

        # Take a few time steps
        initial_norm = norm(u)
        for _ in range(3):
            integrator.advance()

        # Check that solution evolved (should be different from initial)
        final_norm = norm(u)
        assert not abs(final_norm - initial_norm) < 1e-10  # Should have evolved


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
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Create integrator with Constant dt
        dt = Constant(0.01)
        integrator = create_irksome_integrator(equation, u, dt, scheme=ForwardEuler)

        # Set initial condition
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))

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
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        dt = Constant(0.01)
        integrator = create_irksome_integrator(equation, u, dt, scheme=DIRK33)

        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))

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
        with pytest.raises(AttributeError):
            # Try to convert a non-scheme class
            gadopt_to_irksome_tableau(str)

    def test_invalid_order_parameter(self):
        """Test that invalid order parameters are handled."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Test with invalid order (should raise AssertionError from Irksome)
        with pytest.raises(AssertionError):
            _ = RadauIIA(equation, u, dt=0.01, tableau_parameter=0)

    def test_very_small_time_step(self):
        """Test behavior with very small time step."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Test with very small dt
        integrator = create_irksome_integrator(
            equation, u, dt=1e-10, scheme=ForwardEuler
        )
        assert integrator is not None

        # Initialisation should work
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))

        # Advance with very small dt should not crash
        integrator.advance()

        # Solution should have changed very little
        assert norm(u) > 0


class TestIntegrationWithExistingSchemes:
    """Test that existing G-ADOPT schemes still work with Irksome backend."""

    @pytest.mark.parametrize("scheme", rk_schemes_gadopt)
    def test_existing_schemes_still_work(self, scheme):
        """Test that existing schemes still work with Irksome backend."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Create integrator using existing scheme
        integrator = create_irksome_integrator(equation, u, dt=0.01, scheme=scheme)

        # Set initial condition and advance
        x = SpatialCoordinate(mesh)
        u.interpolate(sin(pi * x[0]) * sin(pi * x[1]))
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
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        # Test with custom solver parameters
        solver_params = {"ksp_type": "gmres", "ksp_rtol": 1e-6, "pc_type": "ilu"}
        integrator = create_irksome_integrator(
            equation, u, dt=0.01, scheme=DIRK33, solver_parameters=solver_params
        )

        assert integrator is not None
        # The solver parameters are passed to the Irksome stepper
        assert hasattr(integrator, "stepper")

    def test_direct_irksome_solver_parameters(self):
        """Test solver parameters with direct Irksome schemes."""
        mesh = UnitSquareMesh(5, 5)
        V = FunctionSpace(mesh, "CG", 1)
        u = Function(V)

        test = TestFunction(V)
        eq_attrs = {"diffusivity": Constant(1.0), "source": Constant(0.0)}
        equation = Equation(
            test,
            V,
            residual_terms=[diffusion_term, source_term],
            mass_term=mass_term,
            eq_attrs=eq_attrs,
        )

        solver_params = {"ksp_type": "cg", "pc_type": "jacobi"}
        integrator = RadauIIA(equation, u, dt=0.01, solver_parameters=solver_params)

        assert integrator is not None
