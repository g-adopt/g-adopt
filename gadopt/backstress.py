"""Illustrative isotropic recovering-backstress rheology for GIA.

This specifies a material model, not a calibration to olivine experiments.
The creep strain includes permanent and delayed recoverable deformation.
"""
import math
from numbers import Real

import firedrake as fd
import ufl

from .approximations import CompressibleInternalVariableApproximation
from .utility import ensure_constant


class BackstressApproximation(CompressibleInternalVariableApproximation):
    """s=2 mu(e-p), X=2 mu_b a; p_dot=chi(s-X)(s-X)/(2 eta).

    a_dot=p_dot-a/tau_r. ``viscosity`` is the unhardened viscosity;
    the linear steady viscosity is eta+mu_b*tau_r. Two history tensors,
    [creep strain, strain-like backstress], are required.
    """
    n_history_tensors = 2
    #: `backstress_history_term` has no boundary part, so a `"u"` or `"un"`
    #: displacement condition is not supported.
    history_supports_displacement_conditions = False

    def __init__(self, bulk_modulus, density, shear_modulus, viscosity, *,
                 backstress_modulus, recovery_time, exponent=1,
                 transition_stress=1, **kwargs):
        for name, value in (("shear_modulus", shear_modulus), ("viscosity", viscosity),
                            ("backstress_modulus", backstress_modulus), ("recovery_time", recovery_time)):
            if isinstance(value, (list, tuple)):
                raise ValueError(f"{name} describes one material, not Maxwell modes")
            if isinstance(value, Real) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(exponent, Real) or not math.isfinite(exponent) or exponent < 1:
            raise ValueError("exponent must be a fixed scalar >= 1")
        if not isinstance(transition_stress, Real) or not math.isfinite(transition_stress) or transition_stress <= 0:
            raise ValueError("transition_stress must be a positive fixed scalar")
        if 'background_stress' in kwargs:
            raise ValueError("Represent initial stress memory with the backstress tensor")
        super().__init__(
            bulk_modulus, density, shear_modulus, viscosity,
            exponent=exponent, transition_stress=transition_stress, background_stress=0., **kwargs)
        self.backstress_modulus = ensure_constant(backstress_modulus)
        self.recovery_time = ensure_constant(recovery_time)

    def deviatoric_stress(self, u, internal_variables):
        p, _ = internal_variables
        return 2*self.mu0*(self.deviatoric_strain(u)-p)

    def backstress(self, a):
        return 2*self.backstress_modulus*a

    def creep_fluidity(self, effective_stress):
        if float(self.exponent) == 1:
            return ufl.as_ufl(1.)
        squared = .5*fd.inner(effective_stress, effective_stress)
        if effective_stress.ufl_shape == (2, 2):
            squared += .5*fd.tr(effective_stress)**2
        return 1+(squared/self.transition_stress**2+1e-24)**((self.exponent-1)/2)

    def power_law_factor(self, effective_stress):
        return 1/self.creep_fluidity(effective_stress)

    def internal_variable_space(self, mesh, degree=1, symmetric=True):
        from .internal_variable_equation import internal_variable_space
        return internal_variable_space(mesh, 2, degree, symmetric)

    def history_terms(self):
        """The creep strain and backstress equation, `backstress_history_term`."""
        return [backstress_history_term]

    def quadrature_update(self, owner):
        """One scalar fluidity root per quadrature point for creep and backstress."""
        from .constitutive_integrators import BackstressQuadratureUpdate
        return BackstressQuadratureUpdate(owner)


def backstress_history_term(eq, trial):
    from .internal_variable_equation import history_slices
    p, a = history_slices(trial)
    old_p, old_a = history_slices(eq.trial_old)
    test_p, test_a = history_slices(eq.test)
    material = eq.approximation
    effective = material.deviatoric_stress(eq.displacement, [p, a])-material.backstress(a)
    rate = material.creep_fluidity(effective)*effective/(2*material.viscosity[0])
    alpha = eq.time_alpha
    return (fd.inner(test_p, (alpha*p-old_p)/eq.dt-rate)
            + fd.inner(test_a, (alpha*a-old_a)/eq.dt-rate+a/material.recovery_time))*eq.dx


backstress_history_term.required_attrs = {'displacement', 'trial_old', 'time_alpha', 'dt'}
# `maxwell_times` is accepted and not read, so that a solver can build the
# history equation of every rheology with one set of attributes.
backstress_history_term.optional_attrs = {'maxwell_times'}
