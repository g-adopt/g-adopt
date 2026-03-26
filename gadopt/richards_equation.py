import firedrake as fd

import ufl
from abc import ABC, abstractmethod
from typing import Dict, Any
from .utility import CombinedSurfaceMeasure


class RichardsEquation(ABC):

    """
    Base class for Richards equation solvers. 
    Handles: 
       - mesh and function space setup 
       - facet/volume measures (extruded or not) 
       - solver parameter selection 
    """

    def __init__(self,
                 V: fd.FunctionSpace,
                 soil_curves: Dict,
                 bcs: Dict,
                 solver_parameters='default',
                 time_integrator="BackwardEuler",
                 source_term=0,
                 quad_degree=0,
                 equation_form="MixedForm"):

        self.mesh = mesh = V.mesh()
        self.trial_space = V
        self.test_function = fd.TestFunction(V)

        self.dim = mesh.topological_dimension()
        self.n = fd.FacetNormal(mesh)

        # Check time integration method is valid
        accepted_solvers = ['BackwardEuler', 'CrankNicolson', 'Picard', 'ImplicitMidpoint']
        if time_integrator in accepted_solvers:
            self.time_integrator = time_integrator
        else:
            raise TypeError('Time Integrator not recognised')
        self.soil_curves = soil_curves

        # Check equation form is valid
        accepted_forms = ['MixedForm', 'PressureHeadForm']
        if equation_form in accepted_forms:
            self.equation_form = equation_form
        else:
            raise TypeError('Equation form not recognised')

        self.bcs = bcs

        self.source_term = source_term

        # Ensure quadrature degree is sufficient to integrate the product of trial and test functions (2*k) plus additional points for nonlinear soil curves.
        if quad_degree == 0:      
            degree = V.ufl_element().degree()
            if not isinstance(degree, int):
                degree = max(degree)
            quad_degree = 2 * degree + 1
        self.quad_degree = quad_degree

        # Measures (extruded vs non-extruded)
        measure_kwargs = {"domain": self.mesh, "degree": quad_degree}
        self.dx = fd.dx(**measure_kwargs)

        if self.trial_space.extruded:
            self.ds = CombinedSurfaceMeasure(**measure_kwargs)
            self.dS = fd.dS_v(**measure_kwargs) + fd.dS_h(**measure_kwargs)
        else:
            self.dS = fd.Measure("dS", domain=mesh, metadata={"quadrature_degree": quad_degree})
            self.ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": quad_degree})

        if solver_parameters == "default":
            if mesh.topological_dimension() <= 2:
                solver_parameters = 'direct' # Direct solvers for 2D
            else:
                solver_parameters = 'iterative' # Iterative for 3D

        if solver_parameters == 'direct':
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                'snes_type': 'newtonls',
                "pc_factor_mat_solver_type": "mumps",
                "snes_force_iteration": True,
                "snes_linesearch_type": "bt",
                'snes_atol': 1e-14,
                }
        elif solver_parameters == "iterative":
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": 'gmres',
                "pc_type": 'bjacobi',
                'snes_type': 'newtonls',
                "snes_force_iteration": True,
                "snes_linesearch_type": "bt",
                'snes_monitor': None,
                }
        # User specified solver parameters
        else:
            self.solver_parameters = solver_parameters
