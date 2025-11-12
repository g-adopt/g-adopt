import firedrake as fd
from gadopt import equations
import ufl
from abc import ABC, abstractmethod
from typing import Dict, Any


class richards_equation(ABC):

    def __init__(self, V, mesh, quad_degree):

        self.mesh = mesh
        self.trial_space = V
        self.quad_degree = quad_degree
        self.v = fd.TestFunction(V)
        self.dimen = mesh.topological_dimension()
        self.n = fd.FacetNormal(mesh)

        measure_kwargs = {"domain": self.mesh, "degree": quad_degree}
        
        self.dx = fd.dx(**measure_kwargs)
        
        if self.trial_space.extruded:
            self.ds = equations.CombinedSurfaceMeasure(**measure_kwargs)
            self.dS = fd.dS_v(**measure_kwargs) + fd.dS_h(**measure_kwargs)
        else:
            self.dS = fd.Measure("dS", domain=mesh, metadata={"quadrature_degree": quad_degree})
            self.ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": quad_degree})