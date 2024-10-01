r"""This module contains abstract classes to define the structure of mathematical terms and
equations within the G-ADOPT library. Users should not interact with these classes;
instead, please use the solvers provided in other modules.

"""

from abc import ABC, abstractmethod
from numbers import Number
from typing import Optional

import firedrake

from .utility import CombinedSurfaceMeasure

__all__ = ["BaseEquation", "BaseTerm"]


class BaseEquation:
    """Produces the UFL for the registered terms constituting an equation.

    The equation instance is initialised given test and trial function
    spaces.  Test and trial spaces are only used to determine the
    employed discretisation (i.e. UFL elements); test and trial
    functions are provided separately in the residual method.

    Keyword arguments provided here are passed on to each collected equation term.

    Attributes:
      terms (list[BaseTerm]): List of equation terms defined through inheritance from BaseTerm.

    Arguments:
      test_space:  Firedrake function space of the test function
      trial_space: Firedrake function space of the trial function
      quad_degree:
        Quadrature degree. Default value is `2p + 1`, where p the polynomial degree of the trial space
      prefactor:   Constant prefactor multiplying all terms in the equation

    """
    terms = []

    def __init__(
        self,
        test_space: firedrake.functionspaceimpl.WithGeometry,
        trial_space: firedrake.functionspaceimpl.WithGeometry,
        quad_degree: Optional[int] = None,
        prefactor: Optional[Number | firedrake.Constant] = 1,
        **kwargs
    ):
        self.test_space = test_space
        self.trial_space = trial_space
        self.mesh = trial_space.mesh()

        p = trial_space.ufl_element().degree()
        if isinstance(p, int):  # isotropic element
            if quad_degree is None:
                quad_degree = 2*p + 1
        else:  # tensorproduct element
            p_h, p_v = p
            if quad_degree is None:
                quad_degree = 2*max(p_h, p_v) + 1

        if trial_space.extruded:
            # Create surface measures that treat the bottom and top boundaries similarly
            # to lateral boundaries. This way, integration using the ds and dS measures
            # occurs over both horizontal and vertical boundaries, and we can also use
            # "bottom" and "top" as surface identifiers, for example, ds("top").
            self.ds = CombinedSurfaceMeasure(self.mesh, quad_degree)
            self.dS = (
                firedrake.dS_v(domain=self.mesh, degree=quad_degree) +
                firedrake.dS_h(domain=self.mesh, degree=quad_degree)
            )
        else:
            self.ds = firedrake.ds(domain=self.mesh, degree=quad_degree)
            self.dS = firedrake.dS(domain=self.mesh, degree=quad_degree)

        self.dx = firedrake.dx(domain=self.mesh, degree=quad_degree)

        # General prefactor multiplying all terms in the equation
        # N.b. setting this to a firedrake constant (i.e. prefactor = Constant(1)) breaks
        # Drucker-Prager rheology test even though it is being multiplied by 1...
        self.prefactor = prefactor

        self.kwargs = kwargs

        # self._terms stores the actual instances of the BaseTerm-classes in self.terms
        self._terms = []
        for TermClass in self.terms:
            self._terms.append(TermClass(test_space, trial_space, self.dx, self.ds, self.dS, **kwargs))

    def mass_term(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
    ) -> firedrake.ufl.core.expr.Expr:
        """Typical mass term used in time discretisations.

        Arguments:
          test:  Firedrake test function
          trial: Firedrake trial function

        Returns:
          The UFL expression associated with the mass term of the equation.

        """
        return self.prefactor * firedrake.inner(test, trial) * self.dx

    def residual(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
        trial_lagged: Optional[firedrake.ufl_expr.Argument | firedrake.Function] = None,
        fields: Optional[dict[str, firedrake.Constant | firedrake.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> firedrake.ufl.core.expr.Expr:
        """Finite element residual.

        The final residual is calculated as a sum of all individual term residuals.

        Arguments:
          test:         Firedrake test function
          trial:        Firedrake trial function
          trial_lagged: Firedrake trial function from the previous time step
          fields:       Dictionary of physical fields from the simulation's state
          bcs:          Dictionary of identifier-value pairs specifying boundary conditions

        Returns:
          The UFL expression associated with all equation terms except the mass term.

        """
        if trial_lagged is None:
            trial_lagged = trial
        if fields is None:
            fields = {}
        if bcs is None:
            bcs = {}

        F = 0
        for term in self._terms:
            F += self.prefactor * term.residual(test, trial, trial_lagged, fields, bcs)

        return F


class BaseTerm(ABC):
    """Defines an equation's term using an UFL expression.

    The implemented expression describes the term's contribution to the residual in the
    finite element discretisation.

    Arguments:
      test_space:  Firedrake function space of the test function
      trial_space: Firedrake function space of the trial function
      dx:          UFL measure for the domain, boundaries excluded
      ds:          UFL measure for the domain's outer boundaries
      dS:
        UFL measure for the domain's inner boundaries when using a discontinuous
        function space

    """
    def __init__(
        self,
        test_space: firedrake.functionspaceimpl.WithGeometry,
        trial_space: firedrake.functionspaceimpl.WithGeometry,
        dx: firedrake.Measure,
        ds: firedrake.Measure,
        dS: firedrake.Measure,
        **kwargs,
    ):
        self.test_space = test_space
        self.trial_space = trial_space

        self.dx = dx
        self.ds = ds
        self.dS = dS

        self.mesh = test_space.mesh()
        self.dim = self.mesh.geometric_dimension()
        self.n = firedrake.FacetNormal(self.mesh)

        self.term_kwargs = kwargs

    @abstractmethod
    def residual(
        self,
        test: firedrake.ufl_expr.Argument,
        trial: firedrake.ufl_expr.Argument | firedrake.Function,
        trial_lagged: Optional[firedrake.ufl_expr.Argument | firedrake.Function] = None,
        fields: Optional[dict[str, firedrake.Constant | firedrake.Function]] = None,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
    ) -> firedrake.ufl.core.expr.Expr:
        """Residual associated with the equation's term.

        Arguments:
          test:         Firedrake test function
          trial:        Firedrake trial function
          trial_lagged: Firedrake trial function from the previous time step
          fields:       Dictionary of physical fields from the simulation's state
          bcs:          Dictionary of identifier-value pairs specifying boundary conditions

        Returns:
          A UFL expression for the term's contribution to the finite element residual.

        """
        pass
