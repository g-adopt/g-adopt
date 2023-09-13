from abc import ABC, abstractmethod
import firedrake
from .utility import CombinedSurfaceMeasure


class BaseEquation(ABC):
    """An equation class that can produce the UFL for the registered terms."""

    """This should be a list of BaseTerm sub-classes that form the terms of the equation."""
    terms = []

    def __init__(self, test_space, trial_space, quad_degree=None, **kwargs):
        """
        :arg test_space: the test functionspace
        :arg trial_space: The trial functionspace
        test and trial space are only used to determine the the discretisation that's used (ufl_element)
        not what test and trial functions are actually used (these are provided seperately in residual())
        :arg quad_degree: quadrature degree, default is 2*p+1 where p is the polynomial degree of trial_space
        any further keywords arguments are passed on to the individual terms
        """
        self.test_space = test_space
        self.trial_space = trial_space
        self.mesh = trial_space.mesh()

        p = trial_space.ufl_element().degree()
        if isinstance(p, int):
            # isotropic element
            if quad_degree is None:
                quad_degree = 2*p + 1
        else:
            # tensorproduct element
            p_h, p_v = p
            if quad_degree is None:
                quad_degree = 2*max(p_h, p_v) + 1

        if trial_space.extruded:
            # extruded mesh: create surface measures that treat the bottom and top boundaries the same as lateral boundaries
            # so that ds or dS integrates over both horizontal and vertical boundaries
            # and we can also use "bottom" and "top" as surface ids, e.g. ds("top")
            self.ds = CombinedSurfaceMeasure(self.mesh, quad_degree)
            self.dS = firedrake.dS_v(domain=self.mesh, degree=quad_degree) + firedrake.dS_h(domain=self.mesh, degree=quad_degree)
        else:
            self.ds = firedrake.ds(domain=self.mesh, degree=quad_degree)
            self.dS = firedrake.dS(domain=self.mesh, degree=quad_degree)

        self.dx = firedrake.dx(domain=self.mesh, degree=quad_degree)

        # self._terms stores the actual instances of the BaseTerm-classes in self.terms
        self._terms = []
        for TermClass in self.terms:
            self._terms.append(TermClass(test_space, trial_space, self.dx, self.ds, self.dS, **kwargs))

    def mass_term(self, test, trial):
        r"""Return the UFL for the mass term \int test * trial * dx typically used in the time term."""
        return firedrake.inner(test, trial) * self.dx

    def residual(self, test, trial, trial_lagged=None, fields=None, bcs=None):
        """Return the UFL for all terms (except the time derivative)."""
        if trial_lagged is None:
            trial_lagged = trial
        if fields is None:
            fields = {}
        if bcs is None:
            bcs = {}
        F = 0
        for term in self._terms:
            F += term.residual(test, trial, trial_lagged, fields, bcs)

        return F


class BaseTerm(ABC):
    """A term in an equation, that can produce the UFL expression for its contribution to the FEM residual."""
    def __init__(self, test_space, trial_space, dx, ds, dS, **kwargs):
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
    def residual(self, test, trial, trial_lagged, fields):
        """Return the UFL for this term"""
        pass
