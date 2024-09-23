"""Generates the UFL form for an equation consisting of individual terms.

This module contains a dataclass to define the structure of mathematical equations
within the G-ADOPT library. It provides a convenient way to generate the UFL form
required by Firedrake solvers.

"""

from dataclasses import KW_ONLY, InitVar, dataclass, field
from numbers import Number
from typing import Any, Callable, Optional

import firedrake as fd

from .approximations import Approximation
from .utility import CombinedSurfaceMeasure

__all__ = ["Equation"]


@dataclass
class Equation:
    """Generates the UFL form for the sum of terms constituting an equation.

    The generated UFL form corresponds to a sum of implemented term forms contributing
    to the equation's residual in the finite element discretisation.

    Args:
        test:
          Firedrake test function.
        trial_space:
          Firedrake function space of the trial function.
        residual_terms:
          List of equation terms contributing to the residual.
        mass_term:
          Callable returning the equation's mass term.
        terms_kwargs:
          Dictionary of fields and parameters used in the equation's weak form.
        approximation:
          G-ADOPT approximation for the system of equations considered.
        bcs:
          Dictionary of identifier-value pairs specifying weak boundary conditions.
        quad_degree:
          Integer specifying the quadrature degree. If omitted, it is set to `2p + 1`,
          where p is the polynomial degree of the trial space.
        rescale_factor:
          UFL expression used to rescale mass and residual terms.

    """

    test: fd.Argument | fd.ufl.indexed.Indexed
    trial_space: fd.functionspaceimpl.WithGeometry
    residual_terms: InitVar[Callable | list[Callable]]
    _: KW_ONLY
    mass_term: Optional[Callable] = None
    terms_kwargs: InitVar[dict[str, Any]] = {}
    approximation: Optional[Approximation] = None
    bcs: dict[int, dict[str, Any]] = field(default_factory=dict)
    quad_degree: InitVar[Optional[int]] = None
    rescale_factor: Number | fd.Constant | fd.Function = 1

    def __post_init__(
        self,
        residual_terms: Callable | list[Callable],
        terms_kwargs: dict[str, Any],
        quad_degree: Optional[int],
    ) -> None:
        if not isinstance(residual_terms, list):
            residual_terms = [residual_terms]
        self.residual_terms = residual_terms

        for key, value in terms_kwargs.items():
            setattr(self, key, value)

        if quad_degree is None:
            p = self.trial_space.ufl_element().degree()
            if not isinstance(p, int):  # Tensor-product element
                p = max(p)

            quad_degree = 2 * p + 1

        self.mesh = self.trial_space.mesh()
        self.n = fd.FacetNormal(self.mesh)

        measure_kwargs = {"domain": self.mesh, "degree": quad_degree}
        self.dx = fd.dx(**measure_kwargs)

        if self.trial_space.extruded:
            # Create surface measures that treat the bottom and top boundaries similarly
            # to lateral boundaries. This way, integration using the ds and dS measures
            # occurs over both horizontal and vertical boundaries, and we can also use
            # "bottom" and "top" as surface identifiers, for example, ds("top").
            self.ds = CombinedSurfaceMeasure(**measure_kwargs)
            self.dS = fd.dS_v(**measure_kwargs) + fd.dS_h(**measure_kwargs)
        else:
            self.ds = fd.ds(**measure_kwargs)
            self.dS = fd.dS(**measure_kwargs)

    def mass(
        self, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
    ) -> fd.Form:
        """Generates the UFL form corresponding to the mass term."""

        return self.rescale_factor * self.mass_term(self, trial)

    def residual(
        self, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
    ) -> fd.Form:
        """Generates the UFL form corresponding to the residual terms."""

        return self.rescale_factor * sum(
            [term(self, trial) for term in self.residual_terms]
        )


def cell_edge_integral_ratio(mesh: fd.MeshGeometry, p: int) -> int:
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2 for facets f,
    elements e, and polynomials u of degree p.

    See Equation (3.7), Table 3.1, and Appendix C from Hillewaert's thesis:
    https://www.researchgate.net/publication/260085826
    """
    cell_type = mesh.ufl_cell().cellname()
    if cell_type == "triangle":
        return (p + 1) * (p + 2) / 2.0
    elif cell_type == "quadrilateral" or cell_type == "interval * interval":
        return (p + 1) ** 2
    elif cell_type == "triangle * interval":
        return (p + 1) ** 2
    elif cell_type == "quadrilateral * interval":
        # if e is a wedge and f is a triangle: (p+1)**2
        # if e is a wedge and f is a quad: (p+1)*(p+2)/2
        # here we just return the largest of the the two (for p>=0)
        return (p + 1) ** 2
    elif cell_type == "tetrahedron":
        return (p + 1) * (p + 3) / 3
    else:
        raise NotImplementedError("Unknown cell type in mesh: {}".format(cell_type))


def interior_penalty_factor(eq: Equation, *, shift: int = 0) -> float:
    """Interior Penalty method
    For details on the choice of sigma, see
    https://www.researchgate.net/publication/260085826
    We use Equations (3.20) and (3.23). Instead of getting the maximum over two
    adjacent cells (+ and -), we just sum (i.e. 2 * avg) and have an extra 0.5 for for
    internal facets.
    """
    degree = eq.trial_space.ufl_element().degree()
    if not isinstance(degree, int):
        degree = max(degree)

    if degree == 0:  # probably only works for orthogonal quads and hexes
        sigma = 1.0
    else:
        # safety factor: 1.0 is theoretical minimum
        alpha = eq.interior_penalty if hasattr(eq, "interior_penalty") else 2.0
        num_facets = eq.mesh.ufl_cell().num_facets()
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma
