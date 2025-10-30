"""Generates the UFL form for an equation consisting of individual terms.

This module contains a dataclass to define the structure of mathematical equations
within the G-ADOPT library. It provides a convenient way to generate the UFL form
required by Firedrake solvers.

"""

from collections.abc import Iterable
from dataclasses import KW_ONLY, InitVar, dataclass, field
from numbers import Number
from typing import Any, Callable
from warnings import warn

import firedrake as fd

from .approximations import BaseApproximation
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
          Equation term or a list thereof contributing to the residual.
        mass_term:
          Callable returning the equation's mass term.
        eq_attrs:
          Dictionary of fields and parameters used in the equation's weak form.
        approximation:
          G-ADOPT approximation for the system of equations considered.
        bcs:
          Dictionary specifying weak boundary conditions (identifier, type, and value).
        quad_degree:
          Integer specifying the quadrature degree. If omitted, it is set to `2p + 1`,
          where p is the polynomial degree of the trial space.
        scaling_factor:
          A constant factor used to rescale mass and residual terms.

    """

    test: fd.Argument | fd.ufl.indexed.Indexed
    trial_space: fd.functionspaceimpl.WithGeometry
    residual_terms: InitVar[Callable | list[Callable]]
    _: KW_ONLY
    mass_term: Callable | None = None
    eq_attrs: InitVar[dict[str, Any]] = {}
    approximation: BaseApproximation | None = None
    bcs: dict[int, dict[str, Any]] = field(default_factory=dict)
    quad_degree: InitVar[int | None] = None
    scaling_factor: Number | fd.Constant = 1

    def __post_init__(
        self,
        residual_terms: Callable | list[Callable],
        eq_attrs: dict[str, Any],
        quad_degree: int | None,
    ) -> None:
        if not isinstance(residual_terms, Iterable):
            residual_terms = [residual_terms]
        self.residual_terms = residual_terms

        required_attrs = set.union(*(term.required_attrs for term in residual_terms))
        if missing_attrs := required_attrs - eq_attrs.keys():
            raise ValueError(
                "Provided equation attributes do not match the requirements of "
                f"requested equation terms.\nMissing attributes: {missing_attrs}."
            )

        optional_attrs = set.union(*(term.optional_attrs for term in residual_terms))
        if unused_attrs := eq_attrs.keys() - required_attrs.union(optional_attrs):
            warn(
                "Some unused equation attributes were provided.\nUnused attributes: "
                f"{unused_attrs}"
            )

        for key, value in eq_attrs.items():
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

        return self.scaling_factor * self.mass_term(self, trial)

    def residual(
        self, trial: fd.Argument | fd.ufl.indexed.Indexed | fd.Function
    ) -> fd.Form:
        """Generates the UFL form corresponding to the residual terms."""

        return self.scaling_factor * sum(
            term(self, trial) for term in self.residual_terms
        )

    def irksome_form(
        self,
        solution: fd.Function,
        dt_operator: Any
    ) -> fd.Form:
        """Generates the UFL form suitable for Irksome time integrators.

        This method creates a unified residual form that includes both the mass term
        (with Dt applied) and the spatial residual terms. This ensures that
        solution-dependent coefficients in the mass term are correctly evaluated
        at the current stage solution during implicit Runge-Kutta integration.

        The form is constructed as:
            F = mass_term(eq, Dt(solution)) - residual(solution)

        which represents the equation in residual form:
            ∂u/∂t + spatial_terms(u) = 0

        Args:
            solution: The solution function
            dt_operator: The Irksome Dt operator for time derivatives

        Returns:
            A UFL form suitable for Irksome's TimeStepper, representing the
            complete residual including time derivative and spatial terms.

        Note:
            The negative sign before residual is necessary because G-ADOPT's
            residual terms return -F (RHS convention), but Irksome expects
            everything on the LHS.
        """
        # Apply Dt to the solution for the mass term
        mass_form = self.mass(dt_operator(solution))

        # Get the spatial residual terms
        residual_form = self.residual(solution)

        # Combine: mass term minus residual (which is already negative)
        # This gives us: Dt(u) + spatial_terms = 0 (in weak form)
        return mass_form - residual_form


def cell_edge_integral_ratio(mesh: fd.MeshGeometry, p: int) -> int:
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2 for facets f,
    elements e, and polynomials u of degree p.

    See Equation (3.7), Table 3.1, and Appendix C from Hillewaert's thesis:
    https://www.researchgate.net/publication/260085826
    """
    match cell_type := mesh.ufl_cell().cellname():
        case "triangle":
            return (p + 1) * (p + 2) / 2.0
        case "quadrilateral" | "interval * interval":
            return (p + 1) ** 2
        case "triangle * interval":
            return (p + 1) ** 2
        case "quadrilateral * interval" | "hexahedron":
            # if e is a wedge and f is a triangle: (p+1)**2
            # if e is a wedge and f is a quad: (p+1)*(p+2)/2
            # here we just return the largest of the the two (for p>=0)
            return (p + 1) ** 2
        case "tetrahedron":
            return (p + 1) * (p + 3) / 3
        case _:
            raise NotImplementedError(f"Unknown cell type in mesh: {cell_type}")


def interior_penalty_factor(eq: Equation, *, shift: int = 0) -> float:
    """Interior Penalty method

    For details on the choice of sigma, see
    https://www.researchgate.net/publication/260085826

    We use Equations (3.20) and (3.23). Instead of getting the maximum over two adjacent
    cells (+ and -), we just sum (i.e. 2 * avg) and have an extra 0.5 for internal
    facets.
    """
    degree = eq.trial_space.ufl_element().degree()
    if not isinstance(degree, int):
        degree = max(degree)

    if degree == 0:  # probably only works for orthogonal quads and hexes
        sigma = 1.0
    else:
        # safety factor: 1.0 is theoretical minimum
        alpha = getattr(eq, "interior_penalty", 2.0)
        num_facets = eq.mesh.ufl_cell().num_facets()
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma
