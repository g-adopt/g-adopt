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
from ufl.indexed import Indexed

from .approximations import BaseApproximation, BaseGIAApproximation
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
          A constant factor used to rescale residual terms.

    """

    test: fd.Argument | Indexed
    trial_space: fd.functionspaceimpl.WithGeometry
    residual_terms: InitVar[Callable | list[Callable]]
    _: KW_ONLY
    eq_attrs: InitVar[dict[str, Any]] = {}
    approximation: BaseApproximation | BaseGIAApproximation | None = None
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

    def residual(self, trial: fd.Argument | Indexed | fd.Function) -> fd.Form:
        """Generates the UFL form corresponding to the residual terms."""
        return self.scaling_factor * sum(
            term(self, trial) for term in self.residual_terms
        )


def cell_edge_integral_ratio(mesh: fd.MeshGeometry, p: int) -> int:
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2 for facets f,
    elements e, and polynomials u of degree p.

    See Equation (3.7), Table 3.1, and Appendix C from Hillewaert's thesis:
    https://www.researchgate.net/publication/260085826
    """
    match cell_type := mesh.ufl_cell().cellname:
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
    r"""SIPG penalty coefficient using Hillewaert's sharp trace-inverse bound.

    Returns the dimensionless coefficient :math:`\sigma_0` such that the
    per-facet penalty assembled in the DG bilinear form is
    :math:`\sigma_0 \cdot \mathcal A(f) / \mathcal V(e)` (the geometric ratio
    is applied in the term body via ``FacetArea/CellVolume``). For coercivity
    of the SIPG bilinear form one needs (Hillewaert, Eq. 3.22 / Shahbazi 2005)

    .. math::
        \sigma_f \;>\; \mu \, \max_{e \ni f}
        \bigl( n_e \, C_{e,f}(q) \, \mathcal A(f) / \mathcal V(e) \bigr),

    where :math:`C_{e,f}(q)` is the sharp trace-inverse constant for
    polynomials of degree :math:`q` on the element/facet pair (Hillewaert
    Table 3.1) and :math:`n_e` is the number of facets of element ``e``.
    The implementation replaces ``max`` over the two adjacent elements by
    ``avg``, picking up an extra factor 2 in the form, which is exact on
    quasi-uniform meshes and a conservative estimate on stretched ones.

    The ``shift`` argument selects which polynomial degree is fed to
    :func:`cell_edge_integral_ratio`:

    * ``shift = -1``: use :math:`C(p-1)`. The Shahbazi /
      Epshteyn-Rivière coercivity proof bounds the consistency term
      :math:`\int_f [u]\,\langle\nabla v\rangle\,dS` by applying the trace
      inequality to :math:`\nabla v \in \mathcal P_{p-1}`, so the sharp
      constant is the one for degree :math:`p-1`. This is the
      theoretically tight choice.
    * ``shift = 0`` (signature default): use :math:`C(p)`, i.e. the
      trace constant for the solution space :math:`\mathcal P_p` itself.
      Stricter than required by a factor of ~:math:`((p+1)/p)^2`; the
      default here for historical reasons (matches the original gadopt
      momentum-equation penalty), and what the Richards solver opts
      into via ``eq.penalty_shift=0`` since its nonlinear ``K(h)``
      benefits from the extra coercivity margin.

    An overriding value may be set on the equation as ``eq.penalty_shift``.

    Args:
        eq: Equation instance; reads optional ``interior_penalty`` (safety
            factor, default 2.0) and ``penalty_shift`` (overrides
            ``shift``) attributes.
        shift: Default shift used when ``eq.penalty_shift`` is not set.

    References:
        Hillewaert, K. (2013). *Development of the Discontinuous Galerkin
        Method for high-resolution, large-scale CFD and acoustics in
        industrial geometries*. PhD thesis, Université catholique de
        Louvain. Chapter 3 and Appendix C.

        Shahbazi, K. (2005). An explicit expression for the penalty
        parameter of the interior penalty method. *Journal of
        Computational Physics*, 205(2), 401-407.

        Epshteyn, Y., & Rivière, B. (2007). Estimation of penalty
        parameters for symmetric interior penalty Galerkin methods.
        *Journal of Computational and Applied Mathematics*, 206(2),
        843-872.
    """
    degree = eq.trial_space.ufl_element().degree()
    if not isinstance(degree, int):
        degree = max(degree)

    shift = getattr(eq, "penalty_shift", shift)

    if degree == 0:  # probably only works for orthogonal quads and hexes
        sigma = 1.0
    else:
        # safety factor: 1.0 is the theoretical coercivity floor
        alpha = getattr(eq, "interior_penalty", 2.0)
        num_facets = eq.mesh.ufl_cell().num_facets
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma
