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


def normalise_intersect_measures(
    intersect_measures: Iterable | fd.Measure | fd.MeshGeometry | None,
) -> tuple[fd.Measure, ...]:
    """Turns the `intersect_measures` argument into the tuple UFL expects.

    A mesh is the convenient thing to write at the call site and a measure is
    what UFL wants, so both are accepted and a mesh becomes that mesh's cell
    measure -- the choice `GravitySolver.set_measures` makes for every
    cross-mesh integral it builds. A bare mesh or measure is wrapped, since
    coupling to a single other domain is the common case and a one-element
    tuple is easy to forget.

    Note that an intersected measure whose intersection turns out to be empty
    does not raise: it assembles to zero (see `demos/gravity/spikes/`), so a
    form built this way is worth checking against a known value once.
    """
    if intersect_measures is None:
        return ()

    if isinstance(intersect_measures, (fd.Measure, fd.MeshGeometry)):
        intersect_measures = (intersect_measures,)

    return tuple(
        fd.Measure("cell", domain=item) if isinstance(item, fd.MeshGeometry) else item
        for item in intersect_measures
    )


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
        intersect_measures:
          Other domains that must be admissible in this equation's integrals. Each
          entry is a mesh or a UFL measure; a mesh is turned into that mesh's cell
          measure. Omit it (the default) for a single-mesh equation, in which case
          the measures below are built exactly as they always were. Supply it when
          the equation couples a submesh to its parent, so that arguments living on
          the other mesh -- a potential on the parent, a displacement on the mantle
          submesh -- are accepted by terms integrating over this one. See
          `GravitySolver.set_measures` for the same idiom applied by hand.

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
    intersect_measures: InitVar[
        Iterable[fd.Measure | fd.MeshGeometry] | fd.Measure | fd.MeshGeometry | None
    ] = None

    def __post_init__(
        self,
        residual_terms: Callable | list[Callable],
        eq_attrs: dict[str, Any],
        quad_degree: int | None,
        intersect_measures: Iterable | fd.Measure | fd.MeshGeometry | None,
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

        self.intersect_measures = normalise_intersect_measures(intersect_measures)

        measure_kwargs = {"domain": self.mesh, "degree": quad_degree}
        # Only pass the keyword when there is something to pass, so that the
        # single-mesh path constructs the measures exactly as it always has.
        if self.intersect_measures:
            measure_kwargs["intersect_measures"] = self.intersect_measures

        self.dx = fd.dx(**measure_kwargs)

        if self.trial_space.extruded:
            if self.intersect_measures:
                # `CombinedSurfaceMeasure` builds its three measures itself and
                # takes only a domain and a degree. Cross-mesh coupling arises
                # from `Submesh`, which the extruded meshes do not use, so this
                # is a gap rather than a restriction; widen the helper if it ever
                # stops being one.
                raise NotImplementedError(
                    "`intersect_measures` is not supported on extruded meshes."
                )

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
        num_facets = eq.mesh.ufl_cell().num_facets
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma
