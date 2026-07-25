r"""Gravitational Poisson solver with Dirichlet-to-Neumann boundary treatment.

This module solves the gravitational Poisson equation

$$
  nabla^2 psi = -4 pi G rho,
$$

whose correct boundary condition lives at infinity: psi must match the decaying
free-space solution. On the finite domains G-ADOPT works with (annuli, spherical
shells and their extensions), that condition is imposed exactly (up to mode
truncation) through Dirichlet-to-Neumann (DtN) maps on circular/spherical
boundaries: each angular eigenmode of the boundary trace extends uniquely into
the source-free exterior (or interior core) as the decaying (regular) harmonic
solution, whose normal derivative is proportional to its own trace. The map
needs the trace of psi only - never the density - so the solver discovers the
mode content by itself. The trace coefficients are global scalar unknowns in
R spaces, defined by scalar constraint rows and fed back as boundary flux;
everything is a plain UFL form, so the system remains differentiable, adjoint-
capable, and embeddable in larger coupled residuals.

The DtN terms use the Robin-shifted formulation: the flux is split as

$$
  (d psi)/(d n) = -(alpha/R) psi + sum_k (alpha - lambda_k R)/R \, c_k e_k,
$$

which is algebraically identical for the treated modes but places a pointwise
Robin term in the psi-psi block, keeping it symmetric positive definite for the
Schur fieldsplit (the naive form leaves that block a singular pure-Neumann
Laplacian). Untreated modes above the truncation see Robin(alpha/R) - a
decaying far field - instead of a reflecting homogeneous Neumann condition.

Mass sheets sitting exactly on a DtN boundary (e.g. dynamic topography) enter
through the flux jump $[(d psi)/(d r)] = -4 pi G sigma$, which reduces to the
standard surface-source term $-4 pi G int sigma v \, ds$ with the DtN machinery
unchanged.

Users instantiate the `GravitySolver` class by providing a solution function,
the density, and boundary conditions keyed by boundary marker, then call the
`solve` method to update the solution.
"""

import abc
from collections import namedtuple
from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np
from firedrake import *
from firedrake.ufl_expr import extract_unique_domain

from .solver_options_manager import ConfigType, SolverConfigurationMixin
from .spherical_harmonics import real_spherical_harmonic
from .utility import CombinedSurfaceMeasure, DEBUG, INFO, ensure_constant, log_level

__all__ = ["CylindricalDtN", "SphericalDtN", "GravitySolver"]


direct_gravity_solver_parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Direct solve for the psi (potential) block.

MUMPS LU. Exact and robust, but the sparse factors of a 3-D volume operator
suffer fill-in that outgrows the matrix, so this is the default only in 2-D
(and for small 3-D problems selected explicitly). See `iterative_gravity_solver_parameters`
for the scalable alternative.
"""

iterative_gravity_solver_parameters = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 1000,
    "pc_type": "python",
    "pc_python_type": "gadopt.SPDAssembledPC",
    "assembled_pc_type": "gamg",
    "assembled_mg_levels_pc_type": "sor",
    "assembled_pc_gamg_threshold": 0.01,
    "assembled_pc_gamg_square_graph": 100,
    "assembled_pc_gamg_coarse_eq_limit": 1000,
    "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
}
"""Iterative solve for the psi (potential) block.

Conjugate gradients preconditioned by algebraic multigrid (GAMG). The Robin
shift makes the psi block strictly SPD, so `gadopt.SPDAssembledPC` sets the
PETSc MAT_SPD flag (CG eigen-estimates in the Chebyshev and GAMG setup) and no
near-nullspace is needed. These are the same parameters G-ADOPT's 3-D spherical
Stokes solver uses for its velocity block, which inverts a strictly harder SPD
operator on the same extruded cubed-sphere meshes at production scale. AMG has
O(N) memory with no fill-in, so it replaces MUMPS as the default psi solver in
3-D.
"""


DtNMode = namedtuple("DtNMode", ["key", "expr", "lam", "norm", "scale"])
"""One angular eigenmode of a DtN boundary.

Fields:
  key: label used in coefficient reporting (e.g. "cos3", "Y2,-1", "mean").
  expr: UFL expression for the boundary eigenfunction e_k.
  lam: DtN eigenvalue, i.e. (d psi)/(d r) = -lam psi for the exterior branch
    of this mode (lam >= 0; the interior branch enters with the same sign in
    the weak form).
  norm: analytic value of the boundary integral of e_k^2.
  scale: norm divided by the analytic boundary measure - the constraint-row
    scaling that makes the multiplier equal the trace coefficient of e_k.
"""


class BaseDtN(abc.ABC):
    """Base class for Dirichlet-to-Neumann boundary condition descriptors.

    A DtN descriptor carries the geometry-specific data of the map: the angular
    eigenfunctions of its boundary type, the per-mode DtN eigenvalues on each
    side (exterior map for a boundary enclosing the sources, interior map for a
    boundary enclosing a source-free core), and the analytic normalisation
    constants. Orientation and radius are not part of the descriptor: the
    solver measures both from the marked boundary itself.
    """

    #: Geometric dimension the descriptor applies to.
    dim: int

    @property
    @abc.abstractmethod
    def max_degree(self) -> int:
        """Highest angular degree treated; drives the quadrature default."""

    @abc.abstractmethod
    def n_multipliers(self, side: str) -> int:
        """Number of R-space multipliers on a boundary of the given side."""

    @abc.abstractmethod
    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        """Mode table on a boundary of radius R with coordinates X.

        Args:
          side: "exterior" (sources inside the boundary) or "interior"
            (source-free core inside the boundary).
          R: Boundary radius, as measured by the solver.
          X: Coordinate vector of the mesh.
        """


class CylindricalDtN(BaseDtN):
    r"""DtN map on a circular boundary of a 2-D (cylindrical) domain.

    Azimuthal modes $e_m in {cos(m phi), sin(m phi)}$ for $m = 1 dots M$ are
    treated exactly: the exterior harmonic branch $r^(-m)$ gives
    $(d psi)/(d r) + (m/R) psi = 0$ and the interior branch $r^(+m)$ gives
    $(d psi)/(d r) - (m/R) psi = 0$. Content above the truncation is
    $O((r_s/R)^(M+1))$ for sources at radius $r_s$ and sees the Robin shift (a
    decaying far field) instead.

    The 2-D monopole is exceptional: the exterior $m = 0$ solution is
    logarithmic, so the trace does not determine the flux. Under the
    zero-net-mass assumption (which the solver verifies) the Robin shift is
    exact for it; on an interior boundary the exact monopole condition is
    homogeneous Neumann, recovered by the $m = 0$ mean multiplier.

    Args:
      M: Mode truncation; modes 1..M are treated exactly on this boundary.
    """

    dim = 2

    def __init__(self, M: int):
        if M < 0:
            raise ValueError(f"Require M >= 0, got M={M}")
        self.M = M

    @property
    def max_degree(self) -> int:
        return max(self.M, 1)

    def n_multipliers(self, side: str) -> int:
        return 2 * self.M + (1 if side == "interior" else 0)

    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        phi = atan2(X[1], X[0])
        table = []
        if side == "interior":
            # The m = 0 mean multiplier undoing the Robin shift: the exact
            # interior monopole condition is homogeneous Neumann (lam = 0).
            table.append(DtNMode("mean", 1.0, 0.0, 2 * np.pi * R, 1.0))
        for m in range(1, self.M + 1):
            lam = m / R
            table.append(
                DtNMode(f"cos{m}", cos(m * phi), lam, np.pi * R, 0.5))
            table.append(
                DtNMode(f"sin{m}", sin(m * phi), lam, np.pi * R, 0.5))
        return table


class SphericalDtN(BaseDtN):
    r"""DtN map on a spherical boundary of a 3-D domain.

    Real orthonormal spherical harmonics $Y_(l m)$ for $l = 0 dots L$,
    $-l <= m <= l$ are treated exactly: the exterior branch $r^(-(l+1))$ gives
    $(d psi)/(d r) + ((l+1)/R) psi = 0$ and the interior branch $r^(+l)$ gives
    $(d psi)/(d r) - (l/R) psi = 0$. Unlike 2-D there is no monopole
    exception: the $l = 0$ exterior decays as $1/r$, so the trace-based map is
    complete and no mass restriction applies.

    Args:
      L: Degree truncation; degrees 0..L are treated exactly on this boundary.
        The multiplier count is $(L+1)^2$, so keep L modest; content above the
        truncation decays as $O((r_s/R)^(L+2))$ and sees the Robin shift.
    """

    dim = 3

    def __init__(self, L: int):
        if L < 0:
            raise ValueError(f"Require L >= 0, got L={L}")
        self.L = L

    @property
    def max_degree(self) -> int:
        return max(self.L, 1)

    def n_multipliers(self, side: str) -> int:
        return (self.L + 1) ** 2

    def modes(self, side: str, R: float, X) -> list[DtNMode]:
        table = []
        for l in range(self.L + 1):
            lam = (l + 1) / R if side == "exterior" else l / R
            for m in range(-l, l + 1):
                table.append(
                    DtNMode(f"Y{l},{m}", real_spherical_harmonic(l, m, X),
                            lam, R**2, 1.0 / (4.0 * np.pi)))
        return table


class GravitySolver(SolverConfigurationMixin):
    """Solver for the gravitational Poisson equation with DtN boundaries.

    Solves $nabla^2 psi = -4 pi G rho$ for the perturbation potential psi with
    the infinite-domain behaviour imposed through Dirichlet-to-Neumann maps on
    the boundaries specified in `bcs`. The solution function is updated in
    place by `solve`.

    The density may live on the solver mesh or on a `Submesh` of it (the
    extended-domain configuration, where psi extends beyond the region
    carrying the density); the cross-mesh coupling is set up automatically.

    Args:
      solution:
        Firedrake function for the potential, on a scalar function space
      rho:
        Density: a Firedrake function (on the solution mesh or a Submesh of
        it), a UFL expression, or a number
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      gravitational_constant:
        Value of G in the chosen non-dimensionalisation
      quad_degree:
        Integer denoting the quadrature degree applied on boundary integrals
        involving the DtN eigenfunctions; UFL's automatic estimate is
        unreliable for these, so it defaults to 2 * (max mode degree +
        element degree) and is verified by `check_boundary_quadrature`
      source_quad_degree:
        Optional quadrature degree for the volume source term; useful when
        rho is an analytic UFL expression (e.g. an angular mode restricted to
        a mesh-conforming shell by a conditional) whose exact integration
        removes the source-representation error from convergence studies
      solver_parameters:
        Either a complete dictionary of PETSc solver options, or the string
        "direct"/"iterative" to select a G-ADOPT default. Omitting it (None)
        picks the default by dimension: "direct" in 2-D, "iterative" in 3-D,
        mirroring the Stokes solver. The two presets differ only in how the psi
        (potential) block is inverted - MUMPS versus CG + algebraic multigrid;
        direct does not scale to large 3-D meshes. Passing a full dictionary
        means you own the description of the two Schur blocks that the class
        otherwise sets up.
      solver_parameters_extra:
        Dictionary of PETSc solver options used to update the default G-ADOPT
        options. The safe way to tweak the default (e.g. override only the psi
        block, via a `{"dtn": {"fieldsplit_0": ...}}` entry) while keeping the
        Schur structure.

    ### Valid keys for boundary conditions
    | Condition |  Type  |                    Description                     |
    | :-------- | :----- | :------------------------------------------------: |
    | psi       | Strong | Prescribed potential                               |
    | flux      | Weak   | Prescribed outward normal derivative (d psi)/(d n) |
    | dtn       | Weak   | DtN map (`CylindricalDtN` or `SphericalDtN`)       |
    | sigma     | Weak   | Surface mass density concentrated on the boundary  |

    A boundary with no entry keeps the natural (homogeneous Neumann)
    condition. The orientation of a DtN boundary (exterior map enclosing the
    sources vs interior map enclosing a source-free core) and its radius are
    measured from the marked boundary, not specified by the user. DtN maps
    are only valid on origin-centred coordinate circles/spheres; a boundary
    deviating from constant radius triggers a warning, but the solve
    proceeds with the mean radius and correspondingly degraded accuracy.
    """

    name = "Gravity"
    alpha = 1.0  # Robin shift; see the module docstring

    def __init__(
        self,
        solution: Function,
        rho,
        /,
        *,
        bcs: dict[int | str, dict[str, Any]],
        gravitational_constant: float | Constant = 1.0,
        quad_degree: int | None = None,
        source_quad_degree: int | None = None,
        solver_parameters: ConfigType | str | None = None,
        solver_parameters_extra: ConfigType | None = None,
    ) -> None:
        self.solution = solution
        self.rho = ensure_constant(rho)
        self.bcs = bcs
        self.G = ensure_constant(gravitational_constant)

        self.solution_space = solution.function_space()
        if self.solution_space.value_size != 1:
            raise ValueError("solution must live on a scalar function space")
        self.mesh = self.solution_space.mesh()
        self.X = SpatialCoordinate(self.mesh)

        rho_mesh = extract_unique_domain(self.rho)
        self.cross_mesh = rho_mesh is not None and rho_mesh is not self.mesh
        self.rho_mesh = rho_mesh if self.cross_mesh else self.mesh

        self.set_boundary_conditions()
        self.set_measures(quad_degree, source_quad_degree)
        self.set_boundary_geometry()
        self.set_function_spaces()
        self.set_form()
        self.check_boundary_quadrature(rtol=1e-4, action="warn")
        self.set_solver_options(solver_parameters, solver_parameters_extra)
        self.set_solver()

    def set_boundary_conditions(self) -> None:
        """Sorts the `bcs` dictionary into strong and weak conditions."""
        self.dtn_boundaries = []  # [(bc_id, descriptor)]
        self.sigma_bcs = []  # [(bc_id, expression)]
        self.flux_bcs = []  # [(bc_id, expression)]
        self.dirichlet_bcs = []  # [(bc_id, value)]

        for bc_id, bc in self.bcs.items():
            if "dtn" in bc and "psi" in bc:
                raise ValueError(
                    f"Boundary {bc_id}: 'dtn' and 'psi' are mutually exclusive.")
            for bc_type, val in bc.items():
                match bc_type:
                    case "psi":
                        self.dirichlet_bcs.append((bc_id, val))
                    case "flux":
                        self.flux_bcs.append((bc_id, val))
                    case "sigma":
                        self.sigma_bcs.append((bc_id, val))
                    case "dtn":
                        if not isinstance(val, BaseDtN):
                            raise ValueError(
                                f"Boundary {bc_id}: 'dtn' value must be a "
                                "CylindricalDtN or SphericalDtN instance.")
                        if val.dim != self.mesh.geometric_dimension:
                            raise ValueError(
                                f"Boundary {bc_id}: {type(val).__name__} does "
                                f"not apply to a {self.mesh.geometric_dimension}-D mesh.")
                        self.dtn_boundaries.append((bc_id, val))
                    case _:
                        raise ValueError(
                            f"Boundary {bc_id}: unknown condition '{bc_type}'.")

    def set_measures(
        self, quad_degree: int | None, source_quad_degree: int | None
    ) -> None:
        """Sets volume and surface measures, with explicit boundary quadrature."""
        if quad_degree is None:
            element_degree = self.solution_space.ufl_element().degree()
            if not isinstance(element_degree, int):
                # Tensor-product elements on extruded meshes report a
                # (horizontal, vertical) degree tuple.
                element_degree = max(element_degree)
            max_mode = max(
                (dtn.max_degree for _, dtn in self.dtn_boundaries), default=1)
            quad_degree = 2 * (max_mode + element_degree)
        self.quad_degree = quad_degree

        if self.mesh.extruded:
            self.ds = CombinedSurfaceMeasure(self.mesh, quad_degree)
        else:
            self.ds = ds(domain=self.mesh, degree=quad_degree)

        if self.cross_mesh:
            self.dx = Measure(
                "dx", domain=self.mesh,
                intersect_measures=(Measure("dx", domain=self.rho_mesh),))
            self.dx_rho = Measure(
                "dx", domain=self.rho_mesh,
                intersect_measures=(Measure("dx", domain=self.mesh),))
        else:
            self.dx = self.dx_rho = dx(domain=self.mesh)
        if source_quad_degree is not None:
            self.dx_rho = self.dx_rho(degree=source_quad_degree)

    def set_boundary_geometry(self) -> None:
        """Measures radius and orientation of every DtN boundary.

        The sign of the boundary integral of n . x decides the side: positive
        means the domain lies inside the boundary (exterior map), negative
        means the boundary encloses a source-free core (interior map).
        """
        n = FacetNormal(self.mesh)
        r = sqrt(dot(self.X, self.X))
        self.boundary_geometry = {}  # bc_id -> (side, R)

        for bc_id, _ in self.dtn_boundaries:
            dss = self.ds(bc_id)
            area = assemble(1 * dss)
            if area <= 0.0:
                raise ValueError(f"DtN boundary {bc_id} has zero measure.")
            R = assemble(r * dss) / area
            rms = np.sqrt(abs(assemble((r - R) ** 2 * dss)) / area)
            if rms > 1e-3 * R:
                warn(
                    f"DtN boundary {bc_id} deviates from a coordinate "
                    f"circle/sphere (rms radius deviation {rms:.2e} at radius "
                    f"{R:.4g}); the DtN map assumes a boundary of constant radius.")
            side = "exterior" if assemble(dot(n, self.X) * dss) > 0 else "interior"
            self.boundary_geometry[bc_id] = (side, R)

    def set_function_spaces(self) -> None:
        """Builds the internal mixed space: psi, cross-mesh dummy, multipliers.

        Vector-valued R-space unknowns are unsupported in Firedrake, so each
        trace coefficient is a separate scalar R field.
        """
        spaces = [self.solution_space]
        if self.cross_mesh:
            spaces.append(FunctionSpace(self.rho_mesh, "DG", 0))

        R_scalar = FunctionSpace(self.mesh, "R", 0)
        self.n_multipliers = sum(
            dtn.n_multipliers(self.boundary_geometry[bc_id][0])
            for bc_id, dtn in self.dtn_boundaries)
        if self.n_multipliers > 200:
            warn(
                f"{self.n_multipliers} DtN multipliers requested; the "
                "per-field bookkeeping of scalar R spaces makes large "
                "truncation degrees expensive. Consider a lower truncation "
                "combined with a buffer region.")
        self._multiplier_offset = len(spaces)
        spaces.extend([R_scalar] * self.n_multipliers)

        self.mixed_space = MixedFunctionSpace(spaces)
        self.mixed_solution = Function(self.mixed_space)

    def set_form(self) -> None:
        """Sets the weak form: Poisson, DtN maps, sheets, and flux conditions."""
        trials = split(self.mixed_solution)
        tests = TestFunctions(self.mixed_space)
        psi, v = trials[0], tests[0]
        i_R = self._multiplier_offset
        multipliers = list(zip(trials[i_R:], tests[i_R:]))
        self._multiplier_keys = []

        F = dot(grad(psi), grad(v)) * self.dx
        F -= 4 * np.pi * self.G * self.rho * v * self.dx_rho
        if self.cross_mesh:
            # The dummy field makes Firedrake set up the cross-mesh entity
            # maps; its equation is trivial.
            lam, mu_lam = trials[1], tests[1]
            F += lam * mu_lam * Measure("dx", domain=self.rho_mesh)

        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            dss = self.ds(bc_id)
            F += (self.alpha / R) * psi * v * dss

            for mode in dtn.modes(side, R, self.X):
                (c, mu), multipliers = multipliers[0], multipliers[1:]
                self._multiplier_keys.append((bc_id, mode.key))
                # Constraint row: mu is globally constant, so this is one
                # scalar equation enforcing c = (boundary integral of
                # psi e_k) / norm - the trace coefficient of e_k.
                F += (psi * mode.expr - mode.scale * c) * mu * dss
                # Modal correction to the shifted flux (same sign for
                # exterior and interior maps).
                F += (mode.lam - self.alpha / R) * c * mode.expr * v * dss

        for bc_id, sigma in self.sigma_bcs:
            F -= 4 * np.pi * self.G * sigma * v * self.ds(bc_id)

        for bc_id, flux in self.flux_bcs:
            F -= flux * v * self.ds(bc_id)

        self.F = F
        self.strong_bcs = [
            DirichletBC(self.mixed_space.sub(0), val, bc_id)
            for bc_id, val in self.dirichlet_bcs]

    def set_solver_options(
        self,
        solver_preset: ConfigType | str | None,
        solver_extras: ConfigType | None,
    ) -> None:
        """Sets PETSc solver options.

        The psi (potential) block is the only thing that varies between the two
        presets. Matrices with R-space blocks cannot be assembled
        monolithically, so with DtN multipliers present both presets share a
        full Schur complement eliminating onto the R fields: the psi block is
        assembled and inverted (directly by MUMPS, or iteratively by CG + AMG),
        and the tiny dense multiplier Schur complement is handled by GMRES.
        Without multipliers the psi solver is applied to the whole assembled
        system.

        The split itself is set up by `gadopt.DtNTwoBlockSchurPC`, which
        describes the two blocks to PETSc as index sets rather than as lists of
        field numbers; that is what keeps the solve working past the 128-field
        limit PETSc imposes on field enumeration, which one scalar R space per
        angular mode otherwise crosses at a spherical truncation of L = 7 on a
        two-boundary shell (L = 11 with a single DtN boundary). Its sub-solvers
        are configured under the `dtn` entry.

        A full solver-options dictionary is used verbatim. The strings "direct"
        and "iterative" select a G-ADOPT preset; None picks one by dimension
        (direct in 2-D, iterative in 3-D), mirroring the Stokes solver, so the
        scalable path is the default at production scale.
        """
        if isinstance(solver_preset, Mapping):
            if (self.n_multipliers > 0
                    and solver_preset.get("mat_type") in ("aij", "baij", "sbaij")):
                raise ValueError(
                    "Monolithic matrix assembly is unsupported with DtN "
                    "R-space multipliers. Provide a fieldsplit configuration "
                    "(see the G-ADOPT default) or adjust the default via "
                    "solver_parameters_extra instead.")
            self.add_to_solver_config(solver_preset)
            self.add_to_solver_config(solver_extras)
            self.register_update_callback(self.set_solver)
            return

        if solver_preset is not None and solver_preset not in ("direct", "iterative"):
            raise ValueError(
                f"solver_parameters must be a dictionary or one of 'direct', "
                f"'iterative', got {solver_preset!r}.")
        if solver_preset is None:
            solver_preset = (
                "direct" if self.mesh.topological_dimension == 2 else "iterative")

        if self.n_multipliers == 0:
            # No DtN multipliers: the psi solver acts on the whole assembled
            # system. GAMG wants the plain (unprefixed) options here, since
            # there is no fieldsplit_0 AssembledPC to hand them to. This also
            # means no SPDAssembledPC and hence no MAT_SPD flag - the eigen-
            # estimates fall back to GMRES; benign, and not worth the
            # AssembledPC indirection on a monolithic aij solve. Unlike the
            # multiplier path there is no outer Krylov cleaning up the psi
            # solve, so ksp_rtol here IS the final solution accuracy (the same
            # situation as the Stokes preonly-outer path): tighten it if a
            # no-multiplier 3-D config ever needs gradient-quality output.
            base = {"mat_type": "aij", "snes_type": "ksponly"}
            if solver_preset == "direct":
                self.add_to_solver_config({**base, **direct_gravity_solver_parameters})
            else:
                self.add_to_solver_config({
                    **base,
                    "ksp_type": "cg",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                    "pc_type": "gamg",
                    "mg_levels_pc_type": "sor",
                    "pc_gamg_threshold": 0.01,
                    "pc_gamg_square_graph": 100,
                    "pc_gamg_coarse_eq_limit": 1000,
                    "pc_gamg_mis_k_minimum_degree_ordering": True,
                })
                if INFO >= log_level:
                    self.add_to_solver_config({"ksp_converged_reason": None})
        else:
            if solver_preset == "direct":
                fieldsplit_0 = {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled": dict(direct_gravity_solver_parameters),
                }
            else:
                fieldsplit_0 = dict(iterative_gravity_solver_parameters)
            self.add_to_solver_config({
                "mat_type": "matfree",
                "snes_type": "ksponly",
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-11,
                "pc_type": "python",
                "pc_python_type": "gadopt.DtNTwoBlockSchurPC",
                "dtn": {
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0": fieldsplit_0,
                    "fieldsplit_1": {
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-6,
                        "pc_type": "none",
                    },
                },
            })
            if INFO >= log_level:
                self.add_to_solver_config({"ksp_converged_reason": None})
            if DEBUG >= log_level:
                self.add_to_solver_config({
                    "dtn": {
                        "fieldsplit_0": {"ksp_converged_reason": None},
                        "fieldsplit_1": {"ksp_converged_reason": None},
                    },
                })

        self.add_to_solver_config(solver_extras)
        self.warn_on_stale_fieldsplit_options()
        self.register_update_callback(self.set_solver)

    def warn_on_stale_fieldsplit_options(self) -> None:
        """Warns about fieldsplit options left at the top level of the preset.

        The two Schur blocks used to be described to PETSc by field number, so
        their sub-solvers sat directly under the solver options. They now live
        inside `gadopt.DtNTwoBlockSchurPC` under its own `dtn` prefix, and a
        `fieldsplit_0` entry passed at the top level of
        `solver_parameters_extra` silently stops matching anything rather than
        failing - the one way this change can bite an existing script.
        """
        if self.n_multipliers == 0:
            return
        stale = sorted(
            key for key in self.solver_parameters
            if key.startswith(("fieldsplit_", "pc_fieldsplit_")))
        if stale:
            warn(
                f"Solver options {stale} sit at the top level, where the DtN "
                "preset no longer reads them: the two Schur blocks are now "
                "described by index sets inside gadopt.DtNTwoBlockSchurPC and "
                "its sub-solvers live under the 'dtn' entry. Pass them as "
                "solver_parameters_extra={'dtn': {'fieldsplit_0': ...}} "
                "instead.")

    def set_solver(self) -> None:
        """Sets up the Firedrake variational problem and solver."""
        self.problem = NonlinearVariationalProblem(
            self.F, self.mixed_solution, bcs=self.strong_bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            options_prefix=self.name,
        )

    def check_net_mass(self) -> None:
        """Verifies the zero-net-mass assumption of 2-D exterior DtN maps.

        The 2-D exterior monopole is logarithmic, so its flux is set by the
        total enclosed mass (volumetric density plus boundary sheets) rather
        than the boundary trace. The Robin-shifted map is exact for zero net
        mass; the nonzero case would need the -2 G M / R flux datum, which is
        not implemented.
        """
        if self.mesh.geometric_dimension != 2:
            return
        if not any(self.boundary_geometry[bc_id][0] == "exterior"
                   for bc_id, _ in self.dtn_boundaries):
            return

        mass = assemble(self.rho * self.dx_rho)
        scale = assemble(abs(self.rho) * self.dx_rho)
        for bc_id, sigma in self.sigma_bcs:
            mass += assemble(sigma * self.ds(bc_id))
            scale += assemble(abs(sigma) * self.ds(bc_id))
        if abs(mass) > 1e-8 * max(scale, 1.0):
            raise NotImplementedError(
                f"Net mass {mass:.3e} is nonzero: the 2-D exterior DtN "
                "monopole requires the -2 G M / R flux datum, which is not "
                "implemented. Use a zero-mean density or a 3-D formulation. "
                "If the density should have zero net mass, this can also "
                "indicate monopole leakage from density boundaries that do "
                "not conform to cell edges - a genuine accuracy problem in "
                "itself; align the mesh with the density interfaces.")

    def check_boundary_quadrature(
        self, rtol: float = 1e-8, action: str = "raise"
    ) -> float:
        """Verifies that boundary quadrature and mesh resolve the DtN modes.

        Checks the assembled boundary integral of every eigenfunction squared
        against its analytic normalisation and returns the worst relative
        deviation, raising (or warning, with action="warn") beyond rtol.
        """
        worst, worst_key = 0.0, None
        for bc_id, dtn in self.dtn_boundaries:
            side, R = self.boundary_geometry[bc_id]
            dss = self.ds(bc_id)
            for mode in dtn.modes(side, R, self.X):
                val = assemble(mode.expr**2 * dss)
                dev = abs(val - mode.norm) / mode.norm
                if dev > worst:
                    worst, worst_key = dev, (bc_id, mode.key)
        if worst > rtol:
            message = (
                f"Boundary quadrature/mesh does not resolve DtN mode "
                f"{worst_key}: relative deviation {worst:.3e} > {rtol:.1e}. "
                "Refine the boundary mesh or raise quad_degree.")
            if action == "warn":
                warn(message)
            else:
                raise ValueError(message)
        return worst

    def solve(self) -> None:
        """Solves the system and updates the solution function in place."""
        self.check_net_mass()
        self.solver.solve()
        self.solution.assign(self.mixed_solution.subfunctions[0])

    def coefficients(self) -> dict[int | str, dict[str, float]]:
        """Solved trace coefficients of every DtN boundary, keyed by marker.

        For each DtN boundary, maps mode labels (e.g. "cos3", "Y2,-1") to the
        solved trace coefficients - the spectrum of psi on that boundary,
        i.e. the geoid coefficients when evaluated at the surface.
        """
        i_R = self._multiplier_offset
        out = {bc_id: {} for bc_id, _ in self.dtn_boundaries}
        for (bc_id, key), f in zip(
                self._multiplier_keys, self.mixed_solution.subfunctions[i_R:]):
            out[bc_id][key] = float(f)
        return out
