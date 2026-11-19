r"""The self-gravitating GIA solver: viscoelastic mechanics, potential and couplings.

`SelfGravitatingGIASolver` solves, in one Newton solve per time step, the
compressible viscoelastic momentum equation of the mantle, the material
history, the gravitational Poisson equation with its Dirichlet-to-Neumann
boundary treatment, and the global scalar fields of the couplings (fluid
core, rotation, centre-of-mass frame, sea level). A minimal script:

    parent = Mesh("earth_with_buffer.msh"); parent.cartesian = False
    mantle = Submesh(parent, 2, MANTLE)
    u = Function(VectorFunctionSpace(mantle, "CG", 2), name="displacement")
    psi = Function(FunctionSpace(parent, "CG", 2), name="potential")
    dt = Function(FunctionSpace(mantle, "R", 0)).assign(dt0)
    load = Function(FunctionSpace(mantle, "CG", 2))

    solver = SelfGravitatingGIASolver(
        u, psi, approximation, dt=dt, time_scheme="bdf2",
        internal_variable_family="Quadrature", quad_degree=4,
        gravity_bcs={OUTER: {"dtn": CylindricalDtN(8)},
                     INNER: {"dtn": CylindricalDtN(8)}},
        couplings=[FluidCore(boundary=RC, rho_core=rho_c),
                   Rotation(C=C, Omega_sq=1.566e-3), CentreOfMass(),
                   SurfaceLoad(boundary=RE, sigma=load, g_surface=1.0)],
        nullspace="rigid_rotation")
    for h, value in schedule:
        dt.assign(h); load.assign(value)
        solver.solve()                  # writes u and psi with assign
        output.write(u, psi)

## The sign convention

**`psi` is `GravitySolver`'s potential, which is minus the Newtonian one:**
`grad^2 psi = -4 pi G rho`, the perturbation gravity is `g_1 = +grad psi`,
and a positive point mass has `psi = +G M / r`. The convention is forced:
everything in the potential residual after the source line (the Robin
shift, the DtN constraint rows, the sheets, the 2-D monopole datum) is
`DtNGravityForm`, and both the monopole flux (`-2 G M / R`) and the sheet
term are anchored to it and are not sign-symmetric in `psi`. Three
consequences:

1. Both body-force terms carry a minus, `-c B_mu int rho_0 grad(psi) . w`
   and the same with `psi_rot`, so that the two off-diagonal blocks of the
   displacement and the potential carry the same constant.
2. `psi_rot` is the negated centrifugal perturbation, so that it shares the
   convention of `psi` and the two body forces read alike.
3. The geoid is `N = +psi / g_0` (`BaseGIAApproximation.geoid`).

## The row scalings

`theta_psi` and `theta_rot_i` scale whole residual rows. A row scaling
changes the conditioning of the Jacobian and nothing about the solution. It
makes the Jacobian symmetric, and it keeps one block from dominating the
single residual norm of SNES over the mixed vector. Both are computed from
the approximation's own numbers and must never be fitted to make a symmetry
test pass: a symmetry test is an equation for the row scaling, so a fitted
scaling absorbs a sign error in the block it scales.

    theta_psi   = c B_mu / Lambda
    theta_rot_i = s_i c B_mu Omega_sq,   s = (+1, +1, -1)

with `c` the `scaling_factor`, which multiplies the whole momentum residual
and so both body forces; a `theta_psi` without it would make the Jacobian
asymmetric by that factor. The rotation rows close `K_i m_i = s_i dI_i3`
with `K = (C - A, C - A, C)`: the `m_3` closure has a different constant
and a different sign from the polar-wander pair, and a 2-D annulus has the
`m_3` row only.

`Omega_sq` (the squared rotation rate in units of `g / L`, 1.566e-3 for the
Earth) is small, but the rotational feedback is not: `C - A` is itself
proportional to `Omega^2`, so `Omega^2` cancels from the closure.
Eliminating `u` leaves the secular Liouville relation

    m = (1 + k_L) dI_direct / [ (C - A) (1 - k_T / k_s) ]

in which `k_T / k_s` is of order one. On the Spada benchmark the rotational
body force changes `|m|` by 32 percent at `t = 0`. A tolerance or a `Real`
block scaling must not be chosen on the assumption that rotation is a small
correction. The same relation ties `C - A` to `k_s`: the closure that the
solver solves, `[(C - A) - Q k_T(t)] m = dI_direct` with
`Q = a^5 Omega^2 / (3 G)` (MacCullagh), is the secular relation only when
`C - A = Q k_s`. `Rotation` takes either constant, computes the other and
refuses a pair that disagrees.

## The parts

The solver composes three parts and owns nothing they own:

- the **mechanics**, `InternalVariableMechanics` with a `DGHistory` or a
  `QuadratureHistory` (`gadopt.gia_history`) and one `BDF2History`;
- the **potential**, `GravityPotential` (`gadopt.gravity_potential`): the DtN
  form, the residual row, the 2-D monopole datum, the low-rank operator;
- the **couplings** (`gadopt.gia_couplings`), run in three ordered stages.

What it owns is the mixed space, the residual as the sum of the parts, the
nullspace, the solver options and how one step is solved and recorded. The
residual does not depend on how the step is solved; the step is one of
three variants (`_StockStep`, `_LowRankStep`, `_QuadratureStep`), each of
which puts one solve block on the tape.

## The mixed space

    [ u, (M, DG family only), psi, DtN multipliers (multiplier path),
      core pressure, m1 m2 m3 (or m3), centre-of-mass multipliers, Shift ]

The `Real` fields are contiguous and last, which `DtNTwoBlockSchurPC`
requires, and their order is fixed by the couplings' `order`, not by the
order of the `couplings` list. Field 1 is the history on the DG family
(`CondensedBlockPC` reads it there), and the potential follows the
displacement on the Quadrature family (`ReducedGIABlockPC`).

## The contract of `u`, `psi` and the history

The solver reads `u` and `psi` once, at construction, as the initial state.
After every `solve` it writes the solved displacement and potential into
them with recorded `assign`s, after the recorded kernel projection, so the
recorded and the replayed values agree. To change the state between steps,
use `restore`. `solver.internal_variables` is the accepted history level of
the time integrator, the one history Function of the run.

## State

`state()` returns the named Functions that a restart needs, `restore` puts
them back, `reset` returns to the undeformed state. `gadopt.gia_state`
writes them to a `CheckpointFile`, reads them back, and converts the files
of the earlier solver.
"""

import itertools
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from numbers import Number
from warnings import warn

import numpy as np
import ufl
from firedrake import (DirichletBC, Function, FunctionSpace,
                       MixedFunctionSpace, MixedVectorSpaceBasis,
                       NonlinearVariationalProblem, NonlinearVariationalSolver,
                       SpatialCoordinate, TestFunctions, VectorSpaceBasis,
                       action, adjoint, as_vector, assemble, derivative, dot,
                       grad, split, Constant)
from mpi4py import MPI
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from ufl.algorithms import expand_derivatives, extract_coefficients
from ufl.domain import extract_domains

from .gia_couplings import CouplingContext, ordered, scalar_value
from .gia_presets import (NULL_COUPLING_ROW_SCALE,
                          selfgrav_dtn_iterative_solver_parameters,
                          selfgrav_dtn_lowrank_direct_solver_parameters,
                          selfgrav_dtn_schur_lowrank_solver_parameters,
                          selfgrav_dtn_schur_solver_parameters)
from .gia_history import history_formulation
from .gravity_potential import GravityPotential
from .momentum_equation import rotational_potential_term, self_gravity_term
from .nullspaces import check_kernel_basis, project_out_kernel
from .preconditioners import near_nullspace_basis
from .solver_options_manager import SolverConfigurationMixin
from .stokes_integrators import (InternalVariableMechanics, StokesSolverBase,
                                 _basis_provider,
                                 _displacement_basis,
                                 newton_stokes_solver_parameters)
from .utility import derivative_solve_keywords, is_cartesian, log, upward_normal

__all__ = ["SelfGravitatingGIASolver"]


class _StockStep:
    """One step as Firedrake's own variational solve (multiplier DtN, DG history).

    The residual is a plain UFL form, so the stock solve block on the tape
    is exact. The derivative solves get the forward options without the
    Newton loop (`SelfGravitatingGIASolver.adjoint_solve_keywords`).
    """

    def __init__(self, owner):
        self.owner = owner
        self.problem = NonlinearVariationalProblem(
            owner.F, owner.solution, bcs=owner.strong_bcs)
        self.build()

    def build(self):
        """Build the variational solver from the owner's present options."""
        owner = self.owner
        self.solver = NonlinearVariationalSolver(
            self.problem, solver_parameters=owner.solver_parameters,
            nullspace=owner.nullspace,
            transpose_nullspace=owner.transpose_nullspace,
            appctx=owner.appctx, options_prefix=owner.name)

    def solved_history(self):
        """The history of the solved state: the DG field of the mixed solution."""
        return self.owner.solution.subfunctions[self.owner.field_index["history"]]

    def solve(self):
        """Solve, giving the derivative keywords to Firedrake on a taped solve."""
        self.owner._refresh_operator_version()
        keywords = self.owner.adjoint_solve_keywords()
        solve_keywords = (derivative_solve_keywords(self.solver, keywords["adj_kwargs"])
                          if keywords else {})
        self.solver.solve(**solve_keywords)


class _LowRankStep(_StockStep):
    """One step with the low-rank DtN update on the Jacobian and the residual.

    The update `theta_psi B` acts through two PETSc callbacks, which the tape
    does not see. After an annotated solve the stock solve block is adopted
    (`gadopt.dtn_coupled_adjoint.adopt_coupled_lowrank_block`) so that its
    replay, adjoint and tangent carry `B` and the derivative of `theta_psi`.
    """

    def build(self):
        from .dtn_coupled_adjoint import LowRankVariationalSolver
        owner = self.owner
        self.solver = LowRankVariationalSolver(
            self.problem, solver_parameters=owner.solver_parameters,
            nullspace=owner.nullspace,
            transpose_nullspace=owner.transpose_nullspace,
            appctx=owner.appctx, options_prefix=owner.name,
            post_function_callback=owner.augment_residual,
            post_jacobian_callback=owner.augment_jacobian)
        self.solver.gia_solver = owner

    def solve(self):
        self.owner._refresh_operator_version()
        adopt = annotate_tape()
        if adopt:
            tape = get_working_tape()
            n0 = len(tape.get_blocks())
        # `LowRankVariationalSolver.solve` supplies its own derivative keywords.
        self.solver.solve()
        if adopt:
            from .dtn_coupled_adjoint import adopt_coupled_lowrank_block
            adopt_coupled_lowrank_block(self.owner, tape, n0)


class _QuadratureStep:
    """One step with the history eliminated at quadrature points.

    `DirectConstitutiveBackend` owns the nonlinear solve with the local
    Newton callbacks, the kernel projection inside its map and the tape
    block of the whole step. It is built once; an options change rebuilds
    its variational solver only, so the residual and the local solve keep
    one set of private coefficients.
    """

    def __init__(self, owner):
        from .constitutive_integrators import DirectConstitutiveBackend
        self.owner = owner
        self.backend = DirectConstitutiveBackend(owner, owner.history.material)

    @property
    def solver(self):
        return self.backend.solver

    @solver.setter
    def solver(self, value):
        # `release_taping_solver` puts a stand-in here after it destroyed
        # the taping solver; the backend owns the solver of this step.
        self.backend.solver = value

    def build(self):
        self.backend.build_solver()

    def solved_history(self):
        """The history of the solved state: the backend's updated point values."""
        return self.backend.buffers[0]

    def solve(self):
        self.owner._refresh_operator_version()
        self.backend.solve()


class SelfGravitatingGIASolver(SolverConfigurationMixin):
    r"""Viscoelastic GIA coupled in one solve to self-gravitation and the couplings.

    The residual, with every term on the left-hand side and `c` the
    `scaling_factor`:

        F_u   = c [ mechanics - B_mu int rho_0 grad(psi + psi_rot) . w dx ]
        F_M   = -c [ history equation ]                         (DG family)
        F_psi = theta_psi [ int grad(psi) . grad(v) dx_g
                            - Lambda int rho_0 u . grad(v) dx_m
                            + DtN boundary rows ]
        + the rows of each coupling

    with `theta_psi = c B_mu / Lambda` (`B_mu` floored to one at zero).

    Args:
      u: the displacement, a vector Function on the mechanics mesh (usually
        a `Submesh` of the potential's mesh). Read once as the initial
        state; overwritten with the solution after every `solve`.
      psi: the potential, a scalar Function on the parent mesh. Same rules.
      approximation: a `BaseGIAApproximation` with an internal-variable
        rheology, `B_mu`, `density`, `g` and `self_gravity_number`.
      dt: the time step; a `Real` Function supports taped changes.
      time_scheme: `"backward_euler"` or `"bdf2"` (variable step, backward
        Euler first step).
      internal_variable_family: `"DG"` (the history is an unknown, eliminated
        cell by cell in the preconditioner) or `"Quadrature"` (eliminated at
        quadrature points by a local Newton solve; natural displacement
        boundaries only).
      internal_variable_degree: the DG degree, default 1. The Quadrature
        history takes `quad_degree` and refuses another value here.
      quad_degree: the volume quadrature degree and the degree of the
        Quadrature history. `None` gives 6 on the DG family and 4 on the
        Quadrature family, where a degree-6 rule stores 23 values per
        tetrahedron and component.
      dtn_quad_degree: the boundary quadrature degree of the DtN rows and of
        every coupling facet integral; `None` takes the DtN form's
        calibrated default.
      dtn_alpha: the Robin shift of the DtN form, `None` for its default.
      gravity_bcs: the potential's conditions (`"dtn"`, `"psi"`, `"flux"`).
      bcs: the displacement conditions (`"u"`, `"un"`, `"normal_stress"`,
        ...), as for the other G-ADOPT momentum solvers. A Function in a
        weak condition or a `normal_stress` must be a Function of the
        mechanics mesh; one of the parent is refused
        (`_refuse_parent_coefficients_on_mechanics_facets`).
      couplings: a list of `gadopt.gia_couplings.Coupling` objects.
      dtn_representation: `"lowrank"` (default; the DtN modes are a rank-n
        update and the space does not grow with the truncation) or
        `"multiplier"` (one `Real` unknown per mode).
      nullspace: `None`, `"rigid_rotation"`, or a `VectorSpaceBasis` on the
        displacement space. Each mode is checked at construction
        (`_check_kernel`) and the measured numbers are logged. The same basis
        is the transpose nullspace.
      nullspace_rtol: the largest change `L` of a density moment of degree
        up to 4 that a declared mode may cause (`_check_kernel`). The
        default 1e-2 refuses a lateral variation `1 + a x` with `a = 0.01`
        and warns from `a = 0.001`.
      nullspace_jacobian_rtol: the largest `|J e| / |J s|` of a declared
        mode against a shear of the same size; it refuses only gross errors
        such as `un` on a surface that is not a sphere about the origin (see
        `_check_kernel`). A mode that does not meet a strong displacement
        condition is refused separately.
      check_nullspace: `True` (default) runs the density and Jacobian
        measures of `_check_kernel` on every declared mode at construction;
        `False` skips them (a restart of a configuration that was checked
        before). The strong-condition check runs in both cases, as it costs
        one copy of the mode per condition.
      near_nullspace: the modes GAMG is given on the condensed displacement
        operator: `"incompressible"` (default), `"rigid"` or `"none"`.
      scaling_factor: `c`, a factor of the whole momentum residual. Like
        `approximation.B_mu` and `approximation.self_gravity_number`, which
        scale whole rows too, it is a number, a `Constant` or a Function of
        the potential's (parent) mesh; another mesh is refused. A
        `scaling_factor` that is a Function cannot be used together with a
        weak displacement condition or a `normal_stress` in `bcs`
        (`_refuse_parent_coefficients_on_mechanics_facets`).
      solver_parameters: `None` (direct in 2-D, iterative in 3-D),
        `"direct"`, `"iterative"`, or a complete dictionary.
      preset_options: arguments of
        `gadopt.gia_presets.selfgrav_dtn_iterative_solver_parameters` for the
        iterative preset, for example `{"outer_rtol": 1e-8}`. The solver
        supplies `condensed`, `dtn_representation` and `n_real`.
      solver_parameters_extra: options added last.
      derivative_max_it: the iteration cap of the outer Krylov solve of the
        adjoint and tangent solves. `None` keeps the forward cap. A derivative
        solve that reaches its cap raises, so a long adjoint run needs a cap
        that the adjoint sweep stays below.
    """

    name = "SelfGravitatingGIA"

    def __init__(
        self,
        u: Function,
        psi: Function,
        approximation,
        /,
        *,
        dt,
        gravity_bcs: dict,
        time_scheme: str = "backward_euler",
        internal_variable_family: str = "DG",
        internal_variable_degree: int | None = None,
        quad_degree: int | None = None,
        dtn_quad_degree: int | None = None,
        dtn_alpha=None,
        bcs: dict | None = None,
        couplings=(),
        dtn_representation: str = "lowrank",
        nullspace=None,
        nullspace_rtol: float = 1e-2,
        nullspace_jacobian_rtol: float = 0.5,
        check_nullspace: bool = True,
        near_nullspace: str = "incompressible",
        scaling_factor=1,
        solver_parameters=None,
        preset_options: dict | None = None,
        solver_parameters_extra: dict | None = None,
        derivative_max_it: int | None = None,
    ) -> None:
        if time_scheme not in ("backward_euler", "bdf2"):
            raise ValueError("time_scheme must be 'backward_euler' or 'bdf2'.")
        if internal_variable_family not in ("DG", "Quadrature"):
            raise ValueError(
                "internal_variable_family must be 'DG' or 'Quadrature'.")
        if near_nullspace not in ("incompressible", "rigid", "none"):
            raise ValueError(
                "near_nullspace must be 'incompressible', 'rigid' or 'none'.")
        self.family = internal_variable_family
        if quad_degree is None:
            quad_degree = 6 if self.family == "DG" else 4
        if self.family == "Quadrature":
            if internal_variable_degree not in (None, quad_degree):
                raise ValueError(
                    "The Quadrature history is stored at the volume quadrature "
                    f"points, degree quad_degree={quad_degree}; got "
                    f"internal_variable_degree={internal_variable_degree}.")
            internal_variable_degree = quad_degree
        elif internal_variable_degree is None:
            internal_variable_degree = 1
        self.u = u
        self.psi = psi
        self.approximation = approximation
        self.dt = dt
        self.time_scheme = time_scheme
        self.quad_degree = quad_degree
        self.scaling_factor = scaling_factor
        self.bcs = dict(bcs or {})
        self.near_nullspace_modes = near_nullspace
        self.derivative_max_it = derivative_max_it
        self.nullspace_rtol = nullspace_rtol
        self.nullspace_jacobian_rtol = nullspace_jacobian_rtol
        self.check_nullspace = check_nullspace
        self.dtn_representation = dtn_representation

        self.mesh = u.function_space().mesh()
        self.potential_mesh = psi.function_space().mesh()
        if not hasattr(self.potential_mesh, "cartesian"):
            raise ValueError(
                "The potential's mesh has no `cartesian` attribute; set "
                "`mesh.cartesian = False` for an annulus or a shell.")
        if not hasattr(self.mesh, "cartesian"):
            # `Submesh` does not inherit it, and G-ADOPT reads it on every mesh.
            self.mesh.cartesian = self.potential_mesh.cartesian
        if approximation.self_gravity_number is None:
            raise ValueError(
                "The approximation carries no `self_gravity_number`; the solver "
                "needs Lambda = 4 pi G rho_bar L / g_bar.")
        self.Lambda = approximation.self_gravity_number
        displacement_conditions = any(
            {"u", "un", "ux", "uy", "uz"} & set(bc) for bc in self.bcs.values())
        if displacement_conditions and not getattr(
                approximation, "history_supports_displacement_conditions", True):
            raise NotImplementedError(
                f"{type(approximation).__name__} has a history equation with no "
                "boundary part, so it supports natural displacement boundaries "
                "only; replace the 'u' or 'un' condition, for example by a "
                "FluidCore.")
        if self.family == "Quadrature":
            if displacement_conditions:
                raise NotImplementedError(
                    "The Quadrature history supports natural displacement "
                    "boundaries only; use the DG family, or a FluidCore.")
            if any("psi" in bc for bc in gravity_bcs.values()):
                raise NotImplementedError(
                    "The Quadrature history does not support a strong 'psi' "
                    "condition.")
        self.couplings = ordered(couplings)

        self.gravity = GravityPotential(
            psi.function_space(), self.mesh, gravity_bcs, Lambda=self.Lambda,
            quad_degree=dtn_quad_degree, alpha=dtn_alpha,
            representation=dtn_representation)
        # This solver refuses a DtN boundary that is not a complete circle or
        # sphere and an empty cross-mesh measure (`check_geometry`). The
        # checks assemble numbers that nothing on the tape reads.
        with stop_annotating():
            self.gravity.check_geometry()
        self.form = self.gravity.form
        if self.form.sigma_bcs:
            raise ValueError(
                "gravity_bcs carries a mass sheet ('sigma' or 'interior_sigma'). "
                "This solver takes a prescribed load once, as "
                "SurfaceLoad(boundary=..., sigma=..., g_surface=...) in "
                "couplings, which gives it to the momentum equation, the "
                "potential, the rotation rows and the centre-of-mass rows.")

        order = 2 if time_scheme == "bdf2" else 1
        self.history = history_formulation(
            self.family, self, internal_variable_degree, order)
        self.cross_mesh = self.mesh is not self.potential_mesh
        self._check_row_scaling_meshes()
        self.set_space()
        self.set_boundary_conditions()
        self.set_couplings()
        self.set_equations()
        self.set_form()
        self.set_nullspace(nullspace)
        self.set_lowrank_operator()
        self.set_solver_options(solver_parameters, preset_options,
                                solver_parameters_extra)
        self.set_solver()

    # -- Construction -----------------------------------------------------------

    def set_space(self) -> None:
        """Build the mixed space and the solution, and name the fields.

        The solution takes `u` and `psi` as its initial values, once.
        """
        dim = self.potential_mesh.geometric_dimension
        R = FunctionSpace(self.potential_mesh, "R", 0)
        spaces = [self.u.function_space()]
        names = ["displacement"]
        if self.family == "DG":
            spaces.append(self.history.space)
            names.append("history")
        spaces.append(self.psi.function_space())
        names.append("potential")
        i_real = len(spaces)
        n_mult = self.gravity.n_multipliers
        names += [f"dtn_multiplier_{k}" for k in range(n_mult)]
        for coupling in self.couplings:
            names += list(coupling.real_field_names(dim))
        spaces += [R] * (len(names) - len(spaces))
        self.field_names = tuple(names)
        self.field_index = {name: i for i, name in enumerate(names)}
        self.solution = Function(MixedFunctionSpace(spaces), name="gia_solution")
        # The space as the solution reports it: every argument of the
        # residual is built on this one object, because two arguments of one
        # number on spaces that compare unequal make the form invalid.
        self.solution_space = self.solution.function_space()
        self.solution.subfunctions[0].assign(self.u)
        self.solution.subfunctions[self.field_index["potential"]].assign(self.psi)
        self.solution_split = split(self.solution)
        self.tests = TestFunctions(self.solution_space)
        self.k = upward_normal(self.mesh)

        #: The index of the potential, of the DtN multiplier fields and of
        #: every `Real` field, in space order. `DtNTwoBlockSchurPC` needs the
        #: `Real` fields contiguous and last, which this order is.
        self.potential_index = self.field_index["potential"]
        self.multiplier_indices = tuple(range(i_real, i_real + n_mult))
        self.real_indices = tuple(range(i_real, len(names)))
        #: Whether the displacement lives on a submesh of the potential's mesh.
        self.cross_mesh = self.mesh is not self.potential_mesh

    def set_boundary_conditions(self) -> None:
        """Sort the displacement conditions into strong and weak ones."""
        self.strong_bcs = []
        self.weak_bcs = {}
        V = self.solution_space.sub(0)
        bc_map = {"u": V}
        if is_cartesian(self.mesh):
            bc_map["ux"], bc_map["uy"] = V.sub(0), V.sub(1)
            if self.mesh.geometric_dimension == 3:
                bc_map["uz"] = V.sub(2)
        for bc_id, bc in self.bcs.items():
            weak = defaultdict(float)
            for bc_type, value in bc.items():
                if bc_type in ("u", "ux", "uy", "uz"):
                    self.strong_bcs.append(DirichletBC(bc_map[bc_type], value, bc_id))
                elif bc_type == "free_surface":
                    raise ValueError(
                        "A free surface is not a condition of this solver; the "
                        "surface load is a SurfaceLoad or a SeaLevel coupling.")
                else:
                    weak[bc_type] += value
            self.weak_bcs[bc_id] = weak
        #: The strong displacement conditions, which a nullspace mode must meet.
        self.displacement_bcs = list(self.strong_bcs)
        self.has_strong_displacement_bc = bool(self.strong_bcs)

    @property
    def row_scale_B_mu(self):
        """`B_mu`, floored to one where it is exactly zero.

        `theta_psi` and the rotation and frame rows scale whole rows by
        `B_mu`; at `B_mu = 0` that would delete those rows instead of
        decoupling them (`gadopt.gia_presets.NULL_COUPLING_ROW_SCALE`).
        """
        B_mu = self.approximation.B_mu
        if scalar_value(B_mu) == 0.0:
            return NULL_COUPLING_ROW_SCALE
        return B_mu

    @property
    def _row_scale_B_mu(self):
        # The name the quadrature backend reads.
        return self.row_scale_B_mu

    @property
    def theta_psi(self):
        """`c B_mu / Lambda`, the scaling of the potential rows, UFL."""
        return self.scaling_factor * self.row_scale_B_mu / self.Lambda

    @property
    def theta_psi_value(self) -> float:
        """`theta_psi` as a number, from its factors."""
        return (scalar_value(self.scaling_factor) * scalar_value(self.row_scale_B_mu)
                / scalar_value(self.Lambda))

    def set_couplings(self) -> None:
        """Bind the couplings, check them, and run the potential and sheet stages."""
        self.coupling_context = ctx = CouplingContext(
            solution=self.solution,
            fields=dict(zip(self.field_names, self.solution_split)),
            tests=dict(zip(self.field_names, self.tests)),
            approximation=self.approximation,
            mesh=self.mesh,
            potential_mesh=self.potential_mesh,
            dx_m=self.gravity.dx_m,
            boundary_quad_degree=self.form.quad_degree,
            mechanics_quad_degree=self.quad_degree,
            scaling_factor=self.scaling_factor,
            row_scale_B_mu=self.row_scale_B_mu,
            Lambda=self.Lambda,
            four_pi_G=self.gravity.four_pi_G,
            bcs=self.bcs,
            field_names=self.field_names)
        for coupling in self.couplings:
            coupling.bind(self, ctx)
        # The checks measure the mesh and the couplings; nothing on the tape
        # reads them, so they are not recorded.
        with stop_annotating():
            for coupling in self.couplings:
                coupling.check(self.couplings)
        # Stage 1, then stage 2, over every coupling: the order of the list
        # does not matter, because no sheet is built before every potential.
        for coupling in self.couplings:
            coupling.potentials(ctx)
        for coupling in self.couplings:
            ctx.sheets.extend(coupling.sheets(ctx))
        for bc_id, conditions in self.bcs.items():
            if "normal_stress" in conditions:
                warn(f"Boundary {bc_id} carries a normal_stress in bcs. It loads "
                     "the mechanics only: the potential, the rotation and the "
                     "centre-of-mass rows do not see its mass. For a surface "
                     "load give SurfaceLoad(boundary=..., sigma=..., "
                     "g_surface=...) in couplings instead.")

    @property
    def shear_coefficient(self):
        """The penalty coefficient of the displacement block, from the history family."""
        return self.history.shear_coefficient(self.approximation, self.dt)

    def set_equations(self) -> None:
        """The momentum equation with the body forces, and the history equation."""
        self.equations = []
        ctx = self.coupling_context
        extra_terms = [self_gravity_term]
        extra_attrs = {"psi": ctx.fields["potential"], "mu": self.shear_coefficient}
        if ctx.rotational_potential is not None:
            extra_terms.append(rotational_potential_term)
            extra_attrs["psi_rot"] = ctx.rotational_potential(self.mesh)
        self.mechanics = InternalVariableMechanics(self.history)
        self.mechanics.set_equations(
            self, extra_terms=extra_terms, extra_attrs=extra_attrs,
            intersect_measures=self.potential_mesh if ctx.cross_mesh else None)

    def set_form(self) -> None:
        """The residual: mechanics, the potential row, then every coupling's rows."""
        ctx = self.coupling_context
        unknowns = [self.solution_split[0]]
        if self.family == "DG":
            unknowns.append(self.solution_split[1])
        F = sum(eq.residual(sol) for eq, sol in zip(self.equations, unknowns, strict=True))
        self._mechanics_form = F
        self._refuse_parent_coefficients_on_mechanics_facets(F)
        multipliers = [(self.solution_split[i], self.tests[i])
                       for i in self.multiplier_indices]
        F += self.gravity.residual(
            ctx.fields["potential"], ctx.tests["potential"],
            ctx.fields["displacement"], density=self.approximation.density,
            theta_psi=self.theta_psi, multipliers=multipliers)
        for coupling in self.couplings:
            F += coupling.residual(ctx)
        # Every interior-facet integrand restricts its parent quantities where
        # it is built (`gadopt.utility.interior_facet_integrand`).
        self.F = F
        self.strong_bcs.extend(self.gravity.dirichlet_bcs(
            self.solution_space.sub(self.potential_index)))
        self._detect_solution_dependence()

    def _check_row_scaling_meshes(self) -> None:
        """Refuse a row scaling (`c`, `B_mu`, `Lambda`) on a mesh other than the parent.

        The three scale whole rows of both meshes: `theta_psi = c B_mu /
        Lambda` scales the potential rows, which are integrals over every
        cell of the parent, and `c B_mu` the facet rows of the couplings,
        where they are restricted as quantities of the parent
        (`CouplingContext.on_parent_facet`). A quantity of the mechanics
        mesh has no value on the cells of the parent outside the mechanics
        mesh, and UFL refuses its restriction on an exterior facet of the
        mechanics mesh. So each must be a number, a `Constant`, or a
        Function of the parent; a `Real` Function of the parent is the
        form of a control.

        Raises:
          ValueError: naming the row scaling that lives on another mesh.
        """
        if not self.cross_mesh:
            return
        scalings = {"scaling_factor": self.scaling_factor,
                    "approximation.B_mu": self.approximation.B_mu,
                    "approximation.self_gravity_number (Lambda)": self.Lambda}
        for label, value in scalings.items():
            if isinstance(value, Number):
                continue
            meshes = set(extract_domains(value))
            if meshes - {self.potential_mesh}:
                raise ValueError(
                    f"{label} lives on a mesh other than the potential's "
                    "(parent) mesh. It scales the potential rows over every "
                    "cell of the parent and the coupling rows on the facets "
                    "of the parent, so give it as a number, a Constant or a "
                    "Function of the parent mesh, for example "
                    "Function(FunctionSpace(psi_mesh, 'R', 0)).")

    def _refuse_parent_coefficients_on_mechanics_facets(self, mechanics_form) -> None:
        """Refuse a Function of the parent in a facet term of the momentum equation.

        The momentum equation pairs its exterior facets with the parent's
        cells (`Equation` with `intersect_measures`), because it cannot know
        which of its boundaries are interior facets of the parent. On such a
        facet the residual and the tangent of a coefficient of the parent
        are right, and Firedrake's adjoint of that coefficient is wrong for
        every element measured on the 2-D annulus and the unit square, on 1
        and 2 ranks:

        - `Real` and scalar DG0: the adjoint is assembled on both parent
          cells of the facet. For a `Real` the adjoint pairing is 2.000
          times the tangent pairing; for DG0 the cell outside the mechanics
          mesh gets a contribution too (a ratio of 1.88 to 2.0, depending on
          the direction).
        - CG1, CG2, DG1, DG2 and vector DG0: the adjoint changes from one
          assembly to the next. Mostly by a denormal number, sometimes by
          1e100 or more; the values of the cell outside the mechanics mesh
          are read without being set.

        A gradient through such a coefficient is wrong with no error. The
        same field as a Function of the mechanics mesh has no such pairing
        and is exact. Volume terms on the parent's cells are exact and
        repeatable for every element, and so are the coupling rows, which
        use the facet-to-facet measure (`CouplingContext.surface_measure`).
        Only a boundary that is an interior facet of the parent is affected;
        the check refuses the coefficient on every boundary, because the
        form does not say which boundary is which.

        A row scaling cannot move to the mechanics mesh
        (`_check_row_scaling_meshes`), so a `scaling_factor` that is a
        Function is refused here together with a weak displacement condition
        or a `normal_stress`, which are the terms that put it on such a
        facet.

        This check reads the coefficients of the form and changes nothing.
        It can go once Firedrake assembles the adjoint of a parent
        coefficient on such a facet from the mechanics side only, as it does
        in the tangent.

        Args:
          mechanics_form: the residual of the momentum (and DG history)
            equation, before the potential and coupling rows are added.

        Raises:
          ValueError: naming the coefficient and what to do instead.
        """
        if not self.cross_mesh:
            return
        for integral in mechanics_form.integrals_by_type("exterior_facet"):
            for coefficient in extract_coefficients(integral.integrand()):
                if not isinstance(coefficient, Function):
                    continue
                if coefficient.function_space().mesh() is not self.potential_mesh:
                    continue
                real = coefficient.ufl_element().family() == "Real"
                # How the messages name the coefficient's kind.
                kind = "Real" if real else "Function"
                noun = "Real Function" if real else "Function"
                if coefficient is self.scaling_factor:
                    advice = (
                        "scaling_factor must live on the parent mesh "
                        f"(it scales the potential rows too), so a {kind} "
                        "scaling_factor cannot be used with a weak "
                        "displacement condition or a normal_stress in "
                        "bcs; give it as a number or a Constant")
                elif real:
                    advice = (
                        "define it as a Real Function of the mechanics "
                        "mesh, Function(FunctionSpace(u_mesh, 'R', 0)), "
                        "which gives the same residual and an exact "
                        "derivative")
                else:
                    advice = (
                        "define it as a Function of the mechanics mesh "
                        "(the same element on u's mesh, for example by "
                        "interpolating the expression there), which gives "
                        "the same residual and an exact derivative")
                raise ValueError(
                    f"The {noun} {coefficient.name()!r} of the potential's "
                    "(parent) mesh appears in a boundary term of the "
                    "momentum equation (a weak displacement condition or a "
                    "normal_stress in bcs). Firedrake assembles its adjoint "
                    "wrongly on a boundary that is an interior facet of the "
                    "parent (twice for a Real or DG0 field, from unset "
                    "values for other elements), so a gradient with respect "
                    f"to it would be wrong; {advice}.")

    def _detect_solution_dependence(self) -> None:
        """Decide from the forms whether the Jacobian moves inside one solve.

        `jacobian_depends_on_solution` is true when the solution is a
        coefficient of the Jacobian of the residual (a power law on the DG
        family, a live coastline), or, on the Quadrature family, when the
        local constitutive equation depends on the displacement (a power law
        or a nonlinear backstress). `rheology_nonlinear` is the same test on
        the material alone; it decides whether the condensed displacement
        operator is symmetric. UFL folds a numeric exponent of one out of
        the power-law factor, so a Newtonian rheology is detected; a
        `Constant` exponent of value one is not folded and counts as a power
        law, which is correct and costs a preconditioner rebuild per solve.
        """
        def depends(form, unknown):
            jacobian = expand_derivatives(derivative(form, unknown))
            return any(c is unknown for c in extract_coefficients(jacobian))

        if self.family == "Quadrature":
            material = self.history.material
            rheology = any(c is material.u for c in extract_coefficients(material.equation))
        else:
            rheology = depends(self._mechanics_form, self.solution)
        self.rheology_nonlinear = rheology
        self.jacobian_depends_on_solution = rheology or depends(self.F, self.solution)
        jacobian = expand_derivatives(derivative(self.F, self.solution))
        self._jacobian_coefficients = [
            *extract_coefficients(jacobian),
            *ufl.algorithms.analysis.extract_type(jacobian, Constant)]
        self._jacobian_fingerprint = self._current_fingerprint()
        self._operator_version = 0

    def set_nullspace(self, nullspace) -> None:
        """Build the declared kernel on the mixed space and check it.

        A declared mode that is not a kernel would be projected out of the
        answer after every solve with no error, so each mode is checked here
        (`_check_kernel`): it must meet the strong displacement conditions,
        leave the density moments up to degree 4 unchanged, and the Jacobian
        must nearly annihilate it.
        """
        self.nullspace = self.transpose_nullspace = None
        # The near-nullspace of the outer operator is not used: GAMG runs on
        # the condensed displacement operator inside block 0, and gets its
        # modes from `condensed_field_near_nullspace` (`near_nullspace`).
        self.near_nullspace = None
        if nullspace is None:
            return
        V = self.solution_space.sub(0)
        if isinstance(nullspace, str):
            if nullspace != "rigid_rotation":
                raise ValueError(
                    f"nullspace must be None, 'rigid_rotation' or a "
                    f"VectorSpaceBasis, got {nullspace!r}.")
            if self.has_strong_displacement_bc:
                raise ValueError(
                    "nullspace='rigid_rotation' with a strong displacement "
                    "condition: a pinned boundary removes the rotation.")
            X = SpatialCoordinate(self.mesh)
            if len(X) == 2:
                modes = [as_vector([-X[1], X[0]])]
            else:
                modes = [as_vector([Constant(0.0), -X[2], X[1]]),
                         as_vector([X[2], Constant(0.0), -X[0]]),
                         as_vector([-X[1], X[0], Constant(0.0)])]
            vectors = [Function(V).interpolate(m) for m in modes]
        elif isinstance(nullspace, VectorSpaceBasis):
            # The vectors are Functions on the caller's displacement space,
            # whose layout is the one of the displacement block here.
            vectors = []
            for vector in nullspace._vecs:
                f = Function(V)
                with vector.dat.vec_ro as source, f.dat.vec_wo as target:
                    source.copy(target)
                vectors.append(f)
        else:
            raise TypeError(
                "nullspace must be None, 'rigid_rotation' or a VectorSpaceBasis "
                "on the displacement space.")
        basis = VectorSpaceBasis(vectors)
        basis.orthonormalize()
        entries = [self.solution_space.sub(i) for i in range(len(self.solution_space))]
        entries[0] = basis
        mixed = MixedVectorSpaceBasis(self.solution_space, entries)
        check_kernel_basis(mixed)
        with stop_annotating():
            self._check_kernel(basis)
        self.nullspace = self.transpose_nullspace = mixed

    #: The highest polynomial degree of the density measure of
    #: `_check_kernel`, so the highest harmonic degree of a lateral density
    #: that the measure sees.
    NULLSPACE_DENSITY_DEGREE = 4

    def _density_moment_measure(self):
        r"""Return a function that gives the density measure `L` of one mode.

        `L = max_v |int rho_0 (e . grad v) dx| / int |rho_0| |e . grad v| dx`
        over the monomials `v` of degree 1 to `NULLSPACE_DENSITY_DEGREE`
        (`_check_kernel` gives its meaning). With four degrees there are 34
        monomials in 3-D, and each needs two integrals per mode. As forms,
        that is either 68 kernels per mode or one 1-form on a mixed space of
        34 `Real` spaces, and both are slow to compile and to assemble on a
        production mesh. The integrals are therefore evaluated at the
        quadrature points: `rho_0` times the quadrature weight is assembled
        once as a 1-form on a `Quadrature` space (a test function of that
        space is one at its own point and zero at the others, so the
        assembled value is `w_p rho_0(x_p)`), the coordinates and the mode
        are interpolated onto the same points, and the monomials and their
        gradients are evaluated with NumPy. Each integral is then a sum over
        the points of the rank and one sum over the ranks.

        The quadrature degree is UFL's estimate for the integrand of the
        highest monomial degree, the degree that `assemble` would choose for
        the same integrals as forms, so the numbers do not depend on how
        they are evaluated.

        Returns:
          `measure(vector)`, which takes the mode as a Function on the
          displacement space and returns `L` as a float, the same on every
          rank.
        """
        from firedrake import TestFunction, VectorFunctionSpace
        from ufl.algorithms import estimate_total_polynomial_degree

        mesh = self.mesh
        dim = mesh.geometric_dimension
        degree = self.NULLSPACE_DENSITY_DEGREE
        X = SpatialCoordinate(mesh)
        rho = self.approximation.density
        # The exponents of every monomial of degree 1 to `degree`, one row
        # per monomial: (2, 0, 1) is x^2 z.
        exponents = np.array([
            np.bincount(np.array(powers, dtype=int), minlength=dim)
            for d in range(1, degree + 1)
            for powers in itertools.combinations_with_replacement(range(dim), d)])
        # The quadrature degree of the integrand with the highest monomial.
        highest = X[0] ** degree
        q = estimate_total_polynomial_degree(
            rho * dot(self.solution.subfunctions[0], grad(highest)))
        Q = FunctionSpace(mesh, "Quadrature", q)
        QV = VectorFunctionSpace(mesh, "Quadrature", q)
        metadata = {"quadrature_degree": q}
        # `w_p rho_0(x_p)` per point; `dx_m` reads a density given on the
        # parent mesh through its intersection measure.
        weighted = assemble(rho * TestFunction(Q) * self.gravity.dx_m(metadata=metadata))
        w_rho = np.ravel(weighted.dat.data_ro)
        points = Function(QV).interpolate(X).dat.data_ro.reshape(-1, dim)
        at_points = Function(QV)
        comm = mesh.comm

        def measure(vector):
            e = at_points.interpolate(vector).dat.data_ro.reshape(-1, dim)
            change = np.empty(len(exponents))
            scale = np.empty(len(exponents))
            for k, powers in enumerate(exponents):
                # grad v . e with d/dx_j prod_i x_i^p_i = p_j x_j^(p_j - 1) prod_{i != j} x_i^p_i.
                rate = np.zeros(len(points))
                for j in range(dim):
                    if powers[j] == 0:
                        continue
                    shifted = powers.copy()
                    shifted[j] -= 1
                    rate += powers[j] * np.prod(points ** shifted, axis=1) * e[:, j]
                change[k] = np.sum(w_rho * rate)
                scale[k] = np.sum(np.abs(w_rho) * np.abs(rate))
            # The owned points of every rank: one sum over the ranks.
            change = comm.allreduce(change, op=MPI.SUM)
            scale = comm.allreduce(scale, op=MPI.SUM)
            mask = scale > 0
            return float(np.max(np.abs(change[mask]) / scale[mask])) if mask.any() else 0.0

        return measure

    def _check_kernel(self, basis) -> None:
        r"""Check each declared mode in three ways: boundary values, density, Jacobian.

        **Strong conditions.** A mode that is not zero on the nodes of a
        strong displacement condition is not in the constrained space at
        all, and is refused.

        **Density moments.** A rigid rotation `e` is a kernel of the
        continuous problem only if it moves no density, `e . grad(rho_0) = 0`.
        The measure is

            L = max_v |int rho_0 (e . grad v) dx| / int |rho_0| |e . grad v| dx

        over the monomials `v` of degree 1 to `NULLSPACE_DENSITY_DEGREE` (4).
        For a rotation `int rho_0 (e . grad v)` is the rate of change of the
        moment `int rho_0 v` under the rotation, so `L = 0` says that the
        moments up to degree 4 of the density, its harmonic content up to
        degree 4, do not change. That is what the measure proves. It does not
        prove `e . grad(rho_0) = 0`: a lateral variation of harmonic degree 5
        or higher has unchanged moments and passes, and a localised anomaly is
        seen through its low-degree content only. The same moments are the
        volume parts of the centre-of-mass rows (degree 1) and of the rotation
        rows (degree 2), so for those rows the measure is exact; the potential
        row sees every degree.

        For a density that depends on radius alone every such moment is
        unchanged, whatever the representation: a DG0 or CG1 Function or a
        jump that is not on cell faces leaves a cell-scale staircase that the
        smooth `v` averages out (round-off on the symmetric test annulus, up
        to 1.4e-3 at degree 2 with its interior nodes moved at random by a
        sixth of a cell). A lateral variation `1 + a x` gives `L = 1.41 a` on
        two mesh sizes and at every bulk modulus. The measure cannot tell a
        lateral variation smaller than the staircase from the staircase.
        Above `nullspace_rtol` the constructor raises; above a tenth of it,
        it warns.

        **Jacobian action.** `|J e| / |J s|` with the reference `s = r^2 e`
        at the same coefficient norm. For a rotation `s` is a divergence-free
        shear that is tangential on every sphere, so the bulk modulus does
        not enter `J s`. This measure sees every way in which `e` fails to
        be a kernel of the discrete operator, including two that are not
        errors of the declaration: a density that the mesh represents with a
        cell-scale staircase (1.6e-2 for a DG0 radial density on the coarse
        test annulus), and the facet geometry under a weak `un` condition,
        whose Nitsche penalty carries the bulk modulus (0.35 for a valid
        rigid core at a bulk modulus of 1e3 to 1e5). It warns above 0.1 and
        refuses above `nullspace_jacobian_rtol` (0.5 by default), where the
        mode is about as stiff as a shear: measured 2.2 to 2.5 for `un` about
        an off-centre point or a declared translation. It is not a test of
        the density: a 30 percent lateral density gives 0.02 to 0.14.

        **Cost.** The moments are evaluated at the quadrature points
        (`_density_moment_measure`), which costs one assembly and two
        interpolations per mode. The shear reference is measured once, on the
        first mode, so the Jacobian probe costs two actions per mode (`J e`
        and `J^T e`) and two in all for the reference. A non-finite ratio is
        refused, as it would pass every bound. If the time step is not
        positive at construction (a driver that assigns the first step
        later), the probe runs at a unit step, because the DG history
        equation divides by the step; the caller's value is put back.
        `check_nullspace=False` skips the moments and the Jacobian probe.

        All numbers are logged at every construction and kept in
        `kernel_residuals` as `(L, |J e|/|J s|, |J^T e|/|J^T s|)`.
        """
        J = derivative(self.F, self.solution)
        X = SpatialCoordinate(self.mesh)
        moments = self._density_moment_measure() if self.check_nullspace else None

        def size(direction, transpose=False):
            form = action(adjoint(J) if transpose else J, direction)
            with assemble(form).dat.vec_ro as r:
                return r.norm()

        def unit(field_expression=None, vector=None):
            out = Function(self.solution_space)
            if vector is not None:
                with vector.dat.vec_ro as source, \
                        out.subfunctions[0].dat.vec_wo as target:
                    source.copy(target)
            else:
                out.subfunctions[0].interpolate(field_expression)
            with out.dat.vec_rw as v:
                v.scale(1.0 / v.norm())
            return out

        self.kernel_residuals = []
        # The DG history equation divides by the time step, so a zero step
        # (a driver that assigns the first step after construction) makes
        # the history rows of `J e` non-finite. The probe then runs at a
        # unit step, and the caller's value is put back afterwards; the
        # kernel property does not depend on the step size.
        step = scalar_value(self.dt)
        probe_step = self.check_nullspace and not (np.isfinite(step) and step > 0)
        if probe_step:
            self.dt.assign(1.0)
        try:
            self._check_modes(basis, moments, unit, size, X)
        finally:
            if probe_step:
                self.dt.assign(step)

    def _check_modes(self, basis, moments, unit, size, X) -> None:
        """The loop of `_check_kernel` over the declared modes.

        Args:
          basis: the `VectorSpaceBasis` of the displacement modes.
          moments: the density measure of `_density_moment_measure`, or
            `None` when `check_nullspace` is off.
          unit: makes a mixed Function of unit norm from a mode.
          size: the norm of a Jacobian action (or of its transpose).
          X: the coordinates of the mechanics mesh.
        """
        reference = None
        for vector in basis._vecs:
            mode = unit(vector=vector)
            for bc in self.displacement_bcs:
                # The condition lives on the displacement component of the
                # mixed space, so it zeroes a copy of the whole mixed mode.
                constrained = mode.copy(deepcopy=True)
                bc.zero(constrained)
                with mode.subfunctions[0].dat.vec_ro as x, \
                        constrained.subfunctions[0].dat.vec_ro as y:
                    off = (x - y).norm() / x.norm()
                if off > 1e-12:
                    raise ValueError(
                        f"A declared nullspace mode is not zero on boundary "
                        f"{bc.sub_domain} of a strong displacement condition "
                        f"(relative size {off:.3e} there), so it is not in the "
                        "constrained space and cannot be a kernel.")
            if moments is None:
                continue
            lateral = moments(vector)
            if reference is None:
                # The shear reference sets the scale of a stiff mode. Every
                # rigid rotation gives the same shear stiffness up to the
                # geometry, so it is measured once, on the first mode, and
                # the check costs two actions per mode (`J e`, `J^T e`).
                shear = unit(field_expression=dot(X, X) * vector)
                reference = (size(shear), size(shear, transpose=True))
            ratio = size(mode) / reference[0]
            ratio_t = size(mode, transpose=True) / reference[1]
            if not (np.isfinite(ratio) and np.isfinite(ratio_t)):
                # A NaN compares False with every bound, so it would pass
                # the check below without testing anything.
                raise ValueError(
                    f"The Jacobian probe of a declared nullspace mode is not "
                    f"finite (|J e|/|J s| = {ratio}, |J^T e|/|J^T s| = "
                    f"{ratio_t}), so the mode cannot be checked. The Jacobian "
                    "at the state of construction has a non-finite entry; "
                    "pass check_nullspace=False to build without the check.")
            self.kernel_residuals.append((lateral, ratio, ratio_t))
            log(f"{self.name}: nullspace mode: density variation L = "
                f"{lateral:.3e} (refused above {self.nullspace_rtol:g}); "
                f"|J e|/|J s| = {ratio:.3e}, |J^T e|/|J^T s| = {ratio_t:.3e} "
                f"(refused above {self.nullspace_jacobian_rtol:g})")
            if lateral > self.nullspace_rtol:
                raise ValueError(
                    f"A declared nullspace mode moves density: L = {lateral:.3e}, "
                    f"above nullspace_rtol={self.nullspace_rtol:g}: the "
                    "rotation changes a density moment of degree up to "
                    f"{self.NULLSPACE_DENSITY_DEGREE}. The declaration would "
                    "remove a part of the solution that this lateral density "
                    "drives.")
            if lateral > 0.1 * self.nullspace_rtol:
                warn(f"A declared nullspace mode moves density: L = "
                     f"{lateral:.3e}, within a factor 10 of nullspace_rtol.")
            jacobian = max(ratio, ratio_t)
            if jacobian > self.nullspace_jacobian_rtol:
                raise ValueError(
                    f"A declared nullspace mode is not a kernel of the Jacobian: "
                    f"|J e| / |J s| = {ratio:.3e}, |J^T e| / |J^T s| = "
                    f"{ratio_t:.3e} against the shear s = r^2 e, above "
                    f"nullspace_jacobian_rtol={self.nullspace_jacobian_rtol:g}. "
                    "A rigid rotation is a kernel only with no pinned "
                    "displacement and 'un' on spheres about the origin.")
            if jacobian > 0.1:
                warn(f"A declared nullspace mode is far from a kernel of the "
                     f"discrete Jacobian: |J e| / |J s| = {jacobian:.3e}. A "
                     "weak 'un' condition at a large bulk modulus does this "
                     "through the facet geometry; a lateral density of "
                     f"harmonic degree above {self.NULLSPACE_DENSITY_DEGREE}, "
                     "which the density measure does not see, does it too.")

    def set_lowrank_operator(self) -> None:
        """Build `theta_psi B` on the low-rank path; `None` on the multiplier path."""
        self.dtn_operator = None
        if self.dtn_representation == "lowrank":
            self.dtn_operator = self.gravity.build_lowrank_operator(
                self.solution_space, self.potential_index,
                lambda: self.theta_psi_value)

    # -- Solver options ------------------------------------------------------------

    def set_solver_options(self, solver_parameters, preset_options,
                           solver_parameters_extra) -> None:
        """The PETSc options, from a preset the solver calls or a complete dictionary.

        `"direct"` is LU on block 0 of the two-block Schur split (2-D only).
        `"iterative"` is `selfgrav_dtn_iterative_solver_parameters`, called
        here with the counts this solver knows and the caller's
        `preset_options`. A dictionary is used as given.
        """
        self.appctx = {"mu": self.shear_coefficient / self.approximation.rho_continuity()}
        preset_options = dict(preset_options or {})
        if isinstance(solver_parameters, Mapping):
            if preset_options:
                raise ValueError("preset_options needs a preset, not a dictionary.")
            self._check_dictionary(solver_parameters)
            self.add_to_solver_config(solver_parameters)
        else:
            if solver_parameters not in (None, "direct", "iterative"):
                raise ValueError("solver_parameters must be 'direct', 'iterative', "
                                 "None or a dictionary.")
            preset = solver_parameters or (
                "direct" if self.mesh.topological_dimension == 2 else "iterative")
            reserved = {"condensed", "dtn_representation", "n_real"} & preset_options.keys()
            if reserved:
                raise ValueError(f"The solver supplies {sorted(reserved)}; remove "
                                 "them from preset_options.")
            if preset == "direct" and preset_options:
                raise ValueError("preset_options belongs to the iterative preset.")
            self.add_to_solver_config(newton_stokes_solver_parameters)
            n_real = len(self.real_indices)
            quadrature = self.family == "Quadrature"
            if self.dtn_representation == "lowrank" and n_real == 0:
                if quadrature and preset == "iterative":
                    # No `Real` block: the reduced (u, psi) block is the whole
                    # system and `ReducedGIABlockPC` is the outer preconditioner.
                    options = selfgrav_dtn_iterative_solver_parameters(
                        condensed=True, dtn_representation="lowrank", n_real=0,
                        **preset_options)
                    prefix = "dtn_fieldsplit_0_condensed_"
                    reduced = {k: v for k, v in options.items() if not k.startswith("dtn_")}
                    reduced.update({"condensed_" + k[len(prefix):]: v
                                    for k, v in options.items() if k.startswith(prefix)})
                    reduced.update(pc_type="python",
                                   pc_python_type="gadopt.ReducedGIABlockPC")
                    self.add_to_solver_config(reduced)
                else:
                    if preset_options:
                        raise ValueError(
                            "This configuration (DG history, low-rank DtN, no "
                            "Real field) uses the low-rank direct preset; "
                            "preset_options has no meaning there.")
                    self.add_to_solver_config(selfgrav_dtn_lowrank_direct_solver_parameters)
            elif preset == "direct":
                self.add_to_solver_config(
                    selfgrav_dtn_schur_lowrank_solver_parameters
                    if self.dtn_representation == "lowrank"
                    else selfgrav_dtn_schur_solver_parameters)
            else:
                self.add_to_solver_config(selfgrav_dtn_iterative_solver_parameters(
                    condensed=quadrature, dtn_representation=self.dtn_representation,
                    n_real=n_real, **preset_options))
        self.add_to_solver_config(solver_parameters_extra)
        self._solver_parameters_extra = dict(solver_parameters_extra or {})
        self._refuse_ksponly_on_a_nonlinear_residual()
        self._attach_context(solver_parameters_extra)
        self.register_update_callback(self._rebuild_step)

    def _check_dictionary(self, parameters) -> None:
        """Refuse a dictionary whose block-0 route does not fit this space.

        Two disagreements are silent otherwise: a block-0 class that expects
        a history field the space does not hold (or the reverse), and a
        potential split that does not match the DtN representation.
        """
        block0 = parameters.get("dtn_fieldsplit_0_pc_python_type",
                                parameters.get("pc_python_type"))
        if block0 == "gadopt.CondensedBlockPC" and self.family != "DG":
            raise ValueError("gadopt.CondensedBlockPC eliminates a DG history "
                             "field; the Quadrature family has none.")
        if block0 == "gadopt.ReducedGIABlockPC" and self.family != "Quadrature":
            raise ValueError("gadopt.ReducedGIABlockPC expects (u, psi) in block "
                             "0; the DG family holds a history field there.")
        keys = ("dtn_fieldsplit_0_condensed_fieldsplit_1_pc_python_type",
                "condensed_fieldsplit_1_pc_python_type")
        lowrank_split = any(parameters.get(k) == "gadopt.LowRankPotentialPC" for k in keys)
        if lowrank_split and self.dtn_representation != "lowrank":
            raise ValueError("gadopt.LowRankPotentialPC needs dtn_representation='lowrank'.")
        if (self.dtn_representation == "lowrank" and block0 in
                ("gadopt.CondensedBlockPC", "gadopt.ReducedGIABlockPC")
                and not lowrank_split):
            raise ValueError("A low-rank solver needs gadopt.LowRankPotentialPC on "
                             "the potential split of block 0.")
        pair = ("dtn_fieldsplit_0_fieldsplit_0_pc_python_type",
                "fieldsplit_0_pc_python_type")
        if (self.dtn_representation == "lowrank"
                and any(parameters.get(k) == "gadopt.InternalVariableSCPC" for k in pair)):
            raise ValueError("block0='pair' has no potential split for the "
                             "low-rank update; use block0='condensed'.")

    def _refuse_ksponly_on_a_nonlinear_residual(self) -> None:
        """Refuse `ksponly` on a residual whose Jacobian depends on the solution.

        One linear solve at the initial state reports convergence and returns
        a state wrong by the whole nonlinearity, with nothing in the log. The
        message names each cause that is present and the way out of it, so a
        reader with a Newtonian rheology and a live coastline is not sent to
        the rheology, and the other way round.
        """
        if not (self.solver_parameters.get("snes_type") == "ksponly"
                and self.jacobian_depends_on_solution):
            return
        causes, remedies = [], []
        if self.rheology_nonlinear:
            exponent = getattr(self.approximation, "exponent", None)
            causes.append("the nonlinear rheology (a power law"
                          + (f" of exponent={exponent!r}" if exponent is not None else "")
                          + " or a nonlinear backstress)")
            remedies.append("use a Newtonian rheology if that is what is meant")
        if any(getattr(c, "fixed_ocean", True) is False for c in self.couplings):
            causes.append("the live sea-level sheet of SeaLevel(fixed_ocean=False), "
                          "which follows the sea level and so the unknowns")
            remedies.append("set SeaLevel(fixed_ocean=True) if the coastline is "
                            "meant to be frozen")
        if not causes:
            # A cause of a solution-dependent Jacobian that this list does
            # not name; the general sentence still states the refusal.
            causes.append("a term of the residual")
        verb = "makes" if len(causes) == 1 else "make"
        raise ValueError(
            f"snes_type='ksponly' is refused because {' and '.join(causes)} "
            f"{verb} the residual nonlinear, so one linear solve at the initial "
            "state converges by definition and returns a state that is wrong "
            "by the whole nonlinearity. Use snes_type='newtonls' (the default "
            "of the presets)" + "".join(f", or {r}" for r in remedies) + ".")

    def _attach_context(self, extras) -> None:
        """Publish what the preconditioners read in the application context.

        `approximation`, `dt`, `scaling_factor`, `time_alpha`, the reuse
        counter `operator_version`, `gia_jacobian_depends_on_solution`,
        `gia_solve_index`, the displacement parts of the nullspaces and the
        near-nullspace provider of the condensed displacement operator, and
        on the low-rank path `dtn_operator`. Also replaces a CG on the
        condensed displacement split by a GMRES of the same length when the
        material makes that operator nonsymmetric.
        """
        self._adjust_options(extras)
        ctx = self.appctx
        ctx["approximation"] = self.approximation
        ctx["dt"] = self.dt
        ctx["scaling_factor"] = self.scaling_factor
        ctx["time_alpha"] = self.time_integrator.alpha
        ctx["operator_version"] = self._operator_version
        ctx["gia_jacobian_depends_on_solution"] = self.jacobian_depends_on_solution
        ctx["gia_solve_index"] = 0
        if self.dtn_operator is not None:
            ctx["dtn_operator"] = self.dtn_operator
        for key, basis in (("condensed_field_nullspace", self.nullspace),
                           ("condensed_field_transpose_nullspace", self.transpose_nullspace)):
            displacement_basis = _displacement_basis(basis)
            if displacement_basis is not None:
                ctx[key] = _basis_provider(displacement_basis)
        if self.near_nullspace_modes != "none":
            modes = self.near_nullspace_modes

            def near_nullspace_provider(space, modes=modes):
                return near_nullspace_basis(space, modes)
            ctx["condensed_field_near_nullspace"] = near_nullspace_provider

    def _adjust_options(self, extras) -> None:
        """The option rules that depend on the residual, applied to the options.

        `gia_rebuild = always` on the Quadrature low-rank path with a
        state-dependent Jacobian, where a tangent frozen at the replay's zero
        initial guess is not good enough (a chosen lag policy is kept). A CG
        on the condensed displacement split becomes a GMRES of the same
        length when the material makes that operator nonsymmetric, unless
        the caller named the Krylov method in `extras`.
        """
        if (self.family == "Quadrature" and self.dtn_representation == "lowrank"
                and self.jacobian_depends_on_solution
                and "gia_rebuild" not in self.solver_parameters):
            self.add_to_solver_config({"gia_rebuild": "always"})
        for prefix in ("dtn_fieldsplit_0_fieldsplit_0_condensed_field_",
                       "dtn_fieldsplit_0_condensed_fieldsplit_0_",
                       "condensed_fieldsplit_0_"):
            key = prefix + "ksp_type"
            user_named = bool(extras) and key in extras
            if (self.solver_parameters.get(key) == "cg" and not user_named
                    and self.rheology_nonlinear):
                self.add_to_solver_config({
                    key: "gmres",
                    prefix + "ksp_gmres_restart":
                        self.solver_parameters.get(prefix + "ksp_max_it", 50)})

    def set_solver(self) -> None:
        """Build the step variant: stock, low-rank or quadrature."""
        if self.family == "Quadrature":
            self._step = _QuadratureStep(self)
        elif self.dtn_representation == "lowrank":
            self._step = _LowRankStep(self)
        else:
            self._step = _StockStep(self)
        #: The adopted low-rank solve block of the last annotated solve.
        self.adjoint_block = None
        #: Whether the low-rank adjoint carries the derivative of `theta_psi`.
        self._include_theta_derivative = True

    def _rebuild_step(self) -> None:
        """Rebuild the variational solver after an options change.

        The rules that read the options run again first: the `ksponly`
        refusal, the `gia_rebuild` rule and the Krylov method of the
        condensed displacement split.
        """
        self._refuse_ksponly_on_a_nonlinear_residual()
        self._adjust_options(self._solver_parameters_extra)
        self._step.build()

    @property
    def solver(self):
        """The variational solver of the step."""
        return self._step.solver

    @solver.setter
    def solver(self, value):
        """Replace the variational solver of the step.

        Used by `gadopt.dtn_coupled_adjoint.release_taping_solver`, which
        puts a stand-in that raises on use after it destroyed the taping
        solver of a finished forward run. The step owns the solver, so the
        stand-in goes there and every reader of `solver` sees it.
        """
        self._step.solver = value

    #: SNES and KSP options of the forward Newton loop that a linear
    #: derivative solve does not take; the lists of `StokesSolverBase`.
    FORWARD_ONLY_OPTIONS = (StokesSolverBase.FORWARD_ONLY_SNES_OPTIONS
                            + StokesSolverBase.FORWARD_ONLY_KSP_OPTIONS)

    def adjoint_solver_parameters(self) -> dict:
        """The options of every derivative solve: forward, linear, own cap.

        `ksponly` and no Newton tolerances, because the derivative systems are
        linear. No accepted cap on the outer Krylov solve, so a capped
        derivative solve raises. `derivative_max_it` replaces the forward
        cap when it is given.
        """
        parameters = deepcopy(self.solver_parameters)
        for option in self.FORWARD_ONLY_OPTIONS:
            parameters.pop(option, None)
        parameters["snes_type"] = "ksponly"
        if self.derivative_max_it is not None:
            parameters["ksp_max_it"] = self.derivative_max_it
        return parameters

    def adjoint_solve_keywords(self) -> dict:
        """The keywords of the stock path's derivative solvers, `{}` off the tape."""
        if not annotate_tape():
            return {}
        keywords = dict(getattr(self.solver, "_ad_kwargs", {}))
        keywords.pop("appctx", None)
        keywords["solver_parameters"] = self.adjoint_solver_parameters()
        return {"adj_kwargs": keywords}

    # -- The low-rank callbacks -------------------------------------------------------

    def augment_residual(self, X, F) -> None:
        """`post_function_callback`: add `theta_psi B psi` to the potential rows."""
        self.dtn_operator.apply_local(X.array_r, F.array_w)

    def augment_jacobian(self, X, Jmat) -> None:
        """`post_jacobian_callback`: give the matrix-free action the same `B`."""
        from .dtn_coupled import install_augmented_context
        install_augmented_context(Jmat, self.dtn_operator)

    # -- Operator reuse ------------------------------------------------------------------

    def _current_fingerprint(self) -> tuple:
        """A rank-local snapshot of every Jacobian coefficient but the solution.

        A Function on the `Real` space enters by its values, every other
        Function and every `Constant` by its `dat_version`. The time integrator assigns its `Real` coefficients (`alpha`, `dt`) before
        every solve, also when the value does not change, and every `assign`
        moves `dat_version`. Keyed on the version, the cached preconditioner
        pieces (dense Schur, the cached block-1 apply, the condensed block)
        would be rebuilt at every step at a fixed time step. A `Real`
        Function holds a few numbers that are the same on every rank, so a
        comparison by value costs nothing. A field Function keeps the
        version, because a comparison of its values would cost a copy of the
        field per solve. The rebuild once per tape sweep does not depend on
        this snapshot (`_RebuildRule` reads the sweep marker).
        """
        snapshot = []
        for coefficient in self._jacobian_coefficients:
            if coefficient is self.solution:
                continue
            dat = getattr(coefficient, "dat", None)
            version = getattr(dat, "dat_version", None)
            if (isinstance(coefficient, Function)
                    and coefficient.function_space().ufl_element().family() == "Real"):
                # Copy the values: `data_ro` is a view that a later assign changes.
                snapshot.append(("values", tuple(dat.data_ro.ravel().tolist())))
            elif version is not None:
                snapshot.append(("version", version))
            elif hasattr(coefficient, "values"):
                snapshot.append(("values", tuple(coefficient.values())))
        return tuple(snapshot)

    def invalidate_jacobian(self) -> None:
        """Tell the preconditioners to rebuild their cached operators."""
        self._operator_version += 1
        self.appctx["operator_version"] = self._operator_version

    def _refresh_operator_version(self) -> None:
        """Bump the operator version if a Jacobian coefficient changed.

        `None` when the Jacobian moves inside one solve, which tells every
        cache keyed on it to rebuild at each update.
        """
        if self.jacobian_depends_on_solution:
            self.appctx["operator_version"] = None
            return
        fingerprint = self._current_fingerprint()
        changed = self.mesh.comm.allreduce(
            fingerprint != self._jacobian_fingerprint, op=MPI.LOR)
        if changed:
            self._jacobian_fingerprint = fingerprint
            self.invalidate_jacobian()

    # -- The step -------------------------------------------------------------------

    def solve(self) -> None:
        """One time step.

        In order: the time coefficients, the 2-D enclosed mass, the step as
        one recorded block, the recorded kernel projection, the history
        accept, and the copy into `u` and `psi`.
        """
        from .dtn_coupled_adjoint import TapeSweepMarkerBlock
        self.time_integrator.prepare()
        self.update_total_mass()
        # Counted before the solve, so every preconditioner setup inside it
        # sees one index (`gadopt.preconditioners._RebuildRule`).
        self.appctx["gia_solve_index"] = self.appctx.get("gia_solve_index", 0) + 1
        markers = annotate_tape()
        if markers:
            get_working_tape().add_block(TapeSweepMarkerBlock(
                self, self.appctx["gia_solve_index"], before=True))
        self._step.solve()
        if markers:
            get_working_tape().add_block(TapeSweepMarkerBlock(
                self, self.appctx["gia_solve_index"], before=False))
        snes = self.solver.snes
        self.appctx["gia_last_solve_linear_its"] = int(snes.getLinearSolveIterations())
        self.appctx["gia_last_solve_newton_its"] = max(1, int(snes.getIterationNumber()))
        # Recorded. On the Quadrature family the step block has already
        # projected inside its own map, so this applies P a second time; P is
        # idempotent up to round-off and its adjoint is applied twice as well,
        # which is correct, and the forward values are those of the earlier
        # solver.
        project_out_kernel(self.nullspace, self.solution)
        self.history.advance(self)
        self._copy_out()

    def update_total_mass(self) -> None:
        """Refresh the 2-D monopole datum from the sheets of the couplings.

        The datum solves start from zero (`GravityPotential.update_total_mass`),
        so a run restarted from a checkpoint reproduces the datum bit for bit.
        The net-mass check is a diagnostic and is not recorded.
        """
        sheets = self.coupling_context.sheets
        self.gravity.update_total_mass(
            extra_masses=[(sh.density, sh.measure) for sh in sheets
                          if sh.in_monopole_datum])
        with stop_annotating():
            self.gravity.check_net_mass(
                extra_scales=[(sh.density, sh.measure) for sh in sheets])

    def _copy_out(self) -> None:
        """Write the displacement and the potential into `u` and `psi`, recorded."""
        self.u.assign(self.solution.subfunctions[0])
        self.psi.assign(self.solution.subfunctions[self.potential_index])

    def solved_history(self) -> Function:
        """The history of the state the last step solved, before it is accepted."""
        return self._step.solved_history()

    def restart_time_integration(self) -> None:
        """Take a backward-Euler step next, after a resolved forcing event."""
        self.time_integrator.restart()

    # -- Fields and diagnostics ---------------------------------------------------------

    @property
    def time_integrator(self):
        """The `BDF2History` of the material history."""
        return self.history.time_integrator

    @property
    def internal_variables(self) -> Function:
        """The accepted history level: the one history Function of the run."""
        return self.time_integrator.current[0]

    def field(self, name):
        """The UFL component `name` of the mixed solution, for forms and functionals."""
        return self.solution_split[self.field_index[name]]

    def real_field(self, name) -> Function:
        """The `Real` field `name` of the mixed solution, a view."""
        return self.solution.subfunctions[self.field_index[name]]

    def geoid(self, *, include_rotation: bool = True):
        """The geoid height `N = +(psi + psi_rot) / g_0`, UFL on the parent mesh."""
        total = self.approximation.geoid(self.field("potential"))
        ctx = self.coupling_context
        if include_rotation and ctx.rotational_potential is not None:
            total = total + self.approximation.geoid(
                ctx.rotational_potential(self.potential_mesh))
        return total

    def coefficients(self) -> dict:
        """The solved trace coefficients of every DtN boundary, keyed by marker.

        `{bc_id: {mode key: value}}`. On the low-rank path each value is a
        taped `AdjFloat` (`taped_coupled_trace_coefficients`); on the
        multiplier path it is the multiplier's `Real` Function, a view of the
        solution. Either keeps its dependency on the tape.
        """
        if self.dtn_representation == "lowrank":
            from .dtn_coupled import taped_coupled_trace_coefficients
            return taped_coupled_trace_coefficients(self)
        keys, fields = self.form.multiplier_keys, self.multiplier_indices
        if len(keys) != len(fields):
            raise RuntimeError("multiplier keys and fields disagree in number.")
        out = {bc_id: {} for bc_id, _ in self.form.dtn_boundaries}
        for (bc_id, key), i in zip(keys, fields):
            out[bc_id][key] = self.solution.subfunctions[i]
        return out

    def require_controls_reach_residual(self, *controls) -> None:
        """Raise if a control never enters the adopted low-rank solve block."""
        if self.adjoint_block is None:
            raise RuntimeError("Run one annotated low-rank solve first.")
        from .dtn_coupled_adjoint import require_controls_reach_block
        require_controls_reach_block(self.adjoint_block, controls)

    # -- State ---------------------------------------------------------------------------

    def state(self) -> dict:
        """The named Functions a restart needs, as views of the solver's own state.

        `displacement`, `potential`, every `Real` field by name
        (`dtn_multiplier_k`, `core_pressure`, `m1`/`m2`/`m3`,
        `centre_of_mass_i`, `sea_level_shift`), `history` (the accepted
        level), `history_previous` (BDF2 only) and `previous_dt`, a `Real`
        Function on the parent that holds the accepted step or -1 when the
        next step is a backward-Euler start. Writing a value into a returned
        Function changes the solver; use `restore` instead.
        """
        out = {name: self.solution.subfunctions[i]
               for i, name in enumerate(self.field_names)
               if name != "history"}
        out["history"] = self.internal_variables
        if self.time_integrator.previous is not None:
            out["history_previous"] = self.time_integrator.previous[0]
        previous_dt = self.time_integrator.previous_dt
        with stop_annotating():
            record = Function(FunctionSpace(self.potential_mesh, "R", 0),
                              name="previous_dt")
            record.assign(-1.0 if previous_dt is None else previous_dt)
        out["previous_dt"] = record
        return out

    def _state_targets(self) -> dict:
        """`{name: the Function restore writes}`."""
        targets = self.state()
        targets.pop("previous_dt")
        return targets

    def restore(self, state: Mapping) -> None:
        """Put a state back: from `state()` of a solver, or from `gadopt.gia_state`.

        Each value is a Function on the target's own space, a Function on
        the parent mesh in the checkpoint encoding of `gadopt.gia_state`, or
        a number for a `Real` field. Values are written with recorded
        assignments. The DG family's history field of the mixed solution,
        the initial guess of the next solve, is set to the restored history,
        and `u` and `psi` are written.

        `previous_dt` is required for a BDF2 solver: without it the next step
        would silently be a backward-Euler step. A backward-Euler solver
        accepts a state without it.

        The counters of the preconditioner caches (`gia_solve_index`, the
        operator version, the tape sweep marker) are not part of the state.

        The restored state equals the saved one bit for bit on any number of
        ranks. How the continued run compares with the uninterrupted run
        depends on the rank count. In serial, with the direct presets, it
        continues bit for bit. On more than one rank it agrees to the
        round-off of the direct solve and not bit for bit: the presets
        factorise block 0 with MUMPS, and two parallel factorisations of the
        same matrix in one process sum the contributions of the ranks in an
        order that depends on the timing of the messages, so their solutions
        differ in the last digits (measured up to about 2e-14 relative on two
        ranks). With a lagged rebuild of the `Real`-block complement
        (`gia_rebuild` lagged), the restored run rebuilds at its first solve
        where the uninterrupted run may keep a lagged complement, so the two
        agree to the solver tolerance and not bit for bit.
        """
        from .gia_state import decode_into
        targets = self._state_targets()
        unknown = set(state) - set(targets) - {"previous_dt"}
        if unknown:
            raise ValueError(f"Unknown state entries {sorted(unknown)}; this "
                             f"solver has {sorted(targets)}.")
        missing = set(targets) - set(state)
        if self.time_integrator.order == 2 and "previous_dt" not in state:
            missing.add("previous_dt")
        if missing:
            raise ValueError(f"The state lacks {sorted(missing)}.")
        for name, target in targets.items():
            decode_into(target, state[name], self.potential_mesh, name)
        if "previous_dt" in state:
            value = state["previous_dt"]
            value = scalar_value(value)
            if value != -1.0 and not (np.isfinite(value) and value > 0):
                raise ValueError("previous_dt must be positive or -1.")
            self.time_integrator.previous_dt = None if value == -1.0 else value
        self.time_integrator.prepared_dt = None
        if self.family == "DG":
            self.solution.subfunctions[1].assign(self.internal_variables)
        self._copy_out()

    def reset(self) -> None:
        """Return to the undeformed state, with a backward-Euler start next."""
        self.solution.assign(0.0)
        for level in (self.time_integrator.current, self.time_integrator.previous or []):
            for f in level:
                f.assign(0.0)
        self.time_integrator.restart()
        self.time_integrator.prepared_dt = None
        self._copy_out()
