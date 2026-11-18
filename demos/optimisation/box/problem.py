"""The inverse problem of the box optimisation benchmark.

This module owns the tape. ``build_problem`` clears the working tape, turns
annotation on, runs the forward model once from the control, assembles the objective,
builds the reduced functional and then pauses annotation. Everything downstream, that
is every optimiser in ``optimisers.py``, runs under ``stop_annotating`` and only ever
replays this tape. No other module in this demo may call ``continue_annotation``.

The problem is the twin experiment of the adjoint tutorial with the surface velocity
term removed:

    J(Tic) = int (T(t_final) - Tobs)^2 dx
           + alpha_d * (D_obs / D_damp)   * int (T0 - Taverage)^2 dx
           + alpha_s * (D_obs / D_smooth) * int |grad(T0 - Taverage)|^2 dx

where ``Tic`` is the control, a CG1 temperature at the start of the inversion window,
``T0`` its projection into CG1 with the strong boundary values applied, ``Taverage``
the horizontal layer average at the window start, and the ``D`` factors are the
normalisations of the tutorial so that the weights are dimensionless.

The reference trajectory comes from ``forward.py``. The window is the last ``window``
steps of that trajectory, so the true initial temperature of the window is the stored
temperature at timestep ``steps - window``, and the twin-experiment error is measured
against it.

The regularisation Riesz map lives here as well. The Hessian of the two
regularisation terms in the control is exact and cheap:

    H_reg = 2 alpha_d (D_obs / D_damp) P^T M P + 2 alpha_s (D_obs / D_smooth) P^T K P

with ``P`` the projection ``Tic -> T0`` (identity in the interior, boundary rows
replaced), ``M`` the CG1 mass matrix and ``K`` the CG1 stiffness matrix. The
inner product ``a(u, v) = c_d (u, v) + c_s (grad u, grad v)`` on the control space is
``H_reg`` without the boundary rows, which is what the optimisers use as the metric,
the initial inverse Hessian and the Krylov preconditioner.
"""

import gc

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from gadopt import *
from gadopt.inverse import *  # noqa: F401,F403 -- brings pyadjoint names and starts annotation
from firedrake.cofunction import RieszMap
from pyadjoint import annotate_tape, continue_annotation, get_working_tape, pause_annotation, stop_annotating

# The tutorial uses degree-6 quadrature for the misfit and regularisation integrals,
# so the functional here is the same object. Assemblies in the forward model keep the
# defaults of the solver classes.
_dx = dx(degree=6)


@dataclass
class InnerProduct:
    """A symmetric bilinear form on the control space, and the objects derived from it.

    Attributes:
        name: ``"L2"`` or ``"regularisation"``.
        form: a callable ``(u, v) -> UFL form`` with two arguments in the control
            space, so that ``assemble(form(x, y))`` is the inner product of two
            Functions and ``form(TrialFunction, TestFunction)`` is the bilinear form.
        riesz_map: a firedrake ``RieszMap`` that inverts the form. Passed to
            ``Control(..., riesz_map=...)`` so pyadjoint returns gradients and Hessian
            actions measured in this inner product.
        bcs: the homogeneous ``DirichletBC`` objects of the operator, or an empty
            list. With conditions, the operator is the form restricted to the free
            nodes: ``riesz_map`` returns ``A_ff^{-1} g_f`` on the free nodes and zero
            on the constrained ones, and ignores the entries of ``g`` there. ``form``
            stays the form without conditions, and it equals the operator on every
            pair of functions that vanish on the constrained nodes.
    """

    name: str
    form: Callable
    riesz_map: object
    bcs: list = field(default_factory=list)

    def norm(self, x):
        """Norm of a primal Function in this inner product."""
        return sqrt(assemble(self.form(x, x)))

    def dual_norm(self, g):
        """Norm of a dual Cofunction: sqrt(<g, R g>), with ``R`` the Riesz map."""
        return sqrt(assemble(action(g, self.riesz_map(g))))


@dataclass
class InverseProblem:
    """Everything an optimiser needs, plus what the twin-experiment record needs.

    Attributes:
        reduced_functional: the pyadjoint reduced functional over the control.
        control: the ``Control`` object.
        Tic: the control Function, a CG1 temperature. Its ``block_variable`` holds the
            current iterate during optimisation.
        Tic_true: the true initial temperature of the window, in the DQ2 state space.
        Tobs: the observed final temperature.
        T: the DQ2 state; its ``block_variable.checkpoint`` is the final temperature
            of the last replay.
        bounds: ``(lower, upper)`` Functions in the control space.
        inner_product: the ``InnerProduct`` the control carries.
        weights: ``{"alpha_d": ..., "alpha_s": ...}``.
        window: number of timesteps inverted.
        mesh: the mesh, for output.
        norm_obs: ``int Tobs^2 dx``, the normalisation of the objective. Used to turn
            the final misfit into a relative number.
        norm_ic: ``int Tic_true^2 dx``, the same for the initial misfit.
        delta_t: the timestep length of the reference run, non-dimensional.
        window_time: the physical length of the observation window,
            ``window * delta_t``. The field ``window`` is a step count.
        Taverage: the layer average at the window start, the reference of the
            regularisation terms.
        T0_bcs: the two ``DirichletBC`` objects that turn the control into ``T0``.
        m0: the prior mean, the layer average at the window start with the wall
            values imposed (bottom 1, top 0). It enters no term of the objective
            yet; it is defined here so that a prior term can use it.
        wall_nodes: ``{"bottom": array, "top": array}``, the owned local indices
            of the control nodes on the two walls, in the order of the PETSc
            vector of the control.
        wall_weights: ``{"bottom": Function, "top": Function}``, the sum of the
            CG1 hat functions of the nodes of each wall. With the remainder
            ``1 - bottom - top`` they form a partition of unity, which splits an
            integral into three parts that add up to the whole.
        wall_settings: the wall treatment of this problem, for the record.
    """

    reduced_functional: object
    control: object
    Tic: Function
    Tic_true: Function
    Tobs: Function
    T: Function
    bounds: tuple
    inner_product: InnerProduct
    weights: dict = field(default_factory=dict)
    window: int = 0
    mesh: object = None
    norm_obs: float = 1.0
    norm_ic: float = 1.0
    delta_t: Optional[float] = None
    window_time: Optional[float] = None
    Taverage: Optional[Function] = None
    T0_bcs: list = field(default_factory=list)
    m0: Optional[Function] = None
    wall_nodes: dict = field(default_factory=dict)
    wall_weights: dict = field(default_factory=dict)
    wall_settings: dict = field(default_factory=dict)

    def control_error(self, m=None):
        """Relative L2 error of a control value against the true initial temperature.

        Args:
            m: a Function in the control space. Defaults to the current tape value.

        Returns:
            ``||m - Tic_true|| / ||Tic_true||`` in L2. This is the primary measure
            of the twin experiment and does not exist for a real inversion.
        """
        if m is None:
            m = self.control.tape_value()
        return sqrt(assemble((m - self.Tic_true) ** 2 * _dx)) / sqrt(assemble(self.Tic_true ** 2 * _dx))

    def initial_misfit(self, m=None):
        """Initial misfit of the paper, ``int (Tic - Tic_true)^2 dx``.

        This is panel (c) of Fig. 5 of Ghelichkhan et al. (2024, GMD): the squared
        difference between the reconstructed initial condition and the reference
        initial condition, integrated over the domain. It is reported unnormalised,
        as the paper reports it, so that a run here can be put on the same axes as
        that figure. ``control_error`` is the same quantity as a relative norm.

        Args:
            m: a Function in the control space. Defaults to the current tape value.

        Returns:
            float
        """
        if m is None:
            m = self.control.tape_value()
        return float(assemble((m - self.Tic_true) ** 2 * _dx))

    def final_misfit(self):
        """Final misfit of the paper, ``int (T(t_F) - Tobs)^2 dx``, of the last replay.

        This is panel (b) of Fig. 5 of the same paper: the squared difference between
        the reconstructed final temperature and the observed final temperature. The
        value is taken from ``T.block_variable.checkpoint``, which holds the final
        temperature of the most recent forward replay, so it belongs to the control
        value of that replay and not to the current tape value.

        Returns:
            float
        """
        return float(assemble((self.T.block_variable.checkpoint - self.Tobs) ** 2 * _dx))

    def relative_final_misfit(self):
        """``final_misfit`` divided by ``int Tobs^2 dx``.

        The normalisation is the same ``norm_obs`` that scales the regularisation
        weights, so this number is the first term of the objective measured against
        the size of the observation.

        Returns:
            float
        """
        return self.final_misfit() / self.norm_obs

    def prior_term(self, m=None):
        """The regularisation part of the objective at a control value.

        The objective is the misfit term plus this term. The campaign calls it the
        prior term, because in the prior-covariance form of the problem it is
        ``1/2 ||m - m0||^2_{B^-1}``. Here it is still the legacy regularisation of
        the tutorial:

            c_d int (T0 - Taverage)^2 dx + c_s int |grad(T0 - Taverage)|^2 dx

        with ``T0`` the control projected into CG1 with the wall values imposed,
        the same projection and the same quadrature as the taped objective in
        ``build_problem``. It is evaluated off the tape, so it can be called from a
        reduced-functional callback.

        Args:
            m: a Function in the control space. Defaults to the current tape value.

        Returns:
            float. Exactly 0.0 when both weights are zero, which is the case of every
            run without regularisation in the objective; no assembly happens then.
        """
        c_d = self.weights.get("c_d", 0.0)
        c_s = self.weights.get("c_s", 0.0)
        if c_d == 0.0 and c_s == 0.0:
            return 0.0
        if m is None:
            m = self.control.tape_value()
        with stop_annotating():
            T0 = Function(m.function_space()).project(m, bcs=self.T0_bcs)
            value = 0.0
            if c_d > 0.0:
                value += c_d * float(assemble((T0 - self.Taverage) ** 2 * _dx))
            if c_s > 0.0:
                value += c_s * float(assemble(
                    dot(grad(T0 - self.Taverage), grad(T0 - self.Taverage)) * _dx))
        return value

    def measures(self, m=None):
        """The two misfits of the paper plus their relative forms, in one dictionary.

        Args:
            m: control value for the initial misfit. Defaults to the tape value.

        Returns:
            dict with ``initial_misfit``, ``final_misfit``, ``control_error`` and
            ``relative_final_misfit``.
        """
        out = {
            "initial_misfit": self.initial_misfit(m),
            "final_misfit": self.final_misfit(),
            "control_error": float(self.control_error(m)),
            "relative_final_misfit": self.relative_final_misfit(),
        }
        out.update(self.wall_measures(m))
        return out

    def wall_measures(self, m=None):
        """The initial misfit split into the two wall rows and the remainder.

        The split uses the partition of unity of ``wall_weights``:

            initial_misfit = int e^2 w_bottom dx + int e^2 w_top dx
                           + int e^2 (1 - w_bottom - w_top) dx,  e = m - Tic_true,

        with the same degree-6 quadrature as ``initial_misfit``, so the three
        parts add up to it to round-off. ``w_bottom`` is 1 on the bottom nodes and
        falls to 0 one cell above them, so the bottom part measures the error in
        the bottom cell row, weighted towards the wall. The nodal deviations are
        ``max |m - 1|`` on the bottom nodes and ``max |m|`` on the top nodes, the
        distance of the control from the wall values of the temperature; with
        pinned walls both are exactly zero.

        Args:
            m: a Function in the control space. Defaults to the current tape value.

        Returns:
            dict, empty when the problem holds no wall weights.
        """
        if not self.wall_weights:
            return {}
        if m is None:
            m = self.control.tape_value()
        error_sq = (m - self.Tic_true) ** 2
        w_bottom, w_top = self.wall_weights["bottom"], self.wall_weights["top"]
        values = m.dat.data_ro
        return {
            "initial_misfit_bottom": float(assemble(error_sq * w_bottom * _dx)),
            "initial_misfit_top": float(assemble(error_sq * w_top * _dx)),
            "initial_misfit_interior": float(
                assemble(error_sq * (1 - w_bottom - w_top) * _dx)),
            "wall_max_dev_bottom": float(np.max(np.abs(
                values[self.wall_nodes["bottom"]] - 1.0))),
            "wall_max_dev_top": float(np.max(np.abs(values[self.wall_nodes["top"]]))),
        }


def make_inner_product(Q1, name, c_d=1.0, c_s=0.0, bcs=None):
    """Build an ``InnerProduct`` on the control space.

    Args:
        Q1: the control function space.
        name: ``"L2"`` for the mass matrix, ``"regularisation"`` for
            ``c_d (u, v) + c_s (grad u, grad v)``.
        c_d: coefficient of the mass term. Ignored for ``"L2"``.
        c_s: coefficient of the stiffness term. Ignored for ``"L2"``.

    Returns:
        InnerProduct

    The Riesz solve is a CG1 mass or Helmholtz system, so the direct solver is
    cheap, and the operator never changes, so the factorisation is reused.
    """
    if name == "L2":
        def form(u, v):
            return inner(u, v) * dx
    elif name == "regularisation":
        c_d = Constant(c_d)
        c_s = Constant(c_s)

        def form(u, v):
            return c_d * inner(u, v) * dx + c_s * inner(grad(u), grad(v)) * dx
    else:
        raise ValueError(f"unknown inner product {name!r}; use 'L2' or 'regularisation'")

    solver_parameters = {"ksp_type": "preonly", "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"}
    if not bcs:
        riesz_map = RieszMap(
            form(TrialFunction(Q1), TestFunction(Q1)),
            solver_parameters=solver_parameters,
            constant_jacobian=True,
        )
        return InnerProduct(name=name, form=form, riesz_map=riesz_map)

    # The operator with homogeneous Dirichlet conditions. ``restrict=False`` keeps
    # the full space: firedrake replaces the constrained rows by identity rows and
    # sets the solution there to the condition value, zero, so the solve gives
    # ``A_ff u_f = g_f`` on the free nodes exactly, with the coupling block
    # multiplied by ``u_c = 0``. The default ``restrict=True`` builds a
    # RestrictedFunctionSpace, which gave a relative error of 1.25 against an
    # assembled reference on this mesh and once an IndexError
    # (NOTES/HANDOUT-BOX-PRIOR-2026-11-18.md, step S2).
    riesz_map = RieszMap(
        form(TrialFunction(Q1), TestFunction(Q1)),
        bcs=list(bcs), restrict=False,
        solver_parameters=solver_parameters,
        constant_jacobian=True,
    )
    return InnerProduct(name=name, form=form, riesz_map=riesz_map, bcs=list(bcs))


def build_problem(checkpoint, window=10, alpha_d=1e-3, alpha_s=1e-3,
                  inner_product="L2", initial_guess="average",
                  metric_alpha_d=None, metric_alpha_s=None,
                  prior_sigma=None, prior_L=None,
                  wall_treatment="free", prior_wall_bc="none",
                  initial_guess_walls=False,
                  checkpointing_schedule=None):
    """Tape the forward model once and return the inverse problem.

    Args:
        checkpoint: path of the reference state written by ``forward.py``.
        window: number of timesteps to invert, counted back from the end of the
            reference trajectory.
        alpha_d: weight of the damping term. Zero removes the term from the
            objective.
        alpha_s: weight of the smoothing term. Zero removes the term from the
            objective. Both zero leaves the temperature misfit alone, which is the
            ill-posed problem: the data do not determine the control in every
            direction, so the optimiser can walk away from the truth once it has
            fitted what the observation sees.
        inner_product: ``"L2"`` or ``"regularisation"``. See ``make_inner_product``.
        initial_guess: ``"average"`` starts from the layer average at the window
            start, as the tutorial does. ``"observed"`` starts from the observed
            final temperature, which is what Ghelichkhan et al. (2024, GMD) do.
            ``"true"`` starts from the true initial temperature, which is useful
            only to check that an optimiser stays put.
        metric_alpha_d: weight of the mass term of the ``"regularisation"`` inner
            product. None takes ``alpha_d``, so the metric is the Hessian of the
            damping term. A number given here separates the two, which is what makes
            the inner product usable when the objective carries no regularisation.
        metric_alpha_s: the same for the stiffness term and ``alpha_s``.
        prior_sigma: the formal standard deviation ``sigma`` of the prior
            statement of the metric. Given together with ``prior_L``, it replaces
            the two metric weights: ``metric_c_d = 1 / sigma^2`` and ``metric_c_s =
            L^2 / sigma^2``. The metric is then ``(1 / sigma^2) (M + L^2 K)``
            before the factor 2 below, a first-order precision with correlation
            length ``L``. ``sigma`` is formal: it is not a pointwise standard
            deviation (a first-order prior in two dimensions has no finite
            pointwise variance), and it is relative to an observation covariance
            of the identity, because the misfit term carries weight 1.
        prior_L: the correlation length ``L`` of the same statement, in the
            non-dimensional length of the box.
        wall_treatment: ``"free"`` keeps the bounds [0, 1] on every node.
            ``"pinned"`` gives the wall nodes equal lower and upper bounds, the
            wall values of the temperature (bottom 1, top 0). BNK puts a variable
            with equal bounds in its fixed set, so it never enters the Krylov
            system. The first guess then carries the wall values as well, which TAO
            would impose anyway when it projects the first point onto the bounds.
        prior_wall_bc: ``"none"`` builds the metric with no boundary condition (a
            Neumann stiffness matrix). ``"dirichlet"`` puts a homogeneous Dirichlet
            condition on the top and bottom walls into the operator, see
            ``make_inner_product``. Needs ``inner_product="regularisation"`` and
            ``wall_treatment="pinned"``.
        initial_guess_walls: set the wall nodes of the first guess to the wall
            values, without pinning them.
        checkpointing_schedule: a ``checkpoint_schedules`` schedule, or None to hold
            every state in memory. Hessian-vector products need None, because
            pyadjoint has no checkpoint-manager path for the tangent linear and
            second-order adjoint sweeps.

    Returns:
        InverseProblem

    Taping: the working tape is cleared and annotation turned on at entry, and
    annotation is paused before return. The reduced functional owns the tape from
    then on.
    """
    tape = get_working_tape()
    tape.clear_tape()
    if not annotate_tape():
        continue_annotation()
    if checkpointing_schedule is not None:
        tape.enable_checkpointing(checkpointing_schedule)

    # ---- Reference trajectory ------------------------------------------------------
    with CheckpointFile(checkpoint, mode="r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        mesh.cartesian = True
        T_info = f.get_timestepping_history(mesh, "Temperature")
        steps = int(T_info["index"][-1])  # Temperature has indices 0..steps
        delta_t_value = float(T_info["delta_t"][-1])
        if not 0 < window <= steps:
            raise ValueError(f"window must be in 1..{steps}, got {window}")
        initial_timestep = steps - window
        Tobs = f.load_function(mesh, "Temperature", idx=steps)
        Tic_true = f.load_function(mesh, "Temperature", idx=initial_timestep)
        Taverage = f.load_function(mesh, "Average_Temperature", idx=initial_timestep)
    Tobs.rename("Observed_Temperature")
    Tic_true.rename("True_Initial_Temperature")
    Taverage.rename("Average_Temperature")
    boundary = get_boundary_ids(mesh)

    # ---- Spaces and state ----------------------------------------------------------
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DQ", 2)
    Q1 = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)
    u, p = split(z)
    z.subfunctions[0].rename("Velocity")
    z.subfunctions[1].rename("Pressure")
    T = Function(Q, name="Temperature")

    approximation = BoussinesqApproximation(Constant(1e6))
    delta_t = Constant(delta_t_value)
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    stokes_bcs = {
        boundary.bottom: {"uy": 0},
        boundary.top: {"uy": 0},
        boundary.left: {"ux": 0},
        boundary.right: {"ux": 0},
    }
    temp_bcs = {
        boundary.bottom: {"T": 1.0},
        boundary.top: {"T": 0.0},
    }
    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    stokes_solver = StokesSolver(z, approximation, T, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

    # ---- Normalisations (constants, computed before the control enters the tape) ----
    # These are numbers, not tape entries, because they depend on data only. Same
    # definitions as the tutorial.
    norm_obs = assemble(Tobs ** 2 * _dx)
    norm_damping = assemble(Taverage ** 2 * _dx)
    norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * _dx)
    c_d = alpha_d * norm_obs / norm_damping if alpha_d > 0 else 0.0
    c_s = alpha_s * norm_obs / norm_smoothing if alpha_s > 0 else 0.0

    # ---- Control -------------------------------------------------------------------
    # The inner product must exist before the Control, because the Control carries the
    # Riesz map. The factor 2 makes the form equal to the Hessian of the regularisation
    # terms when the metric weights are the objective weights, which is the default.
    # When the objective has no regularisation the two are separate numbers, and the
    # metric is then an operator of the same shape that is not a term in J.
    if (prior_sigma is None) != (prior_L is None):
        raise ValueError("prior_sigma and prior_L must be given together")
    if prior_sigma is not None:
        # The prior statement of the same operator. The two coefficients are
        # numbers of their own and depend on neither the data nor the mesh, while
        # the legacy weights below are scaled by data norms, which change with the
        # window (norm_obs) and the mesh (norm_smoothing). The legacy weights are
        # left unset, so the record cannot claim a weight that was never used.
        if metric_alpha_d is not None or metric_alpha_s is not None:
            raise ValueError("give either prior_sigma and prior_L, or the metric "
                             "weights, not both")
        if inner_product != "regularisation":
            raise ValueError("prior_sigma and prior_L define the 'regularisation' "
                             f"inner product, and inner_product is {inner_product!r}")
        if prior_sigma <= 0 or prior_L < 0:
            raise ValueError(f"need sigma > 0 and L >= 0, got sigma={prior_sigma}, "
                             f"L={prior_L}")
        m_alpha_d = m_alpha_s = None
        m_c_d = 1.0 / prior_sigma ** 2
        m_c_s = prior_L ** 2 / prior_sigma ** 2
    else:
        m_alpha_d = alpha_d if metric_alpha_d is None else metric_alpha_d
        m_alpha_s = alpha_s if metric_alpha_s is None else metric_alpha_s
        m_c_d = m_alpha_d * norm_obs / norm_damping
        m_c_s = m_alpha_s * norm_obs / norm_smoothing
    if inner_product == "regularisation" and (m_c_d <= 0 or m_c_s < 0):
        # The mass term is what makes the form positive definite. The stiffness
        # matrix alone has the constants in its null space, because the form carries
        # no boundary condition, and a direct solver factorises it without
        # complaining, so the run would go on and give a wrong Riesz map.
        raise ValueError(
            f"the regularisation inner product needs a positive mass weight and a "
            f"non-negative stiffness weight, and got metric_alpha_d={m_alpha_d} "
            f"(mass) and metric_alpha_s={m_alpha_s} (stiffness). With alpha_d at "
            f"zero, pass metric_alpha_d explicitly.")
    # ---- Walls ---------------------------------------------------------------------
    # Every object here is built with annotation stopped, so that the tape is the
    # same for every wall setting: a boundary condition with a constant value can
    # add a block of its own when it is built during annotation.
    if wall_treatment not in ("free", "pinned"):
        raise ValueError(f"unknown wall_treatment {wall_treatment!r}")
    if prior_wall_bc not in ("none", "dirichlet"):
        raise ValueError(f"unknown prior_wall_bc {prior_wall_bc!r}")
    if prior_wall_bc == "dirichlet" and inner_product != "regularisation":
        raise ValueError("prior_wall_bc='dirichlet' needs the 'regularisation' "
                         "inner product")
    if prior_wall_bc == "dirichlet" and wall_treatment != "pinned":
        # The Dirichlet operator maps every wall entry of a dual vector to zero, so
        # as a preconditioner it is singular on the wall nodes. That is harmless only
        # when the wall nodes never enter the Krylov system, which is the case when
        # they are pinned and BNK holds them in its fixed set. With free walls the
        # Krylov solve would see a preconditioner with a null space.
        raise ValueError("prior_wall_bc='dirichlet' needs wall_treatment='pinned': "
                         "with free walls the Dirichlet operator is a singular "
                         "preconditioner on the wall nodes")
    with stop_annotating():
        # The wall values of the temperature, the same as ``temp_bcs`` and T0_bcs.
        wall_value_bcs = [DirichletBC(Q1, 1.0, boundary.bottom),
                          DirichletBC(Q1, 0.0, boundary.top)]
        n_owned = Q1.dof_dset.size
        wall_nodes = {name: np.sort(bc.nodes[bc.nodes < n_owned])
                      for name, bc in zip(("bottom", "top"), wall_value_bcs)}
        wall_weights = {}
        for name, bc in zip(("bottom", "top"), wall_value_bcs):
            weight = Function(Q1, name=f"Wall_Weight_{name}")
            DirichletBC(Q1, 1.0, bc.sub_domain).apply(weight)
            wall_weights[name] = weight
        # The prior mean of the campaign: the layer average with the wall values.
        m0 = Function(Q1, name="Prior_Mean").assign(Taverage)
        for bc in wall_value_bcs:
            bc.apply(m0)
        operator_bcs = None
        if prior_wall_bc == "dirichlet":
            operator_bcs = [DirichletBC(Q1, 0.0, boundary.bottom),
                            DirichletBC(Q1, 0.0, boundary.top)]
    ip = make_inner_product(Q1, inner_product, c_d=2 * m_c_d, c_s=2 * m_c_s,
                            bcs=operator_bcs)

    # The initial guess is data, not part of the model, so it is set without
    # annotation. The Control is the first tape entry that involves ``Tic``.
    Tic = Function(Q1, name="Initial_Condition_Temperature")
    with stop_annotating():
        if initial_guess == "average":
            Tic.assign(Taverage)
        elif initial_guess == "observed":
            # The first guess of the paper: the observed final temperature used as
            # the initial condition. It is far from the truth and it is available in
            # a real inversion, which is why the paper chooses it.
            Tic.project(Tobs)
        elif initial_guess == "true":
            Tic.project(Tic_true)
        else:
            raise ValueError(f"unknown initial_guess {initial_guess!r}")
        guess_walls = initial_guess_walls or wall_treatment == "pinned"
        if guess_walls:
            # The projection of Tobs misses the wall values slightly (0.9977 on the
            # bottom row against 1), so the wall nodes are overwritten with them.
            for bc in wall_value_bcs:
                bc.apply(Tic)
    control = Control(Tic, riesz_map=ip.riesz_map)

    # T0 is the control with the strong boundary values imposed, in the control
    # space. The regularisation acts on T0, so the boundary rows carry no penalty.
    # Only the regularisation uses it, so it is left off the tape when neither term
    # is in the objective.
    T0 = None
    if alpha_d > 0 or alpha_s > 0:
        T0_bcs = [DirichletBC(Q1, 0.0, boundary.top), DirichletBC(Q1, 1.0, boundary.bottom)]
        T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)
    else:
        # ``InverseProblem.prior_term`` repeats the projection off the tape, so the
        # boundary conditions must exist even when T0 does not. They are built with
        # annotation stopped, so that a run without regularisation keeps exactly the
        # tape it had before: a boundary condition with a constant value can add a
        # block of its own when it is built during annotation.
        with stop_annotating():
            T0_bcs = [DirichletBC(Q1, 0.0, boundary.top),
                      DirichletBC(Q1, 1.0, boundary.bottom)]

    # ---- Forward model on the tape ------------------------------------------------
    T.project(Tic, bcs=energy_solver.strong_bcs)
    for _ in tape.timestepper(iter(range(initial_timestep, steps))):
        stokes_solver.solve()
        energy_solver.solve()

    # ---- Objective -----------------------------------------------------------------
    # Each regularisation term is left off the tape when its weight is zero, so that
    # a run without regularisation has the misfit alone and carries no dead blocks.
    t_misfit = assemble((T - Tobs) ** 2 * _dx)
    objective = t_misfit
    if alpha_d > 0:
        damping = assemble((T0 - Taverage) ** 2 * _dx)
        objective = objective + c_d * damping
    if alpha_s > 0:
        smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * _dx)
        objective = objective + c_s * smoothing

    # One garbage collection per Hessian-vector product.
    #
    # A Hessian-vector product leaves garbage that the reference counting of CPython
    # does not free, because the objects are in reference cycles, so the memory of a
    # run grows by about 0.046 GB per product until the automatic collector happens to
    # run. Measured in a separate session, recorded in NOTES/team/hessian-memory/.
    #
    # The collection has to happen once per product. Once after taping does nothing,
    # because the garbage is made by the product and not by the tape. Once per block
    # would be 68 blocks times two sweeps of a full pass over every tracked object, for
    # nothing: current memory equals peak in every run, so there is no transient inside
    # a product for a collection in the middle of a sweep to catch.
    #
    # pyadjoint already calls a callback at the end of every ``hessian()``
    # (``reduced_functional.py:288``), so the collection goes there and every optimiser
    # gets it with nothing to remember at the call site. The callback is given the
    # functional value, the Hessian action and the control values, and needs none of
    # them.
    #
    # Upstream this belongs in ``compute_hessian`` in ``pyadjoint/drivers.py``, after
    # the reverse sweep. pyadjoint has accepted the idea elsewhere: ``CheckpointManager``
    # takes ``gc_timestep_frequency`` and calls ``gc.collect(generation)`` inside the
    # sweep for this same reason. That machinery runs only when a checkpoint schedule
    # is enabled, and a Hessian-vector product cannot use one, because pyadjoint has no
    # checkpoint-manager path for the tangent-linear and second-order adjoint sweeps.
    # So the case that needs it most is the one case the existing code cannot reach.
    def collect_after_hessian(*args):
        """Collect garbage at the end of one Hessian-vector product."""
        gc.collect()

    reduced_functional = ReducedFunctional(objective, control,
                                           hessian_cb_post=collect_after_hessian)

    # From here on the tape is only replayed. Every optimiser must run under
    # ``stop_annotating``; this pause is the guard for anything between.
    pause_annotation()

    # Take the tape out of every later collection. The tape lives for the whole run and
    # is never garbage, so scanning it once per Hessian-vector product is wasted work
    # that grows with the window: at window 80 the tape is about 1300 blocks and 1559
    # block variables, and freezing moves some 590000 objects out of the scan. This
    # changes no result on its own; it keeps the collection above cheap.
    gc.collect()
    gc.freeze()

    T_lb = Function(Q1, name="Lower_Bound_Temperature").assign(0.0)
    T_ub = Function(Q1, name="Upper_Bound_Temperature").assign(1.0)
    if wall_treatment == "pinned":
        # Equal bounds on the wall nodes: BNK's fixed set. The tape is untouched,
        # so the Hessian harness in tests/hessian/ works as before.
        with stop_annotating():
            for bc in wall_value_bcs:
                bc.apply(T_lb)
                bc.apply(T_ub)

    return InverseProblem(
        reduced_functional=reduced_functional,
        control=control,
        Tic=Tic,
        Tic_true=Tic_true,
        Tobs=Tobs,
        T=T,
        bounds=(T_lb, T_ub),
        inner_product=ip,
        weights={"alpha_d": alpha_d, "alpha_s": alpha_s, "c_d": c_d, "c_s": c_s,
                 "metric_alpha_d": m_alpha_d, "metric_alpha_s": m_alpha_s,
                 "metric_c_d": m_c_d, "metric_c_s": m_c_s,
                 "prior_sigma": prior_sigma, "prior_L": prior_L},
        window=window,
        mesh=mesh,
        norm_obs=float(norm_obs),
        norm_ic=float(assemble(Tic_true ** 2 * _dx)),
        delta_t=delta_t_value,
        window_time=window * delta_t_value,
        Taverage=Taverage,
        T0_bcs=T0_bcs,
        m0=m0,
        wall_nodes=wall_nodes,
        wall_weights=wall_weights,
        wall_settings={"bounds": wall_treatment, "operator_bc": prior_wall_bc,
                       "first_guess_walls": bool(guess_walls),
                       "wall_nodes_bottom": int(len(wall_nodes["bottom"])),
                       "wall_nodes_top": int(len(wall_nodes["top"]))},
    )
