"""The step of the Quadrature history and its directly assembled first derivatives.

The mechanics/sea-level residual remains owned by SelfGravitatingGIASolver.
Only the material update and its implicit scalar sensitivity are supplied here.
No external operator is embedded in a form or used by the numerical root solver.

`DirectConstitutiveBackend` solves one step (the coupled Newton solve with the
local constitutive solve at the quadrature points in its callbacks) and
supplies the adjoint and the tangent of that map. `ConstitutiveStepBlock`
records the step on the tape; it shares the binding of the solver state and
the derivative of the low-rank DtN prefactor with the low-rank step block
(`gadopt.gia_step_adjoint.GIAStepBlockMixin`).
"""
import numpy as np
import firedrake as fd
import ufl
from mpi4py import MPI
from pyadjoint import Block
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from ufl.algorithms.analysis import extract_coefficients

from .internal_variable_equation import assign_history_slices, history_slices
from .gia_step_adjoint import GIAStepBlockMixin
from .utility import weak_callback

#: The index of the displacement in the mixed space of both GIA solvers that
#: own a quadrature step; the field order of either solver starts with it.
DISPLACEMENT = 0


class MaxwellQuadratureUpdate:
    """One scalar common-fluidity root per mechanical quadrature point."""

    def _initialise(self, owner):
        self.owner = owner
        self.u = fd.Function(owner.solution_space.sub(DISPLACEMENT))
        self.histories = owner.time_integrator
        element = self.histories.current[0].function_space().ufl_element()
        Q = fd.FunctionSpace(owner.mesh, element.sub_elements[0])
        self.q = fd.Function(Q, name="constitutive_fluidity")
        self.E0 = fd.Function(Q, name="constitutive_residual")
        self.Eq = fd.Function(Q, name="constitutive_derivative").assign(1)
        self.tolerance = 1e-12
        self.max_iterations = 50

    def __init__(self, owner):
        self._initialise(owner)
        material = owner.approximation
        history = history_slices(self.histories.combinations())
        alpha, h = self.histories.alpha, owner.dt
        strain = material.deviatoric_strain(self.u)
        q = ufl.variable(self.q)
        self.updates = [(old+h*q/tau*strain)/(alpha+h*q/tau)
                        for old, tau in zip(history, material.maxwell_times)]
        stress = material.deviatoric_stress(self.u, self.updates)
        E = q - 1/material.power_law_factor(stress)
        self.equation = ufl.replace(E, {q: self.q})
        self.diagonal = ufl.replace(ufl.algorithms.expand_derivatives(ufl.diff(E, q)), {q: self.q})
        self.updates = [ufl.replace(g, {q: self.q}) for g in self.updates]
        self.lower = fd.conditional(fd.eq(material.exponent, 1), 1., 1/(
            1+(material.background_stress/material.transition_stress)**(material.exponent-1)))

    def on_mixed_state(self, expression, state=None):
        state = self.owner.solution if state is None else state
        return ufl.replace(expression, {self.u: fd.split(state)[DISPLACEMENT]})

    def refresh(self, state):
        """Solve with fixed accepted histories, using only mechanics-mesh fields."""
        self.u.assign(state.subfunctions[DISPLACEMENT])
        self.q.interpolate(self.lower)
        lower = self.q.dat.data_ro.copy()
        self.E0.interpolate(self.equation)
        self.q.dat.data[:] = np.maximum(lower, self.q.dat.data_ro-self.E0.dat.data_ro)
        for _ in range(self.max_iterations):
            self.E0.interpolate(self.equation)
            norm = np.max(np.abs(self.E0.dat.data_ro)/(1+self.q.dat.data_ro), initial=0.)
            norm = self.q.comm.allreduce(float(norm), op=MPI.MAX)
            if norm < self.tolerance:
                self.Eq.interpolate(self.diagonal)
                return
            self.Eq.interpolate(self.diagonal)
            current = self.q.dat.data_ro.copy()
            old_residual = np.abs(self.E0.dat.data_ro).copy()
            step = self.E0.dat.data_ro/self.Eq.dat.data_ro
            damping = np.ones_like(current)
            for _ in range(20):
                self.q.dat.data[:] = np.maximum(lower, current-damping*step)
                self.E0.interpolate(self.equation)
                good = (np.abs(self.E0.dat.data_ro) <= old_residual*(1-1e-4*damping)) | (
                    np.abs(self.E0.dat.data_ro) < self.tolerance*(1+self.q.dat.data_ro))
                if self.q.comm.allreduce(bool(good.all()), op=MPI.LAND):
                    break
                damping[~good] *= .5
        raise RuntimeError(f"Local constitutive root failed: relative residual {norm:.3g}")


class BackstressQuadratureUpdate(MaxwellQuadratureUpdate):
    """Eliminate creep strain and recovering backstress through one root."""

    def __init__(self, owner):
        self._initialise(owner)
        material = owner.approximation
        P, B = history_slices(self.histories.combinations())
        alpha, h = self.histories.alpha, owner.dt
        mu, mu_b, eta = material.mu0, material.backstress_modulus, material.viscosity[0]
        c = alpha+h/material.recovery_time
        trial = 2*mu*(material.deviatoric_strain(self.u)-P/alpha)-2*mu_b*B/c
        chi = ufl.variable(self.q)
        effective = trial/(1+h*chi/eta*(mu/alpha+mu_b/c))
        E = chi-material.creep_fluidity(effective)
        self.equation = ufl.replace(E, {chi: self.q})
        self.diagonal = ufl.replace(ufl.algorithms.expand_derivatives(ufl.diff(E, chi)), {chi: self.q})
        rate = chi*effective/(2*eta)
        self.updates = [ufl.replace(g, {chi: self.q})
                        for g in ((P+h*rate)/alpha, (B+h*rate)/c)]
        self.lower = 1.


class ConstitutiveStepBlock(GIAStepBlockMixin, Block):
    """One step of the Quadrature history: coupled solve and material update as one map.

    The map takes the coefficients of the step residual and of the local
    constitutive equation (`DirectConstitutiveBackend.inputs`: the previous
    history levels, the time coefficients, the material and coupling
    parameters, the loads) to the coupled solution and the updated history
    at the quadrature points (`DirectConstitutiveBackend.outputs`). The
    previous solution is not an input: it is only the initial guess of the
    forward Newton solve, and a replay starts from zero.

    The solution leaves the map with the gauge the solve gave it. The
    `KernelProjectionBlock` that `SelfGravitatingGIASolver.solve` records
    after this block removes the rigid rotation, as for the low-rank step
    block (`gadopt.gia_step_adjoint`). The history does not depend on the
    gauge: a rigid rotation has zero strain.

    Replay, adjoint and tangent run the backend with the shared solver state
    bound to this block (`GIAStepBlockMixin._bound_to_block`). The adjoint
    and the tangent add the derivative of the low-rank prefactor
    `theta_psi`, which the forms do not carry. The second-order adjoint is
    not written.

    Args:
      backend: The `DirectConstitutiveBackend` of the step.
    """

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.gia_solver = backend.owner
        # `SelfGravitatingGIASolver.solve` raised the index before the step,
        # so this is the index the forward solve's preconditioners saw.
        self.gia_solve_index = backend.owner.appctx.get("gia_solve_index")
        for coefficient in backend.inputs:
            self.add_dependency(coefficient)

    def __str__(self):
        return "ConstitutiveStepBlock"

    def _restore(self, inputs):
        """Set the private inputs of the backend to this block's input values."""
        for target, value in zip(self.backend.private_inputs, inputs):
            target.assign(value)

    def _restore_state(self, inputs):
        """Set the inputs and the forward solution, and refresh the local root.

        The derivative of the map is taken at the forward solution of this
        block, with the frozen quadrature coefficients of the local solve at
        that solution (`MaxwellQuadratureUpdate.refresh`).
        """
        self._restore(inputs)
        self.backend.work.assign(self.get_outputs()[0].saved_output)
        self.backend.material.refresh(self.backend.work)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        self._restore(inputs)
        # A shared workspace may hold an unrelated later step after reverse.
        # Use a deterministic gauge-free guess for this algebraic solve.
        self.backend.work.assign(0)
        with self._bound_to_block():
            self.backend.forward()

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        return self.backend.numeric_outputs[idx]._ad_create_checkpoint()

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        self._restore_state(inputs)
        with self._bound_to_block():
            self.backend.reverse(adj_inputs)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        gradient = self.backend.input_gradient(idx)
        return self._add_theta_adjoint_term(
            gradient, block_variable.output, self.backend.work, self.backend.adjoint)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        self._restore_state(inputs)
        coefficient = self._theta_tangent_coefficient()
        with self._bound_to_block():
            return self.backend.tangent(
                tlm_inputs, self._B0_action(self.backend.work) if coefficient else None,
                coefficient)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return prepared[idx]

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        raise NotImplementedError(
            "The second-order adjoint of the Quadrature history step is not "
            "written: it needs the second derivatives of the local "
            "constitutive solve.")


class DirectConstitutiveBackend:
    """Direct derivatives of the FULL mixed residual, not just displacement.

    Owns the step of the Quadrature history: the coupled nonlinear solve
    with the local constitutive solve in its callbacks, the history update at
    the quadrature points, and the first derivatives of that map (adjoint and
    tangent) for `ConstitutiveStepBlock`. The residual is rewritten in
    private copies of its coefficients, so that a replay or a derivative
    sweep can set them to the values of one block without touching the live
    Functions.

    Both DtN representations are supported with natural displacement
    boundaries. Strong and weak displacement boundary conditions are not
    implemented for this path and are refused explicitly.

    Args:
      owner: The `SelfGravitatingGIASolver`.
      material: The `MaxwellQuadratureUpdate` (or backstress variant) of
        the owner's history.
    """

    def __init__(self, owner, material):
        self.owner, self.material = owner, material
        material.owner = owner
        auxiliary = (owner.solution, material.q, material.E0, material.Eq)
        coefficients = set(owner.F.coefficients()) | set(extract_coefficients(material.equation))
        for g in material.updates:
            coefficients.update(extract_coefficients(g))
        coefficients.discard(material.u)
        self.inputs = tuple(sorted((c for c in coefficients if isinstance(c, fd.Function)
                                    and all(c is not a for a in auxiliary)), key=lambda c: c.count()))
        self.private_inputs = tuple(c.copy(deepcopy=True) for c in self.inputs)
        self.work = owner.solution.copy(deepcopy=True)
        mapping = dict(zip(self.inputs, self.private_inputs)) | {owner.solution: self.work}
        self.lowrank = owner.dtn_representation == "lowrank"
        self.F = ufl.replace(owner.F, mapping)
        material.equation = ufl.replace(material.equation, mapping)
        material.diagonal = ufl.replace(material.diagonal, mapping)
        material.updates = [ufl.replace(g, mapping) for g in material.updates]
        # E0 and Eq are frozen quadrature coefficients. At the refreshed root,
        # this expression has value q and derivative -E_z/Eq for any input z.
        lifted = material.q+(material.E0-material.on_mixed_state(material.equation, self.work))/material.Eq
        self.sensitivity_F = ufl.replace(self.F, {material.q: lifted})
        self.updates = [material.on_mixed_state(g, self.work) for g in material.updates]
        self.sensitivity_updates = [ufl.replace(g, {material.q: lifted}) for g in self.updates]
        self.J = fd.derivative(self.sensitivity_F, self.work)
        self.problem = fd.NonlinearVariationalProblem(self.F, self.work, bcs=owner.strong_bcs, J=self.J)
        self.build_solver()
        self.buffers = [fd.Function(m.function_space()) for m in owner.time_integrator.current]
        self.outputs = [owner.solution, *self.buffers]
        self.numeric_outputs = [self.work, *self.buffers]
        self.adjoint = fd.Function(owner.solution_space)
        # Each history Function may hold several Maxwell modes in one tensor.
        self.densities = [fd.Function(m.function_space()) for m in self.buffers]
        intersect = (fd.dx(domain=owner.potential_mesh),) if owner.mesh is not owner.potential_mesh else ()
        dx = fd.Measure('dx', domain=owner.mesh, intersect_measures=intersect,
                        metadata={'quadrature_degree': owner.quad_degree})
        densities = history_slices(self.densities)
        self.H = sum(fd.inner(g, s)*dx for g, s in zip(self.sensitivity_updates, densities))
        # Convert coefficient-space seeds to integration densities. Assembling
        # these weights also accounts for the symmetric tensor storage mapping.
        self.weights = []
        for m in self.buffers:
            ones = fd.as_tensor(np.ones(m.ufl_shape).tolist())
            self.weights.append(fd.assemble(fd.inner(ones, fd.TestFunction(m.function_space()))*dx))
        self.rhs_form = fd.derivative(self.H, self.work, fd.TestFunction(owner.solution_space))
        self.reverse_functional = self.H-fd.action(self.sensitivity_F, self.adjoint)
        self.gradient_forms = {}
        #: The tangent forms and Functions, built at the first tangent sweep
        #: (`_build_tangent`): only Taylor and tangent tests use them.
        self._tangent = None

    def build_solver(self):
        """Build the nonlinear solver of the step from the owner's options.

        Separate from the constructor, so that an owner can rebuild the
        solver after its options change without building a second backend.
        A second backend would replace the coefficients of the shared
        material expressions a second time, and the residual and the local
        Newton solve would then read two different sets of private copies.
        The cached derivative solvers read the same options, so they are
        dropped and built again at the next derivative sweep.
        """
        owner = self.owner
        # The callbacks hold the backend and the owner through weak
        # references. A strong reference would close a reference cycle
        # through C (the `_SNESContext` that keeps the callbacks is held by
        # the Python preconditioner contexts, which PETSc holds), and the
        # solver would never be collected (`gadopt.utility.weak_callback`).
        refresh = weak_callback(self.refresh_from_vector)
        self.solver = fd.NonlinearVariationalSolver(
            self.problem, solver_parameters=owner.solver_parameters,
            options_prefix=owner.name, appctx=owner.appctx, nullspace=owner.nullspace,
            transpose_nullspace=owner.transpose_nullspace, near_nullspace=owner.near_nullspace,
            pre_function_callback=refresh, pre_jacobian_callback=refresh,
            **(dict(post_function_callback=weak_callback(owner.augment_residual),
                    post_jacobian_callback=weak_callback(owner.augment_jacobian))
               if self.lowrank else {}))
        #: `{"adjoint": (matrix, solver), "tangent": (matrix, solver)}`, the
        #: linear derivative solves, built at their first use.
        self.derivative_solvers = {}

    def refresh_from_vector(self, vector):
        with stop_annotating():
            with self.work.dat.vec_wo as target:
                vector.copy(target)
            self.material.refresh(self.work)

    def forward(self):
        """Solve the step from the guess in `work` and update the history buffers.

        The solution keeps the gauge the solve gave it; the projection that
        removes the rigid rotation is the recorded one after the step block
        (`SelfGravitatingGIASolver.solve`).
        """
        self.owner.appctx["operator_version"] = None
        self.solver.solve()
        self.material.refresh(self.work)
        # Interpolate on the mechanics mesh only, not through mixed-domain split.
        assign_history_slices(self.buffers, self.material.updates)

    def solve(self):
        """One forward step from the owner's solution, recorded as one block."""
        block = ConstitutiveStepBlock(self) if annotate_tape() else None
        with stop_annotating():
            for target, value in zip(self.private_inputs, self.inputs):
                target.assign(value)
            self.work.assign(self.owner.solution)
            self.forward()
            self.owner.solution.assign(self.work)
        if block:
            get_working_tape().add_block(block)
            for output in self.outputs:
                block.add_output(output.create_block_variable())
            self.owner.adjoint_block = block

    def _linear_solve(self, kind, rhs, solution):
        """Solve one linear derivative system of the step, and check its residual.

        The operator is the Jacobian `J` of the step at the state in `work`
        (`kind="tangent"`) or its transpose (`kind="adjoint"`), matrix-free,
        with the low-rank DtN action on it on the low-rank path. The matrix
        and the solver are built at the first solve of each kind and
        reassembled in place afterwards.

        Args:
          kind: "adjoint" or "tangent".
          rhs: The right-hand side, a Cofunction; its rows of the strong
            boundary conditions are set to zero here.
          solution: The Function to write the solution into.

        Raises:
          RuntimeError: If the true relative residual is above 1e-8.
        """
        owner = self.owner
        owner.appctx["operator_version"] = None
        form = fd.adjoint(self.J) if kind == "adjoint" else self.J
        if kind not in self.derivative_solvers:
            matrix = fd.assemble(form, bcs=owner.strong_bcs, mat_type='matfree',
                                 appctx=owner.appctx)
            if self.lowrank:
                owner.augment_jacobian(None, matrix.petscmat)
            # The derivative options of the solver without its SNES part: no
            # accepted cap on the outer Krylov solve, so a capped derivative
            # solve raises (`StokesSolverBase.adjoint_solver_parameters`).
            options = {k: v for k, v in owner.adjoint_solver_parameters().items()
                       if not k.startswith('snes_')}
            # The transpose operator has the forward nullspaces swapped.
            nullspace, transpose_nullspace = owner.nullspace, owner.transpose_nullspace
            if kind == "adjoint":
                nullspace, transpose_nullspace = transpose_nullspace or nullspace, nullspace
            # A prefix of its own for each derivative solve, so that a
            # command-line option reaches one of them and not the forward
            # solve or the other derivative solve. The low-rank adopted
            # blocks use `<name>_lowrank_adj` and `<name>_lowrank_tlm` in the
            # same way (`gadopt.dtn_coupled_adjoint.derivative_solver_kwargs`).
            suffix = "adj" if kind == "adjoint" else "tlm"
            solver = fd.LinearSolver(
                matrix, solver_parameters=options, nullspace=nullspace,
                transpose_nullspace=transpose_nullspace, near_nullspace=owner.near_nullspace,
                appctx=owner.appctx, options_prefix=f"{owner.name}_quadrature_{suffix}")
            self.derivative_solvers[kind] = (matrix, solver)
        else:
            matrix, solver = self.derivative_solvers[kind]
            fd.assemble(form, bcs=owner.strong_bcs, tensor=matrix, mat_type='matfree',
                        appctx=owner.appctx)
            if self.lowrank:
                owner.augment_jacobian(None, matrix.petscmat)
            solver.invalidate_jacobian()
        for bc in owner.strong_bcs:
            bc.zero(rhs)
        solver.solve(solution, rhs)
        with solution.dat.vec_ro as x, rhs.dat.vec_ro as b:
            r = b.duplicate()
            matrix.petscmat.mult(x, r)
            r.aypx(-1, b)
            relative = r.norm()/max(b.norm(), 1e-30)
            r.destroy()
        if relative > 1e-8:
            raise RuntimeError(f"Direct {kind} solve true relative residual {relative}")

    def reverse(self, seeds):
        """Solve the adjoint system of the step for the seeds of its outputs.

        Args:
          seeds: The adjoint seeds of `outputs`: a Cofunction on the mixed
            space for the solution, then one Cofunction (or None) per history
            buffer.
        """
        for density, seed, weights in zip(self.densities, seeds[1:], self.weights):
            if seed is None:
                density.assign(0)
            else:
                density.dat.data[:] = seed.dat.data_ro/weights.dat.data_ro
        rhs = fd.assemble(self.rhs_form)
        if seeds[0] is not None:
            rhs += seeds[0]
        self._linear_solve("adjoint", rhs, self.adjoint)

    def input_gradient(self, index):
        """The gradient of the step map with respect to input `index`, from the forms.

        The derivative of the low-rank prefactor, which no form carries, is
        added by `ConstitutiveStepBlock`.
        """
        if index not in self.gradient_forms:
            coefficient = self.private_inputs[index]
            self.gradient_forms[index] = fd.derivative(
                self.reverse_functional, coefficient,
                fd.TestFunction(coefficient.function_space()))
        return fd.assemble(self.gradient_forms[index])

    def _build_tangent(self):
        """The forms of the tangent map, with one placeholder per input tangent.

        Built once, at the first tangent sweep. `directions[i]` holds the
        tangent of input `i` (zero where it has none), `state` the tangent
        of the mixed solution and `displacement` its displacement part on the
        mechanics mesh.

        The right-hand side is `(dF/dz) zdot` summed over the inputs that the
        residual reads. The history tangents are the derivatives of the
        history updates along `(displacement, zdot)`. Both are taken through
        the lifted fluidity `q + (E0 - E)/Eq`, whose value is the frozen root
        `q` and whose derivative is that of the local root, so they include
        the change of the local constitutive solve. The history expressions
        are written in the mechanics-mesh displacement `material.u`, as the
        forward update is, so that they interpolate on the mechanics mesh
        only.
        """
        material = self.material
        directions = [fd.Function(c.function_space()) for c in self.private_inputs]
        state = fd.Function(self.owner.solution_space)
        displacement = fd.Function(material.u.function_space())
        in_residual = set(self.sensitivity_F.coefficients())
        rhs_form = sum(fd.derivative(self.sensitivity_F, c, d)
                       for c, d in zip(self.private_inputs, directions) if c in in_residual)
        lifted = material.q + (material.E0 - material.equation)/material.Eq
        history = []
        for g in material.updates:
            g = ufl.replace(g, {material.q: lifted})
            read = set(extract_coefficients(g))
            tangent = ufl.derivative(g, material.u, displacement)
            for c, d in zip(self.private_inputs, directions):
                if c in read:
                    tangent = tangent + ufl.derivative(g, c, d)
            history.append(ufl.algorithms.expand_derivatives(tangent))
        buffers = [fd.Function(m.function_space()) for m in self.buffers]
        self._tangent = dict(directions=directions, state=state, displacement=displacement,
                             rhs_form=rhs_form, history=history, buffers=buffers)

    def tangent(self, tlm_inputs, B0_state, theta_coefficient):
        """The tangent of the step map along the input tangents.

        Solves `J xdot = -(dF/dz) zdot - (d theta_psi/dz zdot) B0 x` and
        evaluates the history tangents at `(xdot, zdot)`.

        Args:
          tlm_inputs: The tangent of each input, in the order of `inputs`,
            or None where the input has none.
          B0_state: `B0 x` at the forward solution (low-rank path with a
            tangent of a factor of `theta_psi`), or None.
          theta_coefficient: `(d theta_psi / dz) zdot`.

        Returns:
          New Functions, the tangent of each of `outputs` in order.
        """
        if self._tangent is None:
            self._build_tangent()
        t = self._tangent
        for direction, value in zip(t["directions"], tlm_inputs):
            if value is None:
                direction.assign(0)
            else:
                direction.assign(value)
        rhs = fd.assemble(-t["rhs_form"])
        if B0_state is not None:
            with rhs.dat.vec_rw as dv, B0_state.dat.vec_ro as bv:
                dv.axpy(-theta_coefficient, bv)
        self._linear_solve("tangent", rhs, t["state"])
        t["displacement"].assign(t["state"].subfunctions[DISPLACEMENT])
        assign_history_slices(t["buffers"], t["history"])
        return [t["state"].copy(deepcopy=True)] + [b.copy(deepcopy=True) for b in t["buffers"]]
