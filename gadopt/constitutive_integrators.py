"""Local quadrature integration and directly assembled first reverse derivatives.

The mechanics/sea-level residual remains owned by SelfGravitatingGIASolver.
Only the material update and its implicit scalar sensitivity are supplied here.
No external operator is embedded in a form or used by the numerical root solver.
"""
from contextlib import contextmanager

import numpy as np
import firedrake as fd
import ufl
from mpi4py import MPI
from pyadjoint import Block
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
from ufl.algorithms.analysis import extract_coefficients

from .internal_variable_equation import assign_history_slices, history_slices
from .nullspaces import apply_kernel_projection


class MaxwellQuadratureUpdate:
    """One scalar common-fluidity root per mechanical quadrature point."""

    def _initialise(self, owner):
        self.owner = owner
        self.u = fd.Function(owner.solution_space.sub(owner.layout.displacement))
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
        return ufl.replace(expression, {self.u: fd.split(state)[self.owner.layout.displacement]})

    def refresh(self, state):
        """Solve with fixed accepted histories, using only mechanics-mesh fields."""
        self.u.assign(state.subfunctions[self.owner.layout.displacement])
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
        self.equation = ufl.replace(E, {chi:self.q})
        self.diagonal = ufl.replace(ufl.algorithms.expand_derivatives(ufl.diff(E, chi)), {chi:self.q})
        rate = chi*effective/(2*eta)
        self.updates = [ufl.replace(g, {chi:self.q})
                        for g in ((P+h*rate)/alpha, (B+h*rate)/c)]
        self.lower = 1.


class ConstitutiveStepBlock(Block):
    """Converged mixed solve and simultaneous material updates as one map."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        for coefficient in backend.inputs:
            self.add_dependency(coefficient)

    def restore(self, inputs):
        for target, value in zip(self.backend.private_inputs, inputs):
            target.assign(value)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        self.restore(inputs)
        # A shared workspace may hold an unrelated later step after reverse.
        # Use a deterministic gauge-free guess for this algebraic solve.
        self.backend.work.assign(0)
        self.backend.forward()

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        return self.backend.numeric_outputs[idx]._ad_create_checkpoint()

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        self.restore(inputs)
        self.backend.work.assign(self.get_outputs()[0].saved_output)
        self.backend.material.refresh(self.backend.work)
        self.backend.reverse(adj_inputs)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return self.backend.input_gradient(idx)


class DirectConstitutiveBackend:
    """Direct derivatives of the FULL mixed residual, not just displacement.

    Both DtN representations are supported with natural displacement boundaries.
    Strong and weak displacement boundary
    conditions are not implemented for this path and are refused explicitly.
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
        self._mapping = mapping
        self._solve_epoch = 0
        self._theta_factors = (owner.scaling_factor, owner._row_scale_B_mu, owner.Lambda)
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
        intersect = (fd.dx(domain=owner.potential_mesh),) if owner.layout.cross_mesh else ()
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
        self.matrix = self.linear_solver = None

    def build_solver(self):
        """Build the nonlinear solver of the step from the owner's options.

        Separate from the constructor, so that an owner can rebuild the
        solver after its options change without building a second backend.
        A second backend would replace the coefficients of the shared
        material expressions a second time, and the residual and the local
        Newton solve would then read two different sets of private copies.
        The cached transpose solver of `reverse` reads the same options, so
        it is dropped and built again at the next reverse sweep.
        """
        owner = self.owner
        self.solver = fd.NonlinearVariationalSolver(self.problem, solver_parameters=owner.solver_parameters,
            options_prefix=owner.name, appctx=owner.appctx, nullspace=owner.nullspace,
            transpose_nullspace=owner.transpose_nullspace, near_nullspace=owner.near_nullspace,
            pre_function_callback=self.refresh_from_vector, pre_jacobian_callback=self.refresh_from_vector,
            **(dict(post_function_callback=owner.augment_residual,
                    post_jacobian_callback=owner.augment_jacobian) if self.lowrank else {}))
        self.matrix = self.linear_solver = None

    def _theta_values(self):
        from .gia_gravity import scalar_value
        return tuple(scalar_value(self._mapping.get(c, c)) for c in self._theta_factors)

    @contextmanager
    def bound_lowrank(self):
        """Bind the operator and PC cache to this step's private coefficients."""
        if not self.lowrank:
            yield
            return
        from .gia_gravity import scalar_value
        op, ctx = self.owner.dtn_operator, self.owner.appctx
        saved_theta = op.theta
        sf, bmu, lam = self._theta_values()
        op.theta = sf*bmu/lam
        missing = object()
        self._solve_epoch += 1
        values = {"gia_dt": scalar_value(self._mapping.get(self.owner.dt, self.owner.dt)),
                  "gia_time_alpha": scalar_value(self._mapping[self.owner.time_integrator.alpha]),
                  # A replay can change controls at the same h/alpha. Its
                  # private operator is a fresh solve even when the live
                  # forward counter has not moved since the original march.
                  "gia_solve_index": self._solve_epoch}
        saved = {key: ctx.get(key, missing) for key in values}
        ctx.update(values)
        try:
            yield
        finally:
            op.theta = saved_theta
            for key, value in saved.items():
                if value is missing:
                    ctx.pop(key, None)
                else:
                    ctx[key] = value

    def refresh_from_vector(self, vector):
        with stop_annotating():
            with self.work.dat.vec_wo as target:
                vector.copy(target)
            self.material.refresh(self.work)

    def forward(self):
        self.owner.appctx["operator_version"] = None
        with self.bound_lowrank():
            self.solver.solve()
        # Projection is part of this recorded map: replay must return the same
        # gauge as the original forward solve, not an arbitrary rigid rotation.
        # `self.work` is private to this block and never on the tape, so the
        # unrecorded form of the projection is the right one here.
        apply_kernel_projection(self.owner.nullspace, self.work)
        self.material.refresh(self.work)
        # Interpolate on the mechanics mesh only, not through mixed-domain split.
        assign_history_slices(self.buffers, self.material.updates)

    def solve(self):
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

    def reverse(self, seeds):
        with self.bound_lowrank():
            self._reverse(seeds)

    def _reverse(self, seeds):
        self.owner.appctx["operator_version"] = None
        for density, seed, weights in zip(self.densities, seeds[1:], self.weights):
            if seed is None:
                density.assign(0)
            else:
                density.dat.data[:] = seed.dat.data_ro/weights.dat.data_ro
        rhs = fd.assemble(self.rhs_form)
        if seeds[0] is not None:
            rhs += seeds[0]
        # The forward result is projected onto the complement of the rigid
        # kernel. Apply the transpose of that Euclidean projection to its seed.
        # `rhs` is a Cofunction, and the projection is symmetric in
        # coefficient space, so the same function applies the transpose.
        apply_kernel_projection(self.owner.nullspace, rhs)
        if self.matrix is None:
            self.matrix = fd.assemble(fd.adjoint(self.J), bcs=self.owner.strong_bcs, mat_type='matfree',
                                      appctx=self.owner.appctx)
            if self.lowrank:
                self.owner.augment_jacobian(None, self.matrix.petscmat)
            # The derivative options of the solver without its SNES part: no
            # accepted cap on the outer Krylov solve, so a capped transpose
            # solve raises (`StokesSolverBase.adjoint_solver_parameters`).
            options = {k: v for k, v in self.owner.adjoint_solver_parameters().items()
                       if not k.startswith('snes_')}
            self.linear_solver = fd.LinearSolver(self.matrix, solver_parameters=options,
                nullspace=self.owner.transpose_nullspace or self.owner.nullspace,
                transpose_nullspace=self.owner.nullspace, near_nullspace=self.owner.near_nullspace,
                appctx=self.owner.appctx)
        else:
            fd.assemble(fd.adjoint(self.J), bcs=self.owner.strong_bcs, tensor=self.matrix, mat_type='matfree',
                        appctx=self.owner.appctx)
            if self.lowrank:
                self.owner.augment_jacobian(None, self.matrix.petscmat)
            self.linear_solver.invalidate_jacobian()
        for bc in self.owner.strong_bcs:
            bc.zero(rhs)
        self.linear_solver.solve(self.adjoint, rhs)
        with self.adjoint.dat.vec_ro as x, rhs.dat.vec_ro as b:
            r = b.duplicate()
            self.matrix.petscmat.mult(x, r)
            r.aypx(-1, b)
            relative = r.norm()/max(b.norm(), 1e-30)
            r.destroy()
        if relative > 1e-8:
            raise RuntimeError(f"Direct transpose solve true relative residual {relative}")

    def input_gradient(self, index):
        if index not in self.gradient_forms:
            coefficient = self.private_inputs[index]
            self.gradient_forms[index] = fd.derivative(self.reverse_functional, coefficient,
                fd.TestFunction(coefficient.function_space()))
        gradient = fd.assemble(self.gradient_forms[index])
        if self.lowrank:
            # The factored DtN term is outside UFL: add its scalar scaling
            # derivative explicitly. Fixed geometry/mode rows have no inputs.
            original = self.inputs[index]
            sf, bmu, lam = self._theta_values()
            dtheta = sum(value for factor, value in zip(
                self._theta_factors, (bmu/lam, sf/lam, -sf*bmu/lam**2)) if original is factor)
            if dtheta:
                if original.ufl_shape or original.function_space().ufl_element().family() != "Real":
                    raise ValueError("DtN row-scaling controls must be scalar Real Functions")
                with self.work.dat.vec_ro as x, self.adjoint.dat.vec_ro as adjoint:
                    b = x.duplicate()
                    b.set(0.)
                    self.owner.dtn_operator.apply_local(x.array_r, b.array_w, theta=1.)
                    pairing = adjoint.dot(b)
                    b.destroy()
                gradient.dat.data[:] -= dtheta*pairing
        return gradient
