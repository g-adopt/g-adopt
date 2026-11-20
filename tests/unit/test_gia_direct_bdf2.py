"""Direct elimination in the complete two-mesh GIA/sea-level mixed system.

The Quadrature family of `SelfGravitatingGIASolver` eliminates the history
at the quadrature points by a local Newton solve and condenses the Jacobian.
These tests compare it with an unreduced reference that keeps the history
as an unknown on the same quadrature points and solves the full mixed system
by Newton with a Jacobian assembled here, block by block.
"""
from unittest import mock

import gadopt
import firedrake as fd
import numpy as np
import ufl
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               pause_annotation, get_working_tape, stop_annotating)
from pyadjoint.tape import annotate_tape
import gadopt.gia_history as gia_history
from gadopt.gia_couplings import CentreOfMass, FluidCore, Rotation, SeaLevel
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.gia_solver import SelfGravitatingGIASolver
import test_gia_gravity as base
from test_gravity_adjoint import assert_taylor_with_guards
import gia_cases

# Every test in this module runs a solve or a time step, so the whole
# module is a `longtest`, and PR CI (`-m 'not longtest'`) leaves it out.
pytestmark = pytest.mark.longtest


@pytest.fixture(scope='module')
def meshes():
    """The small triangle annulus, read from `tests/unit/data` (no gmsh needed)."""
    return gia_cases.annulus_meshes()


STEPS = (.08, .13, .07, .12)


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


#: The quadrature degree of the history and of the volume integrals, on both
#: the Quadrature-family solver and its reference.
QUAD_DEGREE = 4


def quadrature_history_space(mesh, n_maxwell, degree=1, symmetric=True, family="DG"):
    """The space of `internal_variable_space`, always on the degree-4 Quadrature element.

    Patched into `gadopt.gia_history` while the reference solver is built,
    so that its DG-family history unknown (field 1 of the mixed space, with
    the weak history equation of the approximation) lives at the same
    quadrature points as the eliminated history of the Quadrature family.
    The history equation tested with the Quadrature basis at a degree-4 rule
    is then the pointwise constitutive equation at every point, which is the
    equation the Quadrature family eliminates.
    """
    from gadopt.internal_variable_equation import internal_variable_space
    return internal_variable_space(mesh, n_maxwell, QUAD_DEGREE, symmetric,
                                   family="Quadrature")


def build(meshes, control, *, direct, exponent=3, scheme='bdf2', iterative=False, model='maxwell',
          representation='multiplier', gravity_control=None, moving_coastline=False,
          options_extra=None, timestep=None, initial_sea_level=None):
    """The fluid-core, rotation, centre-of-mass and sea-level case on the triangle annulus.

    Args:
      meshes: the module fixture (parent annulus, mantle submesh).
      control: a field that scales the viscosity by `exp(control)`.
      direct: True builds the Quadrature family (the history eliminated by
        the local Newton solve). False builds the unreduced reference: the
        history is an unknown of the mixed space on the same quadrature
        points, and the step is a Newton solve with the unreduced mixed
        Jacobian assembled here (`ReferenceNewton`).
      exponent: the power-law exponent, or "spatial" for a DG0 exponent of 3
        on one half and 1 on the other.
      scheme: "bdf2" or "backward_euler".
      iterative: the iterative preset instead of the direct one.
      model: "maxwell" (two elements) or "backstress".
      representation: the DtN representation.
      gravity_control: a `B_mu` value (a `Real` Function for a control).
      moving_coastline: the live coastline instead of a fixed ocean.
      options_extra: options added last.
      timestep: a `Real` time step to use instead of a new one.
      initial_sea_level: `SL_init` to use instead of `0.005 cos(phi)`.

    Returns:
      `(solver, dt, ice)`.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, 'CG', 2))
    psi = fd.Function(fd.FunctionSpace(parent, 'CG', 2))
    dt = (fd.Function(fd.FunctionSpace(sub, 'R', 0)).assign(STEPS[0])
          if timestep is None else timestep)
    Q = fd.FunctionSpace(sub, 'CG', 2)
    theta = fd.atan2(fd.SpatialCoordinate(sub)[1], fd.SpatialCoordinate(sub)[0])
    initial_sea = (fd.Function(Q).interpolate(.005*fd.cos(theta))
                   if initial_sea_level is None else initial_sea_level)
    initial_ice = fd.Function(Q)
    ice = fd.Function(Q)
    power = exponent
    if exponent == 'spatial':
        power = fd.Function(fd.FunctionSpace(sub, 'DG', 0)).interpolate(
            fd.conditional(fd.SpatialCoordinate(sub)[0] > 0, 3., 1.))
    material = base.approximation(
        shear_modulus=[.4, .6],
        viscosity=[.12*fd.exp(control), .6*fd.exp(control)],
        exponent=power, transition_stress=.003,
        B_mu=base.B_MU if gravity_control is None else gravity_control)
    if model == 'backstress':
        material = gadopt.BackstressApproximation(
            bulk_modulus=1., density=1., shear_modulus=1., viscosity=.3*fd.exp(control),
            backstress_modulus=.8, recovery_time=.4*fd.exp(.3*control),
            exponent=exponent, transition_stress=.003, g=1., B_mu=base.B_MU,
            self_gravity_number=base.LAMBDA)
    sea = SeaLevel(
        boundary=base.CURVE_RE, rho_w=.1, rho_i=.09, g_surface=1.,
        SL_init=initial_sea, I=ice, I_init=initial_ice,
        N_init=fd.Constant(0.), ur_init=fd.Constant(0.), fixed_ocean=not moving_coastline, softplus_width=.001)
    X = fd.SpatialCoordinate(sub)
    measure = fd.Measure('dx', domain=sub, intersect_measures=(fd.dx(domain=parent),))
    couplings = [FluidCore(boundary=base.CURVE_RC, rho_core=1.2, g=1.),
                 Rotation(C=fd.assemble(fd.dot(X, X)*measure), Omega_sq=OMEGA_SQ_EARTH),
                 CentreOfMass(), sea]
    options = dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11,
                   ksp_error_if_not_converged=True)
    if options_extra:
        options.update(options_extra)
    arguments = dict(
        dt=dt, time_scheme=scheme, gravity_bcs=base.gravity_bcs(parent, sheet=False),
        bcs={}, couplings=couplings, dtn_representation=representation,
        nullspace='rigid_rotation', quad_degree=QUAD_DEGREE,
        solver_parameters='iterative' if iterative else 'direct',
        solver_parameters_extra=options)
    if direct:
        solver = SelfGravitatingGIASolver(
            u, psi, material, internal_variable_family='Quadrature', **arguments)
    else:
        # The DG family on the Quadrature element (`quadrature_history_space`):
        # the same discrete equations as the Quadrature family, with the
        # history kept as an unknown instead of eliminated.
        #
        # The construction check of the declared rotation mode
        # (`_check_kernel`) is skipped here. It assembles the Jacobian of the
        # solver's own residual, which TSFC cannot compile with a Quadrature
        # field inside the mixed coefficient (the reason `ReferenceNewton`
        # below differentiates a copy with separate fields). The same mode on
        # the same mesh, boundaries and couplings is checked when the
        # Quadrature-family solver of the same case is built.
        with mock.patch.object(gia_history, 'internal_variable_space',
                               quadrature_history_space), \
                mock.patch.object(SelfGravitatingGIASolver, '_check_kernel',
                                  lambda self, basis: None):
            solver = SelfGravitatingGIASolver(
                u, psi, material, internal_variable_family='DG', **arguments)
    state = solver.solution
    Z = state.function_space()
    if not direct:
        # TSFC attempts to differentiate the entire mixed coefficient when
        # transferring grad(u) across meshes, including quadrature histories.
        # Independent coefficient fields avoid requesting spatial derivatives
        # of quadrature data. Differentiate the original, unreduced equations
        # jointly with respect to those fields to construct the mixed Jacobian.
        fields = [fd.Function(f.function_space()) for f in state.subfunctions]
        flat = [f[index] if index else f for f in fields for index in np.ndindex(f.ufl_shape)]
        F = ufl.replace(ufl.algorithms.expand_derivatives(solver.F), {state: fd.as_vector(flat)})
        F = ufl.algorithms.expand_derivatives(F)
        # Expanded here and not left to the assembler: Firedrake splits the
        # mixed form into blocks before the form compiler expands
        # derivatives, and its splitter handles a derivative with respect to
        # one coefficient only (a tuple raises `ValueError: too many values
        # to unpack` in `formmanipulation.py`).
        J = ufl.algorithms.expand_derivatives(
            ufl.derivative(F, tuple(fields), fd.TrialFunctions(Z)))

        def jacobian_matrix():
            """The unreduced mixed Jacobian at the present state, as one AIJ matrix.

            Assembled block by block. A block between two non-`Real` fields
            is assembled as a matrix. A block with a `Real` row or column is
            built from actions instead, because the pyop3 Firedrake cannot
            assemble a matrix with a `Real` row or column on an interior facet
            of the parent ("Currently cannot deal with non-unit (vector-valued)
            columns"), and the sea-level and fluid-core terms put such blocks
            on those facets. A `Real` space has one basis function, the
            constant 1, so the action of a block on the unit `Real` function
            is its single column, and the action of the adjoint block is its
            single row: the entries are those of the assembled block.

            Replace the actions by direct assembly of every block once
            firedrakeproject/firedrake#5665 ("Cannot assemble a 2-form with a
            Real argument on interior facets (pyop3)") is fixed.
            """
            from firedrake.formmanipulation import ExtractSubBlock
            from firedrake.petsc import PETSc
            assert state.comm.size == 1
            splitter = ExtractSubBlock()
            real = [Z.sub(i).ufl_element().family() == 'Real' for i in range(len(Z))]
            blocks, owned = [], []
            for row in range(len(Z)):
                blocks.append([])
                for col in range(len(Z)):
                    form = splitter.split(J, (row, col))
                    if isinstance(form, ufl.ZeroBaseForm) or form.empty():
                        blocks[-1].append(None)
                        continue
                    if not (real[row] or real[col]):
                        block = fd.assemble(form, mat_type='aij').petscmat
                        blocks[-1].append(block)
                        continue
                    if real[col]:
                        # The column: the action on the unit Real function.
                        unit = fd.Function(Z.sub(col)).assign(1.0)
                        values = fd.assemble(fd.action(form, unit)).dat.data_ro.reshape(-1, 1)
                    else:
                        # The row: the action of the adjoint block.
                        unit = fd.Function(Z.sub(row)).assign(1.0)
                        values = fd.assemble(fd.action(fd.adjoint(form), unit)).dat.data_ro.reshape(1, -1)
                    block = PETSc.Mat().createDense(values.shape, array=values.copy(),
                                                    comm=state.comm).convert('aij')
                    owned.append(block)
                    blocks[-1].append(block)
            nest = PETSc.Mat().createNest(blocks, isrows=Z.field_ises,
                                          iscols=Z.field_ises, comm=state.comm)
            nest.assemble()
            # `convert` without an output matrix converts in place, so the
            # returned matrix is `nest` itself, now of type AIJ.
            matrix = nest.convert('aij')
            for block in owned:
                block.destroy()
            return matrix

        class ReferenceNewton:
            # Newton with the unreduced mixed Jacobian, assembled block by
            # block (`jacobian_matrix`) and factored with MUMPS. This bypasses
            # mixed-coefficient splitting and the prohibition on direct
            # monolithic assembly with Real fields.
            def __init__(self):
                self.snes = self
                self.iterations = 0

            def getLinearSolveIterations(self):
                return self.iterations

            def getIterationNumber(self):
                return self.iterations

            def solve(self, **kwargs):
                from firedrake.petsc import PETSc
                for iteration in range(25):
                    for target, source in zip(fields, state.subfunctions):
                        target.assign(source)
                    residual = fd.assemble(F)
                    with residual.dat.vec_ro as r:
                        norm = r.norm()
                    if norm < 1e-13:
                        self.iterations = iteration
                        return
                    A = jacobian_matrix()
                    ksp = PETSc.KSP().create(comm=state.comm)
                    ksp.setOperators(A)
                    ksp.setType('preonly')
                    ksp.getPC().setType('lu')
                    ksp.getPC().setFactorSolverType('mumps')
                    with residual.dat.vec_ro as r, state.dat.vec_rw as x:
                        delta = x.duplicate()
                        ksp.solve(r, delta)
                        assert ksp.getConvergedReason() > 0
                        x.axpy(-1, delta)
                        delta.destroy()
                    ksp.destroy()
                    A.destroy()
                raise RuntimeError(f'Reference Newton failed, residual={norm}')
        # The step solves with the reference Newton instead of Firedrake's
        # variational solver; the rest of `solve` (time coefficients, monopole
        # datum, kernel projection, history accept) is the solver's own.
        solver._step.solver = ReferenceNewton()
    return solver, dt, ice


@pytest.mark.parametrize('exponent', [1, 3, 'spatial'])
@pytest.mark.parametrize('model', ['maxwell', 'backstress'])
def test_direct_matches_coupled_quadrature(meshes, exponent, model):
    if model == 'backstress' and exponent == 'spatial':
        pytest.skip('Backstress exponent is a fixed scalar in this model')
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1))
    X = fd.SpatialCoordinate(sub)
    c.interpolate(.1*X[0])
    direct, dt, ice = build(meshes, c, direct=True, exponent=exponent, model=model)
    coupled, cdt, cice = build(meshes, c, direct=False, exponent=exponent, model=model)
    for k, h in enumerate(STEPS):
        dt.assign(h)
        cdt.assign(h)
        X = fd.SpatialCoordinate(ice.function_space().mesh())
        ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        cice.assign(ice)
        direct.solve()
        # The reference Newton is not a taped solve.
        with stop_annotating():
            coupled.solve()
        for index in ('displacement', 'potential'):
            a = direct.solution.subfunctions[direct.field_index[index]]
            b = coupled.solution.subfunctions[coupled.field_index[index]]
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=5e-7, atol=2e-9)
        np.testing.assert_allclose(
            direct.internal_variables.dat.data_ro,
            coupled.solution.subfunctions[coupled.field_index['history']].dat.data_ro,
            rtol=5e-7, atol=2e-9)
        # The `Real` fields, by name; both solvers hold the same ones.
        names = [direct.field_names[i] for i in direct.real_indices]
        assert names == [coupled.field_names[j] for j in coupled.real_indices]
        for name in names:
            np.testing.assert_allclose(
                direct.real_field(name).dat.data_ro,
                coupled.real_field(name).dat.data_ro, rtol=5e-7, atol=2e-9)


def march(meshes, control, *, direct, model='maxwell', checkpoint_dir=None):
    with stop_annotating():
        solver, dt, ice = build(meshes, control, direct=direct, model=model)
    if checkpoint_dir is not None:
        from firedrake.adjoint import enable_disk_checkpointing
        from checkpoint_schedules import SingleDiskStorageSchedule
        if meshes[1].comm.rank == 0:
            checkpoint_dir.mkdir()
        meshes[1].comm.barrier()
        enable_disk_checkpointing(checkpoint_comm=meshes[1].comm, checkpoint_dir=str(checkpoint_dir))
        get_working_tape().enable_checkpointing(SingleDiskStorageSchedule(), gc_timestep_frequency=1)
    iterator = enumerate(STEPS)
    if annotate_tape():
        iterator = get_working_tape().timestepper(iter(iterator))
    for k, h in iterator:
        dt.assign(h)
        X = fd.SpatialCoordinate(ice.function_space().mesh())
        ice.interpolate(.001*(1+.2*k)*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        solver.solve()
    sub = meshes[1]
    u = solver.field('displacement')
    if direct:
        histories = solver.internal_variables
    else:
        histories = solver.field('history')
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree': 4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    return fd.assemble((fd.inner(u, u)+.1*fd.inner(histories, histories))*measure)


@pytest.mark.parametrize('model', ['maxwell', 'backstress'])
@pytest.mark.parametrize('checkpoint', ['memory', 'disk'])
def test_full_mixed_direct_adjoint(meshes, model, checkpoint, tmp_path):
    sub = meshes[1]
    with stop_annotating():
        control = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
        direction = fd.Function(control.function_space()).interpolate(.1+.03*fd.SpatialCoordinate(sub)[0])
    continue_annotation()
    c = Control(control)
    directory = None
    if checkpoint == 'disk':
        from pathlib import Path
        directory = Path(sub.comm.bcast(str(tmp_path/'checkpoints'), root=0))
    J = march(meshes, control, direct=True, model=model, checkpoint_dir=directory)
    rf = ReducedFunctional(J, c)
    pause_annotation()
    replay = float(rf(control))
    assert abs(replay-float(J)) < 1e-8*abs(float(J))
    gradient = rf.derivative()
    assert_taylor_with_guards(rf, control, direction, J, min_rate=1.9)
    directional = direction._ad_dot(gradient)
    if sub.comm.size > 1:
        return  # The independently assembled monolithic reference is serial.
    eps = 1e-4
    with stop_annotating():
        plus = control.copy(deepcopy=True).assign(control + eps*direction)
        minus = control.copy(deepcopy=True).assign(control - eps*direction)
        difference = (float(march(meshes, plus, direct=False, model=model))
                      - float(march(meshes, minus, direct=False, model=model)))/(2*eps)
    assert abs(difference-directional) < 2e-5*abs(directional)


def test_direct_iterative_matches_lu_block_reference(meshes):
    _, sub = meshes
    c = fd.Function(fd.FunctionSpace(sub, 'CG', 1)).assign(.1)
    reference, dt, ice = build(meshes, c, direct=True)
    iterative, idt, iice = build(meshes, c, direct=True, iterative=True)
    X = fd.SpatialCoordinate(sub)
    for h in STEPS[:2]:
        dt.assign(h)
        idt.assign(h)
        ice.interpolate(.001*(1+.5*fd.cos(2*fd.atan2(X[1], X[0]))))
        iice.assign(ice)
        reference.solve()
        iterative.solve()
        for a, b in zip(reference.solution.subfunctions, iterative.solution.subfunctions):
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=1e-6, atol=1e-9)
