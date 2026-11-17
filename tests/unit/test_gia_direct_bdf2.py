"""Direct elimination in the complete two-mesh GIA/sea-level mixed system."""
import gadopt
import firedrake as fd
import numpy as np
import ufl
import pytest
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                              pause_annotation, get_working_tape, stop_annotating)
from pyadjoint.tape import annotate_tape
from gadopt.gia_gravity import FluidCore, SeaLevel
import test_gia_gravity as base
from test_gravity_adjoint import assert_taylor_with_guards

@pytest.fixture(scope='module')
def meshes():
    import sys
    import tempfile
    from pathlib import Path
    pytest.importorskip('gmsh')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'demos'/'gravity'))
    from generate_selfgrav_annulus import generate
    from validate_selfgrav_annulus import curve_mesh
    path = Path(tempfile.gettempdir())/'gadopt_bdf2_triangle_annulus.msh'
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        generate(str(path), dr_mantle=.3, n_azimuthal=16, quads=False)
    fd.COMM_WORLD.barrier()
    parent = curve_mesh(fd.Mesh(str(path)))
    parent.cartesian = False
    sub = curve_mesh(fd.Submesh(parent, 2, base.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub

STEPS = (.08, .13, .07, .12)


@pytest.fixture(autouse=True)
def clean_tape():
    get_working_tape().clear_tape()
    yield
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


def build(meshes, control, *, direct, exponent=3, scheme='bdf2', iterative=False, model='maxwell'):
    parent, sub = meshes
    Z, layout = gadopt.self_gravitating_gia_space(
        sub, parent, gravity_bcs=base.gravity_bcs(parent, sheet=False), rotation=True,
        fluid_core=True, centre_of_mass=True, sea_level=True,
        n_internal_variables=2, condense_internal_variables=direct,
        internal_variable_family='Quadrature', internal_variable_degree=4,
        self_gravity_number=base.LAMBDA, dtn_representation='multiplier')
    state = fd.Function(Z)
    dt = fd.Function(fd.FunctionSpace(sub, 'R', 0)).assign(STEPS[0])
    Q = fd.FunctionSpace(sub, 'CG', 2)
    theta = fd.atan2(fd.SpatialCoordinate(sub)[1], fd.SpatialCoordinate(sub)[0])
    initial_sea = fd.Function(Q).interpolate(.005*fd.cos(theta))
    initial_ice = fd.Function(Q)
    ice = fd.Function(Q)
    power = exponent
    if exponent == 'spatial':
        power = fd.Function(fd.FunctionSpace(sub, 'DG', 0)).interpolate(
            fd.conditional(fd.SpatialCoordinate(sub)[0] > 0, 3., 1.))
    material = base.approximation(shear_modulus=[.4, .6],
        viscosity=[.12*fd.exp(control), .6*fd.exp(control)],
        exponent=power, transition_stress=.003)
    if model == 'backstress':
        material = gadopt.BackstressApproximation(
            bulk_modulus=1., density=1., shear_modulus=1., viscosity=.3*fd.exp(control),
            backstress_modulus=.8, recovery_time=.4*fd.exp(.3*control),
            exponent=exponent, transition_stress=.003, g=1., B_mu=base.B_MU,
            self_gravity_number=base.LAMBDA)
    sea = SeaLevel(boundary=base.CURVE_RE, rho_w=.1, rho_i=.09, g_surface=1.,
        SL_init=initial_sea, I=ice, I_init=initial_ice,
        N_init=fd.Constant(0.), ur_init=fd.Constant(0.), fixed_ocean=True, softplus_width=.001)
    X = fd.SpatialCoordinate(sub)
    measure = fd.Measure('dx', domain=sub, intersect_measures=(fd.dx(domain=parent),))
    options = dict(snes_rtol=1e-11, snes_atol=1e-13, ksp_rtol=1e-11,
        ksp_error_if_not_converged=True)
    nullspace = gadopt.rigid_rotation_nullspace(Z, layout)
    solver = gadopt.SelfGravitatingGIASolver(state, material, layout=layout, dt=dt,
        time_scheme=scheme, constitutive_method='direct' if direct else 'pointwise',
        bcs={}, fluid_core=FluidCore(boundary=base.CURVE_RC, rho_core=1.2, g=1.),
        sea_level=sea, rotation_moments={'C': fd.assemble(fd.dot(X, X)*measure)},
        nullspace=nullspace, transpose_nullspace=nullspace, quad_degree=4,
        solver_parameters='iterative' if iterative else 'direct', solver_parameters_extra=options)
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
        J = ufl.derivative(F, tuple(fields), fd.TrialFunctions(Z))
        class ReferenceNewton:
            # Assemble the unreduced mixed Jacobian as a nest, then let PETSc
            # convert it to AIJ. This bypasses mixed-coefficient splitting and
            # the prohibition on direct monolithic assembly with Real fields.
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
                    matrix = fd.assemble(J, mat_type='nest')
                    nest = matrix.petscmat
                    blocks = []
                    explicit = []
                    for row in range(len(Z)):
                        blocks.append([])
                        for col in range(len(Z)):
                            block = nest.getNestSubMatrix(row, col)
                            if block.getType() == 'python':
                                # Real couplings have one row or one column.
                                # Probe that dimension, using public Mat actions.
                                assert state.comm.size == 1
                                x, y = block.createVecs()
                                if block.getSize()[0] == 1 and block.getSize()[1] > 1:
                                    y.set(1.)
                                    block.multTranspose(y, x)
                                    values = x.array_r.copy().reshape(block.getSize())
                                else:
                                    x.set(1.)
                                    block.mult(x, y)
                                    values = y.array_r.copy().reshape(block.getSize())
                                x.destroy()
                                y.destroy()
                                block = PETSc.Mat().createDense(block.getSize(), array=values, comm=state.comm)
                                block = block.convert('aij')
                                explicit.append(block)
                            blocks[-1].append(block)
                    isrows, iscols = nest.getNestISs()
                    assembled = PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=state.comm)
                    assembled.assemble()
                    A = assembled.convert('aij')
                    ksp = PETSc.KSP().create(comm=state.comm)
                    ksp.setOperators(A)
                    ksp.setType('preonly')
                    ksp.getPC().setType('lu')
                    ksp.getPC().setFactorSolverType('mumps')
                    with residual.dat.vec_ro as r, state.dat.vec as x:
                        delta = x.duplicate()
                        ksp.solve(r, delta)
                        assert ksp.getConvergedReason() > 0
                        x.axpy(-1, delta)
                        delta.destroy()
                    ksp.destroy()
                    A.destroy()
                    assembled.destroy()
                    for block in explicit:
                        block.destroy()
                raise RuntimeError(f'Reference Newton failed, residual={norm}')
        solver.solver = ReferenceNewton()
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
        coupled.solve()
        for index in ('displacement', 'potential'):
            a = direct.solution.subfunctions[getattr(direct.layout, index)]
            b = coupled.solution.subfunctions[getattr(coupled.layout, index)]
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=5e-7, atol=2e-9)
        np.testing.assert_allclose(direct.internal_variables.dat.data_ro,
            coupled.solution.subfunctions[coupled.layout.internal_variable_field].dat.data_ro,
            rtol=5e-7, atol=2e-9)
        for i, j in zip(direct.layout.real_fields, coupled.layout.real_fields):
            np.testing.assert_allclose(direct.solution.subfunctions[i].dat.data_ro,
                coupled.solution.subfunctions[j].dat.data_ro, rtol=5e-7, atol=2e-9)


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
    u = fd.split(solver.solution)[solver.layout.displacement]
    if direct:
        histories = solver.internal_variables
    else:
        histories = fd.split(solver.solution)[solver.layout.internal_variable_field]
    measure = fd.Measure('dx', domain=sub, metadata={'quadrature_degree':4},
                         intersect_measures=(fd.dx(domain=meshes[0]),))
    return fd.assemble((fd.inner(u,u)+.1*fd.inner(histories,histories))*measure)


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
        ice.interpolate(.001*(1+.5*fd.cos(2*fd.atan2(X[1],X[0]))))
        iice.assign(ice)
        reference.solve()
        iterative.solve()
        for a, b in zip(reference.solution.subfunctions, iterative.solution.subfunctions):
            np.testing.assert_allclose(a.dat.data_ro, b.dat.data_ro, rtol=1e-6, atol=1e-9)
