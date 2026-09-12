"""Full-temperature approximation and energy-equation checks."""

import firedrake as fd
import pytest

from gadopt import EnergySolver, ImplicitMidpoint
from gadopt.approximations import (
    AnelasticLiquidApproximation,
    TruncatedAnelasticLiquidApproximation,
)
from gadopt import (
    FullTemperatureAnelasticLiquidApproximation as FullALA,
    FullTemperatureExtendedBoussinesqApproximation as FullEBA,
    FullTemperatureTruncatedAnelasticLiquidApproximation as FullTALA,
)


@pytest.fixture
def mesh():
    m = fd.UnitSquareMesh(4, 4, quadrilateral=True)
    m.cartesian = True
    return m


@pytest.mark.parametrize("old_class,new_class", [
    (TruncatedAnelasticLiquidApproximation, FullTALA),
    (AnelasticLiquidApproximation, FullALA),
])
def test_analytic_change_of_variable(mesh, old_class, new_class):
    x, y = fd.SpatialCoordinate(mesh)
    di, ts = 0.5, 0.091
    reference = ts * fd.exp(di * (1 - y))
    density = fd.exp(di * (1 - y))
    offset = reference - ts  # Existing Cartesian demo convention.
    theta = x * (1 - x) + 1 - y
    temperature = theta + reference
    u = fd.as_vector((x * y, y * (1 - y)))
    p = x - y
    old = old_class(1e5, di, rho=density, Tbar=offset)
    new = new_class(1e5, di, rho=density,
                    reference_temperature=reference)
    sink = old.linearized_energy_sink(u)
    # Compare complete steady strong energy residuals, including advection.
    old_r = (old.rhocp() * fd.dot(u, fd.grad(theta)) + sink * theta
             - fd.div(fd.grad(theta + offset)) - old.energy_source(u))
    new_r = (new.rhocp() * fd.dot(u, fd.grad(temperature))
             + sink * temperature - fd.div(fd.grad(temperature))
             - new.energy_source(u))
    assert fd.assemble((old_r - new_r)**2 * fd.dx) < 1e-20
    assert fd.assemble((old.buoyancy(p, theta)
                        - new.buoyancy(p, temperature))**2 * fd.dx) < 1e-16
    if new_class is FullALA:
        assert fd.assemble((old.dbuoyancydp(p, theta)
                            - new.dbuoyancydp(p, temperature))**2 * fd.dx) == 0


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
@pytest.mark.parametrize("family", ["CG", "DG"])
@pytest.mark.parametrize("offset", [0, 0.091])
def test_full_temperature_manufactured_energy_step(mesh, approx_class, family, offset):
    _, y = fd.SpatialCoordinate(mesh)
    q_space = fd.FunctionSpace(mesh, family, 2)
    v_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    exact = 0.091 + 1 - y - offset
    temperature = fd.Function(q_space).interpolate(exact)
    u = fd.Function(v_space).interpolate(fd.as_vector((0, 0.2)))
    di = 0.5
    # For this affine solution, diffusion and viscous heating vanish.
    # rho H balances rho u.grad(T) + Di rho w T, with cp=alpha=g=1.
    heating = fd.Function(q_space).interpolate(-0.2 + di * 0.2 * (exact + offset))
    kwargs = {}
    reference = 0.091
    if approx_class is not FullEBA:
        reference = 0.091 * fd.exp(di * (1 - y))
        kwargs["rho"] = fd.exp(di * (1 - y))
    approximation = approx_class(1e5, di,
                                 reference_temperature=reference,
                                 H=heating, temperature_offset=offset, **kwargs)
    solver = EnergySolver(
        temperature, u, approximation, fd.Constant(0.01), ImplicitMidpoint,
        bcs={i: {"T": exact} for i in (1, 2, 3, 4)},
        # This manufactured problem is linear; perform one direct linear solve.
        solver_parameters={"snes_type": "ksponly", "ksp_type": "preonly",
                           "pc_type": "lu"},
    )
    solver.solve()
    assert fd.assemble((temperature - exact)**2 * fd.dx) < 1e-18


def test_eba_offset_is_physically_required(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    u = fd.as_vector((x, 1 + y))
    approximation = FullEBA(1e5, 0.5, reference_temperature=0.091)
    theta = 1 - y
    full_work = approximation.work_against_gravity(u, theta + 0.091)
    incorrect_work = approximation.work_against_gravity(u, theta)
    assert fd.assemble((full_work - incorrect_work)**2 * fd.dx) > 1e-4


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
def test_ambiguous_offsets_rejected(approx_class):
    with pytest.raises(ValueError, match="ambiguous"):
        approx_class(1e5, 0.5, reference_temperature=0.091, Tbar=1)


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
def test_vector_reference_rejected(mesh, approx_class):
    with pytest.raises(ValueError, match="scalar"):
        approx_class(1e5, 0.5, reference_temperature=fd.SpatialCoordinate(mesh))


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
def test_constant_shift_physics(mesh, approx_class):
    x, y = fd.SpatialCoordinate(mesh)
    reference = 0.091 if approx_class is FullEBA else 0.091*fd.exp(0.5*(1-y))
    kwargs = dict(reference_temperature=reference, heating_weight=0.7,
                  H=0.2, rho=fd.exp(0.5*(1-y)))
    absolute = approx_class(1e4, 0.5, **kwargs)
    offset = fd.Constant(0.091)
    shifted = approx_class(1e4, 0.5, temperature_offset=offset, **kwargs)
    t, p = 1-y+x*y, x-y
    u = fd.as_vector((x*y, y*(1-y)))
    for a, b in [
        (absolute.buoyancy(p, t+offset), shifted.buoyancy(p, t)),
        (absolute.work_against_gravity(u, t+offset), shifted.work_against_gravity(u, t)),
        (absolute.linearized_energy_sink(u)*(t+offset)-absolute.energy_source(u),
         shifted.linearized_energy_sink(u)*t-shifted.energy_source(u)),
        (absolute.viscous_dissipation(u), shifted.viscous_dissipation(u)),
        (absolute.temperature_anomaly(t+offset), shifted.temperature_anomaly(t)),
    ]:
        assert fd.assemble((a-b)**2*fd.dx) < 1e-18
    # Changing a Constant must retain its UFL dependency, not freeze a float.
    offset.assign(0.2)
    assert fd.assemble((shifted.absolute_temperature(t)-t-0.2)**2*fd.dx) < 1e-24
    assert fd.assemble((shifted.buoyancy(p, t)-absolute.buoyancy(p, t+offset))**2*fd.dx) < 1e-18


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
def test_spatial_offset_rejected(mesh, approx_class):
    _, y = fd.SpatialCoordinate(mesh)
    with pytest.raises(ValueError, match="temperature_offset"):
        approx_class(1e4, 0.5, reference_temperature=0.091, temperature_offset=y)


@pytest.mark.parametrize("approx_class", [FullEBA, FullTALA, FullALA])
def test_offset_energy_adjoint(approx_class):
    from firedrake.adjoint import Control, ReducedFunctional, taylor_test
    from pyadjoint.tape import (
        Tape, annotate_tape, continue_annotation, get_working_tape,
        pause_annotation, set_working_tape,
    )

    old_tape, was_annotating = get_working_tape(), annotate_tape()
    set_working_tape(Tape())
    continue_annotation()
    try:
        mesh = fd.UnitSquareMesh(4, 4, quadrilateral=True)
        mesh.cartesian = True
        _, y = fd.SpatialCoordinate(mesh)
        Q = fd.FunctionSpace(mesh, "CG", 2)
        V = fd.VectorFunctionSpace(mesh, "CG", 2)
        T = fd.Function(Q).interpolate(1-y)
        u = fd.Function(V).interpolate(fd.as_vector((0, 0.3)))
        offset = fd.Function(fd.FunctionSpace(mesh, "R", 0)).assign(0.091)
        control = Control(offset)
        approximation = approx_class(
            1e4, 0.5, reference_temperature=0.091, temperature_offset=offset)
        solver = EnergySolver(
            T, u, approximation, fd.Constant(0.1), ImplicitMidpoint,
            bcs={3: {"T": 1}, 4: {"T": 0}},
            solver_parameters={"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "lu"})
        solver.solve()
        objective = fd.assemble(1e4*T**2*fd.dx)
        functional = ReducedFunctional(objective, control)
        assert taylor_test(functional, offset, fd.Function(offset.function_space()).assign(0.1)) > 1.9
    finally:
        set_working_tape(old_tape)
        if not was_annotating:
            pause_annotation()
