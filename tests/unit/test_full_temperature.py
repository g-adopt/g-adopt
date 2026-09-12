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
def test_full_temperature_manufactured_energy_step(mesh, approx_class, family):
    _, y = fd.SpatialCoordinate(mesh)
    q_space = fd.FunctionSpace(mesh, family, 2)
    v_space = fd.VectorFunctionSpace(mesh, "CG", 2)
    exact = 0.091 + 1 - y
    temperature = fd.Function(q_space).interpolate(exact)
    u = fd.Function(v_space).interpolate(fd.as_vector((0, 0.2)))
    di = 0.5
    # For this affine solution, diffusion and viscous heating vanish.
    # rho H balances rho u.grad(T) + Di rho w T, with cp=alpha=g=1.
    heating = fd.Function(q_space).interpolate(-0.2 + di * 0.2 * exact)
    kwargs = {}
    reference = 0.091
    if approx_class is not FullEBA:
        reference = 0.091 * fd.exp(di * (1 - y))
        kwargs["rho"] = fd.exp(di * (1 - y))
    approximation = approx_class(1e5, di,
                                 reference_temperature=reference,
                                 H=heating, **kwargs)
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
