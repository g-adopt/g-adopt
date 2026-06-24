import firedrake as fd
from scipy.constants import R, g
from ufl.core.operator import Operator

from gadopt import material_field

import parameters as prms
from utility import clip_expression, tensor_second_invariant


def viscous_creep(
    pressure: Operator,
    temperature: Operator,
    strain_rate_sec_inv: Operator,
    prefactor: float,
    n: float,
    act_nrg: float,
    act_vol: float,
    bounds: dict[str, float | Operator] | None = None,
) -> Operator:
    viscosity = prefactor ** (-1.0 / n) / 2.0 * strain_rate_sec_inv ** ((1.0 - n) / n)
    viscosity *= fd.exp((act_nrg + pressure * act_vol) / n / R / temperature)
    if bounds is not None:
        viscosity = clip_expression(viscosity, **bounds)

    return viscosity


def effective_viscosity(
    values: list[Operator], bounds: dict[str, float | Operator] | None = None
) -> Operator:
    viscosity = 1.0 / sum(1.0 / value for value in values)
    if bounds is not None:
        viscosity = clip_expression(viscosity, **bounds)

    return viscosity


def viscosity_strain_rate_balance(
    mesh: fd.MeshGeometry,
    strain_rate: Operator,
    pressure: Operator,
    temperature: Operator,
    viscous_creep_params: dict[str, dict[str, float]],
    bounds: Operator,
) -> dict[str, dict[str, Operator]]:
    measure = fd.dx(domain=mesh, degree=5)
    domain_volume = fd.assemble(1.0 * measure)

    bounds_wide = {"minimum": 1e15, "maximum": 1e30}
    viscosity = {}
    for mantle, mechanism_params in viscous_creep_params.items():
        viscosity[mantle] = {}

        strain_rate_sec_inv = tensor_second_invariant(
            strain_rate, 1e-25 * prms.time_scale
        )
        for mechanism, params in mechanism_params.items():
            viscosity[mantle][mechanism] = viscous_creep(
                pressure,
                temperature,
                strain_rate_sec_inv / prms.time_scale,
                **params,
                bounds=bounds_wide,
            )

        if len(mechanism_params) > 1:
            viscosity_eff = effective_viscosity(viscosity[mantle].values())
            while True:
                for mechanism, params in mechanism_params.items():
                    strain_rate_mechanism = (
                        viscosity_eff / viscosity[mantle][mechanism] * strain_rate
                    )
                    strain_rate_sec_inv = tensor_second_invariant(
                        strain_rate_mechanism, 1e-25 * prms.time_scale
                    )
                    viscosity[mantle][mechanism] = viscous_creep(
                        pressure,
                        temperature,
                        strain_rate_sec_inv / prms.time_scale,
                        **params,
                        bounds=bounds_wide,
                    )

                viscosity_eff_old = viscosity_eff
                viscosity_eff = effective_viscosity(viscosity[mantle].values())
                relative_variation = (
                    abs(viscosity_eff - viscosity_eff_old) / viscosity_eff_old
                )
                if fd.assemble(relative_variation * measure) / domain_volume <= 5e-4:
                    break

        viscosity[mantle]["effective"] = effective_viscosity(
            viscosity[mantle].values(), bounds
        )

    return viscosity


def yield_point(
    pressure: Operator, surf_strength: float, max_strength: float, friction_coeff: float
) -> Operator:
    return fd.min_value(surf_strength + friction_coeff * pressure, max_strength)


def plastic_deformation(
    yield_strength: Operator,
    strain_rate_sec_inv: Operator,
    bounds: dict[str, float] | None = None,
) -> Operator:
    viscosity = yield_strength / 2.0 / strain_rate_sec_inv
    if bounds is not None:
        viscosity = clip_expression(viscosity, **bounds)

    return viscosity


def dominant_creep(
    creep_laws: dict[str, Operator],
    deformation_tags: dict[str, int],
    visc_min: Operator | float,
    def_mech: Operator | int = -1,
) -> Operator:
    creep, viscosity = creep_laws.popitem()
    visc_min = fd.min_value(viscosity, visc_min)
    def_mech = fd.conditional(viscosity <= visc_min, deformation_tags[creep], def_mech)

    if len(creep_laws) == 0:
        return def_mech
    else:
        return dominant_creep(creep_laws, deformation_tags, visc_min, def_mech)


def active_deformation(
    creep_laws: dict[str, Operator],
    deformation_tags: dict[str, int],
    visc_bounds: dict[str, float],
    yield_criterion: Operator,
    viscosity: Operator,
) -> Operator:
    def_mech = dominant_creep(
        creep_laws.copy(), deformation_tags, len(creep_laws) * visc_bounds["maximum"]
    )
    if prms.plastic_deformation:
        def_mech = fd.conditional(
            yield_criterion, deformation_tags["plastic"], def_mech
        )
    def_mech = fd.conditional(
        viscosity <= visc_bounds["minimum"], deformation_tags["minimum"], def_mech
    )
    def_mech = fd.conditional(
        viscosity >= visc_bounds["maximum"], deformation_tags["maximum"], def_mech
    )

    return def_mech


def material_viscosity(
    mesh: fd.MeshGeometry, u: fd.Function, T: fd.Function, psi: fd.Function
) -> tuple[Operator, dict[str, Operator]]:
    epsilon_dot = fd.sym(fd.grad(u))
    epsilon_dot_ii = tensor_second_invariant(epsilon_dot, 1e-25 * prms.time_scale)

    depth = prms.domain_dims[1] - fd.SpatialCoordinate(mesh)[1]
    # The following expressions are dimensional
    lith_pres = prms.rho_mantle * g * depth * prms.distance_scale
    T_full = (
        prms.temperature_scaling(T, dimensional=False)
        + depth * prms.distance_scale * prms.adiab_grad
    )

    eta_bounds_material = {}
    for bound, material_bounds in prms.eta_bounds.items():
        eta_bounds_material[bound] = material_field(
            psi, list(material_bounds.values()), "geometric"
        )

    yield_strength = [
        yield_point(lith_pres, **params)
        for params in prms.plastic_deformation_params.values()
    ]
    yield_strength_material = material_field(psi, yield_strength, "geometric")
    eta_plastic_material = plastic_deformation(
        yield_strength_material, epsilon_dot_ii / prms.time_scale, eta_bounds_material
    )

    eta_viscous = viscosity_strain_rate_balance(
        mesh,
        epsilon_dot,
        lith_pres,
        T_full,
        prms.viscous_creep_params,
        eta_bounds_material,
    )
    eta_viscous_effective = fd.conditional(
        depth <= prms.depth_lower_mantle,
        eta_viscous["upper"].pop("effective"),
        eta_viscous["lower"].pop("effective"),
    )

    tau = 2.0 * eta_viscous_effective / prms.viscosity_scale * epsilon_dot
    tau_ii = tensor_second_invariant(tau)

    yield_criterion = (
        tau_ii * prms.viscosity_scale / prms.time_scale >= yield_strength_material
    )
    if prms.plastic_deformation:
        eta_material = fd.conditional(
            yield_criterion, eta_plastic_material, eta_viscous_effective
        )
    else:
        eta_material = eta_viscous_effective

    def_mech = {}
    for mantle, creep_laws in eta_viscous.items():
        def_mech[mantle] = active_deformation(
            creep_laws,
            prms.def_mech_tags,
            eta_bounds_material,
            yield_criterion,
            eta_material,
        )
    def_mech_material = fd.conditional(
        depth <= prms.depth_lower_mantle, def_mech["upper"], def_mech["lower"]
    )

    eta_material /= prms.viscosity_scale

    rheol_expr = {
        "Deformation mechanism": def_mech_material,
        "Deviatoric stress (second invariant)": tau_ii,
        "Strain-rate (second invariant)": epsilon_dot_ii,
        "Viscosity": eta_material,
    }

    return eta_material, rheol_expr
