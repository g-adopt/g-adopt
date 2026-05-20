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
    bounds: dict[str, float | Operator],
    bounds_material: Operator,
) -> dict[str, dict[str, Operator]]:
    measure = fd.dx(domain=mesh, degree=5)

    viscosity = {}
    for mantle, mechanism_params in viscous_creep_params.items():
        viscosity[mantle] = {}
        bounds_mantle = bounds.copy()
        bounds_mantle["maximum"] *= len(mechanism_params)

        strain_rate_sec_inv = tensor_second_invariant(
            strain_rate, 1e-25 * prms.time_scale
        )
        for mechanism, params in mechanism_params.items():
            viscosity[mantle][mechanism] = viscous_creep(
                pressure,
                temperature,
                strain_rate_sec_inv / prms.time_scale,
                **params,
                bounds=bounds_mantle,
            )

        viscosity_eff = effective_viscosity(viscosity[mantle].values(), bounds_material)
        viscosity_eff_old = viscosity_eff

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
                    bounds=bounds_mantle,
                )

            viscosity_eff = effective_viscosity(
                viscosity[mantle].values(), bounds_material
            )
            variation = fd.assemble(abs(viscosity_eff - viscosity_eff_old) * measure)
            relative_variation = variation / fd.assemble(viscosity_eff_old * measure)
            if relative_variation <= 1e-4:
                break
            viscosity_eff_old = viscosity_eff

        viscosity[mantle]["effective"] = viscosity_eff

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
    def_mech = fd.conditional(
        fd.eq(viscosity, visc_min), deformation_tags[creep], def_mech
    )

    if len(creep_laws) == 0:
        return def_mech
    else:
        return dominant_creep(creep_laws, deformation_tags, visc_min, def_mech)


def active_deformation(
    deformation_tags: dict[str, int],
    creep_laws: dict[str, Operator],
    yield_criterion: Operator,
    viscosity: Operator,
    visc_bounds: dict[str, float],
) -> Operator:
    def_mech = dominant_creep(
        creep_laws.copy(), deformation_tags, visc_bounds["maximum"]
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
    y = fd.SpatialCoordinate(mesh)[1]
    depth = prms.domain_dims[1] - y
    # The following expressions are dimensional
    lith_pres = prms.rho_mantle * g * depth * prms.distance_scale
    T_full = (
        prms.temperature_scaling(T, dimensional=False)
        + depth * prms.distance_scale * prms.adiab_grad
    )

    eta_bounds_material = {}
    for bound in prms.eta_bounds["mantle"]:
        eta_bounds_material[bound] = material_field(
            psi,
            [prms.eta_bounds["mantle"][bound], prms.eta_bounds["weak layer"][bound]],
            "geometric",
        )

    epsilon_dot = fd.sym(fd.grad(u))
    epsilon_dot_ii = tensor_second_invariant(epsilon_dot, 1e-25 * prms.time_scale)

    yield_strength = {}
    eta_plastic = {}
    for material, params in prms.plastic_deformation_params.items():
        yield_strength[material] = yield_point(lith_pres, **params)
        eta_plastic[material] = plastic_deformation(
            yield_strength[material],
            epsilon_dot_ii / prms.time_scale,
            prms.eta_bounds[material],
        )
    yield_strength_material = material_field(
        psi, [yield_strength["mantle"], yield_strength["weak layer"]], "geometric"
    )
    eta_plastic_material = material_field(
        psi, [eta_plastic["mantle"], eta_plastic["weak layer"]], "geometric"
    )

    eta_viscous = viscosity_strain_rate_balance(
        mesh,
        epsilon_dot,
        lith_pres,
        T_full,
        prms.viscous_creep_params,
        prms.eta_bounds["mantle"],
        eta_bounds_material,
    )

    tau_upp_mant = (
        2.0 * eta_viscous["upper"]["effective"] / prms.viscosity_scale * epsilon_dot
    )
    tau_ii_upp_mant = tensor_second_invariant(tau_upp_mant)

    yield_criterion = (
        tau_ii_upp_mant * prms.viscosity_scale / prms.time_scale
        >= yield_strength_material
    )
    if prms.plastic_deformation:
        eta_upp_mant = fd.conditional(
            yield_criterion, eta_plastic_material, eta_viscous["upper"].pop("effective")
        )
    else:
        eta_upp_mant = eta_viscous["upper"].pop("effective")
    eta_material = fd.conditional(
        depth <= prms.depth_lower_mantle,
        eta_upp_mant,
        eta_viscous["lower"].pop("effective"),
    )

    def_mech = {}
    for mantle, creep_laws in eta_viscous.items():
        def_mech[mantle] = active_deformation(
            prms.def_mech_tags,
            creep_laws=creep_laws,
            yield_criterion=yield_criterion,
            viscosity=eta_material,
            visc_bounds=eta_bounds_material,
        )
    def_mech_material = fd.conditional(
        depth <= prms.depth_lower_mantle, def_mech["upper"], def_mech["lower"]
    )

    eta_material /= prms.viscosity_scale

    rheol_expr = {
        "Deformation mechanism": def_mech_material,
        "Deviatoric stress (second invariant)": tau_ii_upp_mant,
        "Strain-rate (second invariant)": epsilon_dot_ii,
        "Viscosity": eta_material,
    }

    return eta_material, rheol_expr
