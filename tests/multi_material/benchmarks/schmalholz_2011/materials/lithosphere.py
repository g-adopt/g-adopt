import firedrake as fd


def mu(velocity, *args):
    # Formulation from Glerum et al. (2018)
    # strain_rate = fd.dev(fd.sym(fd.grad(velocity)))
    # strain_rate_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2 + 1e-36)

    # Formulation from Schmalholz et al. (2008)
    vel_grad = fd.grad(velocity)
    strain_rate_inv = (
        fd.sqrt(fd.inner(vel_grad, vel_grad) - 2 * fd.det(vel_grad) + 1e-38) / 2
    )

    viscosity = visc_coeff * strain_rate_inv ** (1 / stress_exponent - 1)

    return fd.min_value(fd.max_value(viscosity, visc_bounds[0]), visc_bounds[1])


rho = 3300

visc_coeff = 4.75e11
stress_exponent = 4.0
visc_bounds = (1e21, 1e25)
