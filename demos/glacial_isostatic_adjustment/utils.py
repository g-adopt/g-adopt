import ufl
from firedrake import *
from gadopt.utility import vertical_component

def ice_sheet_disc(
        X: ufl.geometry.SpatialCoordinate,
        disc_centre: float,
        disc_halfwidth: float,
        radius: float = 6371e3,
        surface_dx_smooth: float = 200e3
) -> ufl.core.expr.Expr:
    '''Initialises ice sheet disc with a smooth transition given
    by a tanh profile

    Args:
      X:
        Spatial coordinate associated with the mesh
      disc_centre:
        centre of disc in radians (assumed to be between -pi to pi)
      disc_halfwidth:
        Half width of disc in radians
      radius:
        Radius of domain in m
      surface_dx_smooth:
        characteristic length scale for tanh smoothing in m

    Returns:
      Expression for ice sheet disc with tanh smoothing
    '''

    # Setup lengthscales for tanh smoothing
    surface_resolution_radians_smooth = surface_dx_smooth / radius

    # angle phi defined between -pi -> pi radians with zero at 'north pole'
    # (x,y) = (0,R) and -pi / pi transition at 'south pole' (x,y) = (0, -R)
    # where R is the radius of the domain
    phi = atan2(X[0], X[1])

    angular_distance_raw = abs(phi - disc_centre)
    # Angular distance accounting for discontinuity at 'south pole'
    angular_distance = conditional(
        angular_distance_raw < pi, angular_distance_raw, 2 * pi - angular_distance_raw
    )

    arg = angular_distance - disc_halfwidth
    disc = 0.5*(1-tanh(arg / (2*surface_resolution_radians_smooth)))
    return disc


def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(
        X: ufl.geometry.SpatialCoordinate,
        background_viscosity: Function,
        viscosity_scale: float = 1e21,
        r_lith: float = 6301e3,
        domain_depth: float = 2891e3,
) -> Function:
    '''Adds lateral variations to a background viscosity field in a 2D annulus

    The synthetic lateral viscosity variations consist of 5 'blobs'
    constructed from bivariate gaussian functions to represent interesting features
    in the mantle. We assume the background viscosity only varies in the
    radial direction and do not make modifications to the viscosity structure
    in the lithosphere i.e. for r > `r_lith' where r is the radial distance.

    Args:
      X:
        Spatial coordinate associated with the mesh
      background_viscosity:
        Background radial viscosity field (N.b. this is not modified)
      viscosity_scale:
        Characteristc viscosity used for nondimensionalisation
      r_lith:
        Radius of the lithosphere-mantle boundary in m
      domain_depth:
        Domain depth in m used for nondimensionalisation

    Returns:
      heterogenous_viscosity_field
        A new field containing the updated lateral viscosity variations
    '''
    heterogenous_viscosity_field = Function(background_viscosity.function_space(),
                                            name='viscosity')

    # Set up magnitudes of low and high viscosity regions
    low_visc = 1e20/viscosity_scale
    high_visc = 1e22/viscosity_scale

    # Add a low viscosity region in the bottom left corner
    # of the domain, aiming to mimic the low viscosity zone under the West
    # Antarctic ice sheet
    southpole_x, southpole_y = -2e6/domain_depth, -5.5e6/domain_depth
    low_viscosity_southpole = bivariate_gaussian(X[0], X[1],
                                                 southpole_x, southpole_y,
                                                 1.5e6/domain_depth,
                                                 0.5e6/domain_depth,
                                                 -0.4)

    heterogenous_viscosity_field.interpolate(
        low_visc*low_viscosity_southpole + background_viscosity * (1-low_viscosity_southpole))

    # Add two symmetrical low viscosity zones near the core-mantle boundary, inspired by
    # Large Low-Shear-Velocity Provinces (referred to as `llsvp`) so that
    # we can investigate sensitivity in the lower mantle.
    llsvp1_x, llsvp1_y = 3.5e6/domain_depth, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6/domain_depth,
                                1e6/domain_depth, 0)

    heterogenous_viscosity_field.interpolate(low_visc*llsvp1 +
                                             heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6/domain_depth, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6/domain_depth,
                                1e6/domain_depth, 0)

    heterogenous_viscosity_field.interpolate(low_visc*llsvp2 +
                                             heterogenous_viscosity_field * (1-llsvp2))

    # Add an elongated high viscosity region in the top right corner of the domain
    # to represent a slab geometry
    slab_x, slab_y = 3e6/domain_depth, 4.5e6/domain_depth
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6/domain_depth,
                              0.35e6/domain_depth, 0.7)

    heterogenous_viscosity_field.interpolate(high_visc*slab +
                                             heterogenous_viscosity_field * (1-slab))

    # Add a high viscosity feature at the top of the domain representing a craton
    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6/domain_depth
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x,
                                               high_viscosity_craton_y,
                                               1.5e6/domain_depth,
                                               0.5e6/domain_depth, 0.2)

    heterogenous_viscosity_field.interpolate(
        high_visc*high_viscosity_craton +
        heterogenous_viscosity_field * (1-high_viscosity_craton)
    )

    # We usually assume the lithosphere is purely elastic for GIA simulations,
    # so we reset viscosity in the lithosphere to the original background viscosity
    # value, which is assumed to be an arbitarily high constant so that the Maxwell
    # time in this layer is much larger than the timestep and simulation duration.
    heterogenous_viscosity_field.interpolate(
        conditional(vertical_component(X) > r_lith/domain_depth,
                    background_viscosity,
                    heterogenous_viscosity_field))

    return heterogenous_viscosity_field
