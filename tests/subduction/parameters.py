import firedrake as fd
import numpy as np
from scipy.constants import g, zero_Celsius
from ufl.core.operator import Operator


def buoyancy_number(density_contrast: float) -> float:
    return density_contrast / rho_mantle / alpha / (T_pot - T_surf)


def temperature_scaling(
    temperature: float | fd.Function | Operator, dimensional: bool = True
) -> float | fd.Function | Operator:
    if not dimensionless:
        return temperature

    if dimensional:
        return (temperature - T_surf) / (T_pot - T_surf)
    else:
        return temperature * (T_pot - T_surf) + T_surf


# Switches
dimensionless = True
free_surface = True
mesh_generation = "gmsh"

# Spatial parameters
domain_height = 1e6
domain_aspect_ratio = 3.0
distance_scale = domain_height if dimensionless else 1.0
domain_dims = (
    domain_aspect_ratio * domain_height / distance_scale,
    domain_height / distance_scale,
)
depth_lower_mantle = 6.6e5 / distance_scale

# Mesh parameters (Firedrake)
mesh_elements = (225, 75)
# Mesh parameters (Gmsh)
mesh_layers = {
    "thickness": [x / distance_scale for x in [3.4e5, 5.8e5, 8e4]],
    "vertical_resolution": [x / distance_scale for x in [6.8e4, 2.9e4, 1e4]],
    "horizontal_resolution": 2e4 / distance_scale,
}

# Physical parameters
rho_mantle = 3.3e3
rho_weak_layer = 3.2e3
if free_surface:
    rho_water = 1e3
alpha = 3e-5
kappa = 6e-7
cp = 1.2e3
T_surf = zero_Celsius
T_pot = T_surf + 1.35e3
adiab_grad = alpha * g * T_pot / cp

# Non-dimensional numbers
mu_scale = 1e21 if dimensionless else 1.0
if dimensionless:
    Ra = rho_mantle * alpha * (T_pot - T_surf) * g * domain_height**3 / mu_scale / kappa
    B = buoyancy_number(rho_weak_layer - rho_mantle)

    if free_surface:
        BFS = [buoyancy_number(rho - rho_water) for rho in [rho_mantle, rho_weak_layer]]

# Temporal parameters
time_scale = domain_height**2 / kappa if dimensionless else 1.0
myr_to_seconds = 1e6 * 365.25 * 8.64e4
time_end = 50.0 * myr_to_seconds / time_scale
time_step = 1e11 / time_scale

# Rheology
mu_bounds = {
    "mantle": {"minimum": 1e18, "maximum": 1e25},
    "weak layer": {"minimum": 1e18, "maximum": 1e25},
}
plastic_deformation_params = {
    "mantle": {"surf_strength": 2e6, "max_strength": 1e10, "friction_coeff": 0.2},
    "weak layer": {"surf_strength": 2e6, "max_strength": 1e10, "friction_coeff": 0.02},
}
viscous_creep_params = {
    "upper": {
        "diffusion": {"prefactor": 1.5e-11, "n": 1.0, "act_nrg": 3e5, "act_vol": 4e-6},
        "dislocation": {
            "prefactor": 4.4e-17,
            "n": 3.5,
            "act_nrg": 5.4e5,
            "act_vol": 1.2e-5,
        },
        "Peierls": {
            "prefactor": 9.5e-157,
            "n": 20.0,
            "act_nrg": 5.4e5,
            "act_vol": 1e-5,
        },
    },
    "lower": {
        "diffusion": {"prefactor": 1e-18, "n": 1.0, "act_nrg": 2e5, "act_vol": 1.5e-6}
    },
}
viscous_creep_params["lower"]["diffusion"]["prefactor"] *= 30.0
def_mech_tags = {
    "diffusion": 0,
    "dislocation": 1,
    "Peierls": 2,
    "plastic": 3,
    "minimum": 4,
    "maximum": 5,
}

# Adaptivity
initial_adapt_loops = 3
adapt_calls = 3
metric_parameters = {  # For further information: `set_parameters` in animate/metric.py
    "dm_plex_metric": {
        "target_complexity": 20_000,  # Metric complexity, analogous to cell count
        "h_min": 3e3 / distance_scale,  # Minimum metric magnitude (i.e. cell size)
        "h_max": 5e5 / distance_scale,  # Maximum metric magnitude (i.e. cell size)
        "a_max": 10.0,  # Maximum metric anisotropy (cell aspect ratio)
        "p": np.inf,  # Metric normalisation order
        "gradation_factor": 1.3,  # Maximum variation in length between adjacent edges
    }
}
mesh_fields = {
    "Stokes": {"add_to_metric": [True, False], "scaling": [1e-3, None]},
    "Temperature": {"add_to_metric": True, "scaling": 1e0},
    "Level set": {"add_to_metric": True, "scaling": 1e0},
}
if free_surface:
    mesh_fields["Stokes"]["add_to_metric"].append(False)
    mesh_fields["Stokes"]["scaling"].append(None)

# Time loop
subcycles = 1
iterations = 10
output_frequency = 1

# Field initialisation
age_plate = 100.0 * myr_to_seconds / time_scale
age_overriding = 20.0 * myr_to_seconds / time_scale
plate_extremity_coords = (0.0, domain_dims[1])
trench_coords = (domain_dims[0] / 2.0, domain_dims[1])
weak_layer_thickness = 5e3 / distance_scale
ann_outer_radius = 2.5e5 / distance_scale
ann_centre = (trench_coords[0], trench_coords[1] - ann_outer_radius)
slab_tip_angle = np.deg2rad(77.0)
