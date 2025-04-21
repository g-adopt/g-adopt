import firedrake as fd
from gadopt.utility import step_func
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np


def initialise_background_field(X, field, background_values, radius_values, vertical_tanh_width=40e3):
    profile = background_values[0]
    sharpness = 1 / vertical_tanh_width
    depth = fd.sqrt(X[0]**2 + X[1]**2)-radius_values[0]
    for i in range(1, len(background_values)):
        centre = radius_values[i] - radius_values[0]
        mag = background_values[i] - background_values[i-1]
        profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

    field.interpolate(profile)


def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
    numerator = fd.exp(-1/(2*(1-rho**2))*arg)
    if normalised_area:
        denominator = 2*fd.pi*sigma_x*sigma_y*(1-rho**2)**0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(X, viscosity):
    heterogenous_viscosity_field = fd.Function(viscosity.function_space(), name='viscosity')
    antarctica_x, antarctica_y = -2e6, -5.5e6

    low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4)
    heterogenous_viscosity_field.interpolate(-3*low_viscosity_antarctica + viscosity * (1-low_viscosity_antarctica))

    llsvp1_x, llsvp1_y = 3.5e6, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp1 + heterogenous_viscosity_field * (1-llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
    heterogenous_viscosity_field.interpolate(-3*llsvp2 + heterogenous_viscosity_field * (1-llsvp2))

    slab_x, slab_y = 3e6, 4.5e6
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
    heterogenous_viscosity_field.interpolate(-1*slab + heterogenous_viscosity_field * (1-slab))

    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6
    high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6, 0.5e6, 0.2)
    heterogenous_viscosity_field.interpolate(-1*high_viscosity_craton + heterogenous_viscosity_field * (1-high_viscosity_craton))

    return heterogenous_viscosity_field


def setup_normalised_ice_discs(X, radius_values, Hice1):
    # Disc ice load but with a smooth transition given by a tanh profile
    disc_halfwidth1 = (2*fd.pi/360) * 10  # Disk half width in radians
    disc_halfwidth2 = (2*fd.pi/360) * 20  # Disk half width in radians
    surface_dx = 200e3
    ncells = 2*fd.pi*radius_values[0] / surface_dx
    surface_resolution_radians = 2*fd.pi / ncells
    colatitude = fd.atan2(X[0], X[1])
    disc1_centre = (2*fd.pi/360) * 25  # Centre of disc1
    disc2_centre = fd.pi  # Centre of disc2
    disc1 = 0.5*(1-fd.tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
    disc2 = 0.5*(1-fd.tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))
    Hice2 = 2*Hice1
    return disc1 + (Hice2/Hice1)*disc2


ice_cmap = plt.get_cmap("Blues", 25)


def add_ice(p, m, scalar="normalised ice thickness", scalar_bar_args=None):

    if scalar_bar_args is None:
        scalar_bar_args = {
            "title": 'Normalised ice thickness',
            "position_x": 0.2,
            "position_y": 0.8,
            "vertical": False,
            "title_font_size": 22,
            "label_font_size": 18,
            "fmt": "%.1f",
            "font_family": "arial",
            "n_labels": 5,
        }
    data = m.read()[0]  # MultiBlock mesh with only 1 block

    # Extract boundary surface, remove inner surface and expand ring width
    surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                      feature_edges=False, manifold_edges=False)
    sphere = pv.Sphere(radius=5e6)
    clipped_surf = surf.clip_surface(sphere, invert=False)

    # Stretch line by 20%
    transform_matrix = np.array(
        [
            [1.2, 0, 0, 0],
            [0, 1.2, 0, 0],
            [0, 0, 1.2, 0],
            [0, 0, 0, 1],
        ])
    transformed_surf = clipped_surf.transform(transform_matrix)

    p.add_mesh(transformed_surf, scalars=scalar, line_width=10, clim=[0, 2], cmap=ice_cmap, scalar_bar_args=scalar_bar_args)


visc_cmap = plt.get_cmap("inferno_r", 25)


def add_viscosity(p, fname="viscosity.pvd"):
    reader = pv.get_reader(fname)
    visc_data = reader.read()[0]
    p.add_mesh(
        visc_data,
        component=None,
        lighting=False,
        show_edges=False,
        cmap=visc_cmap,
        clim=[-3, 2],
        scalar_bar_args={
            "title": 'Normalised viscosity',
            "position_x": 0.2,
            "position_y": 0.1,
            "vertical": False,
            "title_font_size": 22,
            "label_font_size": 18,
            "fmt": "%.0f",
            "font_family": "arial",
        }
    )


def add_displacement(p, m, disp="Displacement", scalar_bar_args=None):
    data = m.read()[0]  # MultiBlock mesh with only 1 block

    # Make a colour map
    boring_cmap = plt.get_cmap("inferno_r", 25)

    # Artificially warp the output data by the displacement field
    # Note the mesh is not really moving!
    warped = data.warp_by_vector(vectors=disp, factor=1500)
    if scalar_bar_args is None:
        scalar_bar_args = {
            "title": 'Displacement (m)',
            "position_x": 0.2,
            "position_y": 0.8,
            "vertical": False,
            "title_font_size": 20,
            "label_font_size": 16,
            "fmt": "%.0f",
            "font_family": "arial",
        }

    # Add the warped displacement field to the frame
    p.add_mesh(
        warped,
        scalars=disp,
        component=None,
        lighting=False,
        clim=[0, 600],
        cmap=boring_cmap,
        scalar_bar_args=scalar_bar_args,
    )


def add_sensitivity_ring(p, m, scalar_bar_args=None):
    # Make a colour map
    adj_cmap = plt.get_cmap("coolwarm", 25)
    if scalar_bar_args is None:
        scalar_bar_args = {
            "title": 'Adjoint sensitivity',
            "position_x": 0.2,
            "position_y": 0.8,
            "vertical": False,
            "title_font_size": 22,
            "label_font_size": 18,
            "fmt": "%.1e",
            "font_family": "arial",
            "n_labels": 3,
        }
    data = m.read()[0]  # MultiBlock mesh with only 1 block

    # Extract boundary surface, remove inner surface and expand ring width
    surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                      feature_edges=False, manifold_edges=False)
    sphere = pv.Sphere(radius=5e6)
    clipped_surf = surf.clip_surface(sphere, invert=False)

    transform_matrix = np.array(
        [
            [1.1, 0, 0, 0],
            [0, 1.1, 0, 0],
            [0, 0, 1.1, 0],
            [0, 0, 0, 1],
        ])

    transformed_surf = clipped_surf.transform(transform_matrix)
    p.add_mesh(transformed_surf, line_width=8, scalar_bar_args=scalar_bar_args, clim=[-5e-7, 5e-7], cmap=adj_cmap)
