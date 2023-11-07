import firedrake as fd
import numpy as np
from scipy.special import erf

from helper import node_coordinates


def initialise_temperature(T, benchmark, parameters):
    """Set up the initial temperature field"""
    node_coords_x, node_coords_y = node_coordinates(T)

    match benchmark:
        case "van_Keken_1997_thermochemical":
            Ra, domain_length_x = parameters

            u0 = (
                domain_length_x ** (7 / 3)
                / (1 + domain_length_x**4) ** (2 / 3)
                * (Ra / 2 / np.sqrt(np.pi)) ** (2 / 3)
            )
            v0 = u0
            Q = 2 * np.sqrt(domain_length_x / np.pi / u0)
            Tu = erf((1 - node_coords_y) / 2 * np.sqrt(u0 / node_coords_x)) / 2
            Tl = 1 - 1 / 2 * erf(
                node_coords_y / 2 * np.sqrt(u0 / (domain_length_x - node_coords_x))
            )
            Tr = 1 / 2 + Q / 2 / np.sqrt(np.pi) * np.sqrt(
                v0 / (node_coords_y + 1)
            ) * np.exp(-(node_coords_x**2) * v0 / (4 * node_coords_y + 4))
            Ts = 1 / 2 - Q / 2 / np.sqrt(np.pi) * np.sqrt(
                v0 / (2 - node_coords_y)
            ) * np.exp(
                -((domain_length_x - node_coords_x) ** 2) * v0 / (8 - 4 * node_coords_y)
            )

            T.dat.data[:] = Tu + Tl + Tr + Ts - 3 / 2
            fd.DirichletBC(T.function_space(), 1, 3).apply(T)
            fd.DirichletBC(T.function_space(), 0, 4).apply(T)
            T.interpolate(fd.max_value(fd.min_value(T, 1), 0))
        case "Robey_2019":
            A, k = parameters

            mask_bottom = node_coords_y <= 1 / 10
            mask_top = node_coords_y >= 9 / 10

            T.dat.data[:] = 0.5
            T.dat.data[mask_bottom] = (
                1
                - 5 * node_coords_y[mask_bottom]
                + A
                * np.sin(10 * np.pi * node_coords_y[mask_bottom])
                * (1 - np.cos(2 / 3 * k * np.pi * node_coords_x[mask_bottom]))
            )
            T.dat.data[mask_top] = (
                5
                - 5 * node_coords_y[mask_top]
                + A
                * np.sin(10 * np.pi * node_coords_y[mask_top])
                * (1 - np.cos(2 / 3 * k * np.pi * node_coords_x[mask_top] + np.pi))
            )
        case "Trim_2023":
            Ra, RaB, f, k, layer_interface_y, domain_length_x = parameters

            C0 = 1 / (1 + np.exp(-2 * k * (layer_interface_y - node_coords_y)))

            T.dat.data[:] = (
                -np.pi**3
                * (domain_length_x**2 + 1) ** 2
                / domain_length_x**3
                * np.cos(np.pi * node_coords_x / domain_length_x)
                * np.sin(np.pi * node_coords_y)
                * f
                + RaB * C0
                + (Ra - RaB) * (1 - node_coords_y)
            ) / Ra
