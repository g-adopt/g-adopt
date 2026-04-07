import firedrake as fd
import numpy as np
import shapely as sl
from mpi4py import MPI

from gadopt import assign_level_set_values, interface_thickness

import parameters as prms
from utility import half_space_cooling_model


def initial_temperature(T: fd.Function) -> None:
    x, y = fd.SpatialCoordinate(T.ufl_domain())
    depth = prms.domain_dims[1] - y

    hsr_plate = prms.trench_coords[0] / prms.age_plate
    hsr_overriding = (prms.domain_dims[0] - prms.trench_coords[0]) / prms.age_overriding
    dist_ann_centre = fd.sqrt(
        (x - prms.ann_centre[0]) ** 2 + (y - prms.ann_centre[1]) ** 2
    )
    slab_angle = fd.acos((prms.ann_outer_radius - depth) / dist_ann_centre)
    slab_rel_dist = prms.trench_coords[0] + dist_ann_centre * slab_angle
    slab_rel_depth = prms.ann_outer_radius - dist_ann_centre

    T_plate = half_space_cooling_model(
        prms.T_surf,
        prms.T_pot,
        depth * prms.distance_scale,
        prms.kappa,
        x / hsr_plate * prms.time_scale,
    )
    T_overriding = half_space_cooling_model(
        prms.T_surf,
        prms.T_pot,
        depth * prms.distance_scale,
        prms.kappa,
        (prms.domain_dims[0] - x) / hsr_overriding * prms.time_scale,
    )
    T_slab = half_space_cooling_model(
        prms.T_surf,
        prms.T_pot,
        slab_rel_depth * prms.distance_scale,
        prms.kappa,
        slab_rel_dist / hsr_plate * prms.time_scale,
    )

    is_slab = fd.And(
        dist_ann_centre <= prms.ann_outer_radius,
        y
        - ((x - prms.ann_centre[0]) / fd.tan(prms.slab_tip_angle) + prms.ann_centre[1])
        >= 0.0,
    )
    T_not_plate = fd.conditional(
        is_slab,
        prms.temperature_scaling(T_slab),
        prms.temperature_scaling(T_overriding),
    )
    T.interpolate(
        fd.conditional(
            x <= prms.trench_coords[0], prms.temperature_scaling(T_plate), T_not_plate
        )
    )


def initial_level_set(psi: fd.Function) -> None:
    surface_plate_coords = [
        prms.plate_extremity_coords,
        prms.trench_coords,
        (prms.trench_coords[0], prms.trench_coords[1] - prms.weak_layer_thickness),
        (
            prms.plate_extremity_coords[0],
            prms.plate_extremity_coords[1] - prms.weak_layer_thickness,
        ),
        prms.plate_extremity_coords,
    ]
    surface_plate = sl.Polygon(surface_plate_coords)

    ann_outer_circle = sl.Point(prms.ann_centre).buffer(prms.ann_outer_radius)
    ann_inner_circle = sl.Point(prms.ann_centre).buffer(
        prms.ann_outer_radius - prms.weak_layer_thickness
    )
    annulus = ann_outer_circle.difference(ann_inner_circle)

    slab_tip_triangle_coords = [
        prms.trench_coords,
        (
            prms.trench_coords[0] + prms.ann_outer_radius * np.tan(prms.slab_tip_angle),
            prms.trench_coords[1],
        ),
        prms.ann_centre,
        prms.trench_coords,
    ]
    slab_tip_triangle = sl.Polygon(slab_tip_triangle_coords)

    weak_layer_polygon = surface_plate.union(annulus.intersection(slab_tip_triangle))
    weak_layer_polygon_coords = weak_layer_polygon.exterior.coords._coords
    ([plate_extremity_index],) = np.all(
        weak_layer_polygon_coords == prms.plate_extremity_coords, axis=1
    ).nonzero()
    weak_layer_interface_coords = weak_layer_polygon_coords[: plate_extremity_index + 1]
    weak_layer_interface = sl.LineString(weak_layer_interface_coords)

    boundary_coordinates = weak_layer_polygon_coords[plate_extremity_index + 1 :]

    epsilon = interface_thickness(psi.function_space(), min_cell_edge_length=True)
    epsilon = MPI.COMM_WORLD.allreduce(epsilon.dat.data_ro.min(), MPI.MIN)
    assign_level_set_values(
        psi,
        epsilon,
        interface_geometry="shapely",
        interface=weak_layer_interface,
        boundary_coordinates=boundary_coordinates,
    )
