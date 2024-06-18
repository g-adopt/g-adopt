import numpy as np
import shapely as sl

import gadopt as ga


def straight_line(x, slope, intercept):
    """Straight line equation"""
    return slope * x + intercept


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    """Cosine curve equation with an amplitude and a vertical shift"""
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


def isd_simple_curve(domain_origin_x, domain_dim_x, curve, parameters, level_set):
    """Initialise signed-distance function from a simple curve described by a
    mathematical function"""
    interface_x = np.linspace(domain_origin_x, domain_origin_x + domain_dim_x, 1000)
    interface_y = curve(interface_x, *parameters)
    line_string = sl.LineString([*np.column_stack((interface_x, interface_y))])
    sl.prepare(line_string)

    node_coords_x, node_coords_y = ga.node_coordinates(level_set)
    node_relation_to_curve = [
        (
            node_coord_y > curve(node_coord_x, *parameters),
            line_string.distance(sl.Point(node_coord_x, node_coord_y)),
        )
        for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y)
    ]
    return [dist if is_above else -dist for is_above, dist in node_relation_to_curve]


def isd_rectangle(parameters, level_set):
    """Initialise signed-distance function from a rectangle"""
    ref_vertex_x, ref_vertex_y, edge_length = parameters

    rectangle = sl.Polygon(
        [
            (ref_vertex_x, ref_vertex_y),
            (ref_vertex_x + edge_length, ref_vertex_y),
            (ref_vertex_x + edge_length, ref_vertex_y + edge_length),
            (ref_vertex_x, ref_vertex_y + edge_length),
            (ref_vertex_x, ref_vertex_y),
        ]
    )
    sl.prepare(rectangle)

    node_coords_x, node_coords_y = ga.node_coordinates(level_set)
    node_relation_to_rectangle = [
        (
            rectangle.contains(sl.Point(x, y))
            or rectangle.boundary.contains(sl.Point(x, y)),
            rectangle.boundary.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    return [
        dist if is_inside else -dist for is_inside, dist in node_relation_to_rectangle
    ]


def isd_schmalholz(parameters, level_set):
    """Initialise signed-distance function from the model setup of Schmalholz (2011)"""
    rectangle_lith = sl.Polygon(
        [(0, 6.6e5), (1e6, 6.6e5), (1e6, 5.8e5), (0, 5.8e5), (0, 6.6e5)]
    )
    sl.prepare(rectangle_lith)
    rectangle_slab = sl.Polygon(
        [
            (4.6e5, 5.8e5),
            (5.4e5, 5.8e5),
            (5.4e5, 3.3e5),
            (4.6e5, 3.3e5),
            (4.6e5, 5.8e5),
        ]
    )
    sl.prepare(rectangle_slab)
    polygon_lith = sl.union(rectangle_lith, rectangle_slab)
    sl.prepare(polygon_lith)

    interface_x = np.array([0, 4.6e5, 4.6e5, 5.4e5, 5.4e5, 1e6])
    interface_y = np.array([5.8e5, 5.8e5, 3.3e5, 3.3e5, 5.8e5, 5.8e5])
    curve = sl.LineString([*np.column_stack((interface_x, interface_y))])
    sl.prepare(curve)

    node_coords_x, node_coords_y = ga.node_coordinates(level_set)
    node_relation_to_curve = [
        (
            polygon_lith.contains(sl.Point(x, y))
            or polygon_lith.boundary.contains(sl.Point(x, y)),
            curve.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    return [dist if is_inside else -dist for is_inside, dist in node_relation_to_curve]


def isd_schmeling(parameters, level_set):
    """Initialise signed-distance function from the model setup of
    Schmeling et al. (2008)"""
    rectangle_lith = sl.Polygon(
        [(1e6, 7e5), (3e6, 7e5), (3e6, 6e5), (1e6, 6e5), (1e6, 7e5)]
    )
    sl.prepare(rectangle_lith)
    rectangle_slab = sl.Polygon(
        [(1e6, 6e5), (1.1e6, 6e5), (1.1e6, 5e5), (1e6, 5e5), (1e6, 6e5)]
    )
    sl.prepare(rectangle_slab)
    polygon_lith = sl.union(rectangle_lith, rectangle_slab)
    sl.prepare(polygon_lith)

    node_coords_x, node_coords_y = ga.node_coordinates(level_set)
    node_relation_to_curve = [
        (
            polygon_lith.contains(sl.Point(x, y))
            or polygon_lith.boundary.contains(sl.Point(x, y)),
            polygon_lith.boundary.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    return [dist if is_inside else -dist for is_inside, dist in node_relation_to_curve]
