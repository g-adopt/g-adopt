import numpy as np
import shapely as sl

from helper import node_coordinates


def cosine_curve(x, amplitude, wavelength, vertical_shift):
    """Cosine curve with an amplitude and a vertical shift"""
    return amplitude * np.cos(2 * np.pi / wavelength * x) + vertical_shift


def initialise_signed_distance(level_set, benchmark, parameters):
    """Set up the initial signed distance function to the material interface"""
    node_coords_x, node_coords_y = node_coordinates(level_set)

    match benchmark:
        case (
            "van_Keken_1997_isothermal"
            | "van_Keken_1997_thermochemical"
            | "Robey_2019"
            | "Trim_2023"
        ):
            domain_length_x, layer_interface_y, interface_deflection = parameters

            interface_x = np.linspace(0, domain_length_x, 1000)
            interface_y = cosine_curve(
                interface_x,
                interface_deflection,
                2 * domain_length_x,
                layer_interface_y,
            )
            curve = sl.LineString([*np.column_stack((interface_x, interface_y))])
            sl.prepare(curve)

            node_relation_to_curve = [
                (
                    node_coord_y
                    > cosine_curve(
                        node_coord_x,
                        interface_deflection,
                        2 * domain_length_x,
                        layer_interface_y,
                    ),
                    curve.distance(sl.Point(node_coord_x, node_coord_y)),
                )
                for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y)
            ]
            node_sign_dist_to_interface = [
                dist if is_above else -dist for is_above, dist in node_relation_to_curve
            ]
        case "Gerya_2003":
            ref_vertex_x, ref_vertex_y, edge_length = parameters

            square = sl.Polygon(
                [
                    (ref_vertex_x, ref_vertex_y),
                    (ref_vertex_x + edge_length, ref_vertex_y),
                    (ref_vertex_x + edge_length, ref_vertex_y + edge_length),
                    (ref_vertex_x, ref_vertex_y + edge_length),
                    (ref_vertex_x, ref_vertex_y),
                ]
            )
            sl.prepare(square)

            node_relation_to_square = [
                (
                    square.contains(sl.Point(x, y))
                    or square.boundary.contains(sl.Point(x, y)),
                    square.boundary.distance(sl.Point(x, y)),
                )
                for x, y in zip(node_coords_x, node_coords_y)
            ]
            node_sign_dist_to_interface = [
                dist if is_inside else -dist
                for is_inside, dist in node_relation_to_square
            ]
        case "Schmalholz_2011":
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

            node_relation_to_curve = [
                (
                    polygon_lith.contains(sl.Point(x, y))
                    or polygon_lith.boundary.contains(sl.Point(x, y)),
                    curve.distance(sl.Point(x, y)),
                )
                for x, y in zip(node_coords_x, node_coords_y)
            ]
            node_sign_dist_to_interface = [
                dist if is_inside else -dist
                for is_inside, dist in node_relation_to_curve
            ]

    return node_sign_dist_to_interface
