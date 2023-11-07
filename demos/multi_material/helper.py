import firedrake as fd


def node_coordinates(function):
    """Extract mesh coordinates and interpolate them onto the relevant function space"""
    func_space = function.function_space()
    mesh_coords = fd.SpatialCoordinate(func_space.mesh())
    node_coords_x = fd.Function(func_space).interpolate(mesh_coords[0]).dat.data
    node_coords_y = fd.Function(func_space).interpolate(mesh_coords[1]).dat.data

    return node_coords_x, node_coords_y
