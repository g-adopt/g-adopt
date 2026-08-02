"""Utilities for assigning control-volume measures to mesh nodes."""

import firedrake as fd


def _validate_degree(degree):
    """Validate the supported nodal layout and return its integer degree."""
    if isinstance(degree, bool) or degree not in (1, 2):
        raise ValueError("degree must be either 1 or 2")
    return int(degree)


def _control_volume_space(mesh, degree):
    """Build a positive, low-order-refined space with CG-degree nodes."""
    if degree == 1:
        return fd.FunctionSpace(mesh, "CG", 1)

    return fd.FunctionSpace(mesh, "CG", 1, variant="equispaced,iso(2)")


def nodal_control_volumes(mesh, degree=2, name="Nodal control volume"):
    """Return the domain measure associated with every continuous-Galerkin node.

    The returned :class:`firedrake.Function` has the nodal layout of a degree-
    ``degree`` continuous Lagrange field. Its value at node ``i`` is the
    integral of the corresponding low-order-refined basis function over
    ``mesh``. Consequently, all values are non-negative and their sum is the
    measure of the mesh (length, area, or volume, depending on its topological
    dimension).

    For ``degree=2``, this is the P1-iso-P2 construction: piecewise-linear
    basis functions are integrated on the uniformly refined macro mesh, but
    the result is stored with the same nodes and ordering as a CG2 field.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh on which to compute the nodal control volumes. Both extruded and
        non-extruded meshes, including embedded manifold meshes, are supported.
    degree : int, optional
        Degree of the output nodal layout. Supported values are 1 and 2.
    name : str, optional
        Name assigned to the returned function.

    Returns
    -------
    firedrake.Function
        Nodal control-volume measures in a CG``degree`` function space.
    """
    degree = _validate_degree(degree)
    output_space = fd.FunctionSpace(mesh, "CG", degree)
    control_space = _control_volume_space(mesh, degree)

    control_volumes = fd.assemble(fd.TestFunction(control_space) * fd.dx)
    result = fd.Function(output_space, name=name)

    if result.dat.data_ro.shape != control_volumes.dat.data_ro.shape:
        raise RuntimeError(
            "The control-volume and output spaces do not have matching local "
            "nodal layouts."
        )

    result.dat.data[:] = control_volumes.dat.data_ro
    return result


def layerwise_nodal_control_volumes(
    mesh, degree=2, name="Layerwise nodal control volume"
):
    """Return horizontal control-volume measures at every node of an extruded mesh.

    Each horizontal nodal layer is treated as an independent manifold. The
    value at a node is its share of that layer's measure: length in a 2-D
    domain and area in a 3-D domain. Thus, values in every layer sum to that
    layer's total length or area. The actual coordinates of every layer are
    used, so the routine handles Cartesian and radially extruded cylindrical
    or spherical meshes, deformed meshes, and meshes loaded from checkpoints.

    This quantity is deliberately different from :func:`nodal_control_volumes`,
    which integrates over the full-dimensional domain. The returned function
    can be written directly, for example::

        cv_area_3d = layerwise_nodal_control_volumes(mesh)
        VTKFile("cv_area_3d.pvd").write(cv_area_3d)

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        An extruded mesh whose base mesh is one dimension lower.
    degree : int, optional
        Degree of the output nodal layout. Supported values are 1 and 2.
    name : str, optional
        Name assigned to the returned function.

    Returns
    -------
    firedrake.Function
        Layerwise nodal measures in a CG``degree`` function space on ``mesh``.

    Raises
    ------
    ValueError
        If ``mesh`` is not a non-periodically extruded mesh or ``degree`` is
        unsupported.
    NotImplementedError
        If ``mesh`` has a variable number of layers.
    """
    degree = _validate_degree(degree)
    if not mesh.extruded or mesh.layers is None:
        raise ValueError("layerwise nodal control volumes require an extruded mesh")
    if mesh.extruded_periodic:
        raise ValueError("layerwise nodal control volumes require non-periodic extrusion")
    if mesh.variable_layers:
        raise NotImplementedError(
            "layerwise nodal control volumes do not support variable-layer meshes"
        )

    output_space = fd.FunctionSpace(mesh, "CG", degree)
    coordinate_space = fd.VectorFunctionSpace(mesh, "CG", degree)
    coordinates = fd.Function(coordinate_space).interpolate(mesh.coordinates)

    # Firedrake stores the vertical degree of freedom as the fastest-varying
    # index on an extruded mesh. ``mesh.layers`` counts vertices, not cells.
    level_count = degree * (mesh.layers - 1) + 1
    coordinate_data = coordinates.dat.data_ro.reshape(
        (-1, level_count, mesh.geometric_dimension)
    )

    # Construct a temporary manifold with a coordinate degree matching the
    # requested output. This is necessary for Cartesian bases (whose original
    # coordinates are normally CG1) as well as curved quadratic bases.
    base_coordinate_space = fd.VectorFunctionSpace(
        mesh._base_mesh, "CG", degree, dim=mesh.geometric_dimension
    )
    layer_coordinates = fd.Function(base_coordinate_space, name="Layer coordinates")
    layer_coordinates.dat.data[:] = coordinate_data[:, 0, :]
    layer_mesh = fd.Mesh(layer_coordinates)
    control_space = _control_volume_space(layer_mesh, degree)
    control_volume_form = fd.TestFunction(control_space) * fd.dx

    result = fd.Function(output_space, name=name)
    result_data = result.dat.data.reshape((-1, level_count))

    if result_data.shape[0] != layer_coordinates.dat.data_ro.shape[0]:
        raise RuntimeError(
            "The base and extruded meshes do not have matching horizontal "
            "nodal layouts."
        )

    for level in range(level_count):
        layer_coordinates.dat.data[:] = coordinate_data[:, level, :]
        control_volumes = fd.assemble(control_volume_form)
        result_data[:, level] = control_volumes.dat.data_ro

    return result
