"""Positive nodal measures for whole meshes and layers of extruded meshes.

Degree-two measures use Firedrake's P1-iso-P2 construction: linear basis
functions on a uniformly refined macro mesh share the nodal layout of CG2.
"""

from operator import index

import firedrake as fd
from mpi4py import MPI
import numpy as np


__all__ = ["nodal_control_volumes", "layerwise_nodal_control_volumes"]

_HOMOTHETY_TOLERANCE_FACTOR = 512
_HOMOTHETY_BLOCK_BYTES = 8 * 1024**2


def _validate_degree(degree):
    """Validate the supported nodal layout and return its integer degree."""
    if isinstance(degree, bool):
        raise ValueError("degree must be either 1 or 2")
    try:
        degree = index(degree)
    except TypeError as exc:
        raise ValueError("degree must be either 1 or 2") from exc
    if degree not in (1, 2):
        raise ValueError("degree must be either 1 or 2")
    return degree


def _control_volume_space(mesh, degree):
    """Build a positive, low-order-refined space with CG-degree nodes."""
    if degree == 1:
        return fd.FunctionSpace(mesh, "CG", 1)

    return fd.FunctionSpace(mesh, "CG", 1, variant="equispaced,iso(2)")


def _coordinate_block(
    source_coordinate_data, source_vertical_degree, target_degree, start, stop
):
    """Return target-degree coordinate levels for a block of horizontal nodes."""
    source_block = source_coordinate_data[start:stop]
    if source_vertical_degree == target_degree:
        return source_block
    if source_vertical_degree == 2 and target_degree == 1:
        return source_block[:, ::2, :]
    if source_vertical_degree == 1 and target_degree == 2:
        target_level_count = 2 * (source_block.shape[1] - 1) + 1
        target_block = np.empty(
            (source_block.shape[0], target_level_count, source_block.shape[2]),
            dtype=source_block.dtype,
        )
        target_block[:, ::2, :] = source_block
        target_block[:, 1::2, :] = 0.5 * (
            source_block[:, :-1, :] + source_block[:, 1:, :]
        )
        return target_block
    raise NotImplementedError(
        "layerwise nodal control volumes require vertical coordinate degree 1 or 2"
    )


def _coordinate_layer(
    source_coordinate_data, source_vertical_degree, target_degree, level
):
    """Return one target-degree coordinate layer without building all layers."""
    if source_vertical_degree == target_degree:
        return source_coordinate_data[:, level, :]
    if source_vertical_degree == 2 and target_degree == 1:
        return source_coordinate_data[:, 2 * level, :]
    if source_vertical_degree == 1 and target_degree == 2:
        source_level, is_midpoint = divmod(level, 2)
        if not is_midpoint:
            return source_coordinate_data[:, source_level, :]
        return 0.5 * (
            source_coordinate_data[:, source_level, :]
            + source_coordinate_data[:, source_level + 1, :]
        )
    raise NotImplementedError(
        "layerwise nodal control volumes require vertical coordinate degree 1 or 2"
    )


def _homothety_scales(
    source_coordinate_data, source_vertical_degree, target_degree, comm
):
    """Return scale factors for translated or homothetically scaled layers.

    A non-finite entry denotes a layer that is not a translated homothetic
    copy of the first layer to floating-point round-off and therefore needs an
    independent finite-element assembly. Global moments and errors ensure that
    every MPI rank makes the same decision.
    """
    local_node_count, source_level_count, geometric_dimension = (
        source_coordinate_data.shape
    )
    if (source_level_count - 1) % source_vertical_degree:
        raise RuntimeError("coordinate layout is inconsistent with its vertical degree")
    layer_count = (source_level_count - 1) // source_vertical_degree
    level_count = target_degree * layer_count + 1
    node_count = comm.allreduce(local_node_count, op=MPI.SUM)
    if node_count == 0:
        raise RuntimeError("cannot compute nodal measures on an empty mesh")

    bytes_per_horizontal_row = (
        level_count * geometric_dimension * source_coordinate_data.dtype.itemsize
    )
    rows_per_block = max(1, _HOMOTHETY_BLOCK_BYTES // bytes_per_horizontal_row)

    local_coordinate_sums = np.zeros(
        (level_count, geometric_dimension), dtype=source_coordinate_data.dtype
    )
    for start in range(0, local_node_count, rows_per_block):
        stop = min(start + rows_per_block, local_node_count)
        coordinate_block = _coordinate_block(
            source_coordinate_data,
            source_vertical_degree,
            target_degree,
            start,
            stop,
        )
        local_coordinate_sums += coordinate_block.sum(axis=0)
    coordinate_sums = np.empty_like(local_coordinate_sums)
    comm.Allreduce(local_coordinate_sums, coordinate_sums, op=MPI.SUM)
    layer_centres = coordinate_sums / node_count

    reference = source_coordinate_data[:, 0, :] - layer_centres[0]
    local_denominator = np.einsum("ij,ij->", reference, reference)
    denominator = comm.allreduce(local_denominator, op=MPI.SUM)

    scales = np.full(level_count, np.nan)
    scales[0] = 1.0
    if denominator == 0.0:
        return scales

    local_reference_sum = reference.sum(axis=0)
    local_numerators = np.zeros(level_count, dtype=source_coordinate_data.dtype)
    for start in range(0, local_node_count, rows_per_block):
        stop = min(start + rows_per_block, local_node_count)
        coordinate_block = _coordinate_block(
            source_coordinate_data,
            source_vertical_degree,
            target_degree,
            start,
            stop,
        )
        local_numerators += np.einsum(
            "ij,ilj->l", reference[start:stop], coordinate_block
        )
    local_numerators -= layer_centres @ local_reference_sum
    numerators = np.empty_like(local_numerators)
    comm.Allreduce(local_numerators, numerators, op=MPI.SUM)
    candidate_scales = numerators / denominator
    local_errors = np.zeros((level_count, geometric_dimension))
    local_sizes = np.zeros_like(local_errors)
    local_reference_sizes = np.max(np.abs(reference), axis=0, initial=0.0)

    # Process complete horizontal rows so reads are sequential. Each principal
    # coordinate/residual temporary is bounded by ``_HOMOTHETY_BLOCK_BYTES``.
    for start in range(0, local_node_count, rows_per_block):
        stop = min(start + rows_per_block, local_node_count)
        residual = _coordinate_block(
            source_coordinate_data,
            source_vertical_degree,
            target_degree,
            start,
            stop,
        ).copy()
        residual -= layer_centres[None, :, :]
        local_sizes = np.maximum(
            local_sizes, np.max(np.abs(residual), axis=0, initial=0.0)
        )
        residual -= (
            reference[start:stop, None, :]
            * candidate_scales[None, :, None]
        )
        local_errors = np.maximum(
            local_errors, np.max(np.abs(residual), axis=0, initial=0.0)
        )
    local_sizes = np.maximum(
        local_sizes,
        np.abs(candidate_scales[:, None]) * local_reference_sizes[None, :],
    )

    errors = np.empty_like(local_errors)
    sizes = np.empty_like(local_sizes)
    comm.Allreduce(local_errors, errors, op=MPI.MAX)
    comm.Allreduce(local_sizes, sizes, op=MPI.MAX)

    # Use a layer-wide extent so harmless round-off in a constant embedding
    # coordinate does not reject a Cartesian layer. Raw coordinate magnitudes
    # are deliberately excluded: they could hide deformation on a small mesh
    # located far from the origin. Ambiguous cases safely fall back to assembly.
    eps = np.finfo(source_coordinate_data.dtype).eps
    layer_errors = np.max(errors, axis=1)
    layer_sizes = np.max(sizes, axis=1)
    homothetic = (
        layer_errors
        <= _HOMOTHETY_TOLERANCE_FACTOR * eps * layer_sizes
    )
    scales[homothetic] = candidate_scales[homothetic]
    scales[0] = 1.0
    return scales


def _build_layer_mesh(mesh, horizontal_coordinate_degree, first_layer_coordinates):
    """Build a horizontal mesh using the first extruded coordinate layer."""
    # Firedrake currently has no public base-topology accessor. Prefer the
    # base MeshGeometry retained by a normal ExtrudedMesh, then fall back to
    # the base MeshTopology retained when a mesh is reconstructed from its
    # coordinate field. Keeping this private-API boundary here makes it easy
    # to update if Firedrake adds a public accessor.
    base_mesh = getattr(mesh, "_base_mesh", None)
    base_domain = (
        base_mesh if base_mesh is not None else getattr(mesh.topology, "_base_mesh", None)
    )
    if base_domain is None:
        raise ValueError("extruded mesh does not expose its base-mesh topology")

    base_coordinate_space = fd.VectorFunctionSpace(
        base_domain,
        "CG",
        horizontal_coordinate_degree,
        dim=mesh.geometric_dimension,
    )
    if base_mesh is None:
        coordinates = fd.CoordinatelessFunction(
            base_coordinate_space, name="Layer coordinates"
        )
    else:
        coordinates = fd.Function(
            base_coordinate_space, name="Layer coordinates"
        )

    coordinate_data = coordinates.dat.data
    if coordinate_data.shape != first_layer_coordinates.shape:
        raise NotImplementedError(
            "layerwise nodal control volumes do not support discontinuous "
            "horizontal coordinate layouts, such as radial-hedgehog extrusion"
        )
    coordinate_data[:] = first_layer_coordinates
    layer_mesh = fd.Mesh(coordinates)
    return layer_mesh, layer_mesh.coordinates


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

    # Firedrake's CG2 and equispaced P1-iso-P2 spaces deliberately share raw
    # node ordering; the shape check above guards the corresponding local
    # layout before copying values without interpolation.
    result.dat.data[:] = control_volumes.dat.data_ro
    return result


def layerwise_nodal_control_volumes(
    mesh, degree=2, name="Layerwise nodal control volume"
):
    """Return horizontal control-volume measures at every node of an extruded mesh.

    Each horizontal nodal layer is treated as a manifold. The value at a node
    is its share of the layer measure: length for a one-dimensional layer and
    area for a two-dimensional layer. Thus, values in every layer sum to that
    layer's total length or area. The actual coordinates of every layer are
    used, so the routine handles Cartesian and radially extruded cylindrical
    or spherical meshes with either increasing or decreasing radius, deformed
    meshes, and meshes loaded from checkpoints. Layers that are translated or
    homothetically scaled copies to floating-point round-off reuse the first
    layer's assembly; geometrically warped layers are assembled independently.

    The layer total is the measure of its discrete finite-element geometry.
    On curved circle and sphere meshes this converges to, but is not silently
    renormalised to, the analytical ``2*pi*r`` or ``4*pi*r**2`` measure.

    This quantity is deliberately different from :func:`nodal_control_volumes`,
    which integrates over the whole mesh. The returned function can be written
    directly, for example::

        from gadopt import VTKFile, layerwise_nodal_control_volumes
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
        If ``mesh`` has a variable number of layers, vertical coordinate
        degree other than 1 or 2, or a discontinuous horizontal coordinate
        layout such as radial-hedgehog extrusion.
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

    coordinate_degree = mesh.coordinates.ufl_element().degree()
    if isinstance(coordinate_degree, tuple):
        horizontal_coordinate_degree = coordinate_degree[0]
        source_vertical_degree = coordinate_degree[1]
    else:
        horizontal_coordinate_degree = coordinate_degree
        source_vertical_degree = 1
    source_level_count = source_vertical_degree * (mesh.layers - 1) + 1
    source_coordinate_data = mesh.coordinates.dat.data_ro.reshape(
        (-1, source_level_count, mesh.geometric_dimension)
    )

    # Firedrake stores the vertical degree of freedom as the fastest-varying
    # index on an extruded mesh. ``mesh.layers`` counts vertices, not cells.
    # The CG output and P1-iso-P2 control spaces also share horizontal node
    # ordering; regression tests cover the supported Firedrake cell layouts.
    level_count = degree * (mesh.layers - 1) + 1

    # Reconstruct a temporary horizontal manifold without lowering the source
    # geometry order when the requested output is CG1. A reconstructed
    # extruded MeshGeometry may retain only the base MeshTopology needed by
    # the coordinate space.
    layer_mesh, layer_coordinates = _build_layer_mesh(
        mesh,
        horizontal_coordinate_degree,
        _coordinate_layer(
            source_coordinate_data, source_vertical_degree, degree, 0
        ),
    )
    control_space = _control_volume_space(layer_mesh, degree)
    control_volume_form = fd.TestFunction(control_space) * fd.dx

    layer_dimension = layer_mesh.topological_dimension
    homothety_scales = _homothety_scales(
        source_coordinate_data, source_vertical_degree, degree, mesh.comm
    )

    control_volumes = fd.assemble(control_volume_form)
    reference_values = control_volumes.dat.data_ro.copy()

    # Coordinate levels are streamed from the mesh, so no full target-degree
    # coordinate copy overlaps the potentially large output allocation.
    result = fd.Function(output_space, name=name)
    result_data = result.dat.data.reshape((-1, level_count))
    if result_data.shape[0] != reference_values.shape[0]:
        raise RuntimeError(
            "The layer control-volume and extruded output spaces do not have "
            "matching horizontal nodal layouts."
        )

    all_layers_homothetic = np.all(np.isfinite(homothety_scales))
    if all_layers_homothetic:
        factors = np.abs(homothety_scales) ** layer_dimension
        np.multiply(reference_values[:, None], factors[None, :], out=result_data)
        return result

    result_data[:, 0] = reference_values

    for level in range(1, level_count):
        scale = homothety_scales[level]
        if np.isfinite(scale):
            result_data[:, level] = reference_values * abs(scale) ** layer_dimension
            continue

        layer_coordinates.dat.data[:] = _coordinate_layer(
            source_coordinate_data, source_vertical_degree, degree, level
        )
        fd.assemble(control_volume_form, tensor=control_volumes)
        result_data[:, level] = control_volumes.dat.data_ro

    return result
