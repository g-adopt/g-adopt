"""Utilities for assigning control-volume measures to mesh nodes."""

from operator import index

import firedrake as fd
from mpi4py import MPI
import numpy as np


_SIMILARITY_TOLERANCE = 1.0e-12
_SIMILARITY_WORKING_SET_BYTES = 8 * 1024**2


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


def _similarity_scales(coordinate_data, comm):
    """Return scale factors for layers that are translated/scaled copies.

    A non-finite entry denotes a layer that is not a similarity transform of
    the first layer and therefore needs an independent finite-element
    assembly. Global moments and errors are used so every MPI rank makes the
    same decision.
    """
    local_node_count, level_count, _ = coordinate_data.shape
    node_count = comm.allreduce(local_node_count, op=MPI.SUM)
    if node_count == 0:
        raise RuntimeError("cannot compute nodal measures on an empty mesh")

    local_coordinate_sums = coordinate_data.sum(axis=0)
    coordinate_sums = np.empty_like(local_coordinate_sums)
    comm.Allreduce(local_coordinate_sums, coordinate_sums, op=MPI.SUM)
    layer_centres = coordinate_sums / node_count

    reference = coordinate_data[:, 0, :] - layer_centres[0]
    local_denominator = np.einsum("ij,ij->", reference, reference)
    denominator = comm.allreduce(local_denominator, op=MPI.SUM)

    scales = np.full(level_count, np.nan)
    scales[0] = 1.0
    if denominator == 0.0:
        return scales

    local_reference_sum = reference.sum(axis=0)
    local_numerators = np.einsum("ij,ilj->l", reference, coordinate_data)
    local_numerators -= layer_centres @ local_reference_sum
    numerators = np.empty_like(local_numerators)
    comm.Allreduce(local_numerators, numerators, op=MPI.SUM)
    candidate_scales = numerators / denominator
    geometric_dimension = coordinate_data.shape[2]
    local_errors = np.zeros((level_count, geometric_dimension))
    local_sizes = np.zeros_like(local_errors)
    local_reference_sizes = np.max(np.abs(reference), axis=0, initial=0.0)

    # Process complete horizontal rows so reads are sequential, while keeping
    # the temporary centred/residual arrays bounded on large 3-D meshes.
    bytes_per_horizontal_row = (
        level_count * geometric_dimension * coordinate_data.dtype.itemsize
    )
    rows_per_block = max(
        1, _SIMILARITY_WORKING_SET_BYTES // bytes_per_horizontal_row
    )
    for start in range(0, local_node_count, rows_per_block):
        stop = min(start + rows_per_block, local_node_count)
        residual = coordinate_data[start:stop].copy()
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

    # Deliberately do not use raw coordinate magnitudes in the tolerance: for
    # a small mesh at a large offset that could hide a material deformation.
    # A numerically ambiguous similarity safely falls back to assembly.
    eps = np.finfo(coordinate_data.dtype).eps
    roundoff = 64 * eps * sizes
    similar = np.all(
        errors <= _SIMILARITY_TOLERANCE * sizes + roundoff, axis=1
    )
    scales[similar] = candidate_scales[similar]
    scales[0] = 1.0
    return scales


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

    Each horizontal nodal layer is treated as a manifold. The value at a node
    is its share of that layer's measure: length in a 2-D domain and area in a
    3-D domain. Thus, values in every layer sum to that layer's total length or
    area. The actual coordinates of every layer are used, so the routine
    handles Cartesian and radially extruded cylindrical or spherical meshes,
    deformed meshes, and meshes loaded from checkpoints.
    Translated or uniformly scaled layers reuse the first layer's assembly;
    geometrically warped layers are detected and assembled independently.

    The layer total is the measure of its discrete finite-element geometry.
    On curved circle and sphere meshes this converges to, but is not silently
    renormalised to, the analytical ``2*pi*r`` or ``4*pi*r**2`` measure.

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

    coordinate_degree = mesh.coordinates.ufl_element().degree()
    if isinstance(coordinate_degree, tuple):
        horizontal_coordinate_degree = coordinate_degree[0]
    else:
        horizontal_coordinate_degree = coordinate_degree
    coordinate_space = fd.VectorFunctionSpace(
        mesh,
        "CG",
        horizontal_coordinate_degree,
        vfamily="CG",
        vdegree=degree,
    )
    coordinates = fd.Function(coordinate_space).interpolate(mesh.coordinates)

    # Firedrake stores the vertical degree of freedom as the fastest-varying
    # index on an extruded mesh. ``mesh.layers`` counts vertices, not cells.
    level_count = degree * (mesh.layers - 1) + 1
    coordinate_data = coordinates.dat.data_ro.reshape(
        (-1, level_count, mesh.geometric_dimension)
    )

    # Reconstruct a temporary horizontal manifold without lowering the source
    # geometry order when the requested output is CG1.  A copied checkpoint
    # mesh may no longer expose ``_base_mesh`` as a MeshGeometry, but its
    # topology still retains the base MeshTopology needed by the space.
    base_mesh = mesh._base_mesh
    base_topology = (
        base_mesh if base_mesh is not None else getattr(mesh.topology, "_base_mesh", None)
    )
    if base_topology is None:
        raise ValueError("extruded mesh does not expose its base-mesh topology")
    base_coordinate_space = fd.VectorFunctionSpace(
        base_topology,
        "CG",
        horizontal_coordinate_degree,
        dim=mesh.geometric_dimension,
    )
    if base_mesh is None:
        topological_coordinates = fd.CoordinatelessFunction(
            base_coordinate_space, name="Layer coordinates"
        )
        topological_coordinates.dat.data[:] = coordinate_data[:, 0, :]
        layer_mesh = fd.Mesh(topological_coordinates)
        layer_coordinates = layer_mesh.coordinates
    else:
        layer_coordinates = fd.Function(
            base_coordinate_space, name="Layer coordinates"
        )
        layer_coordinates.dat.data[:] = coordinate_data[:, 0, :]
        layer_mesh = fd.Mesh(layer_coordinates)
    control_space = _control_volume_space(layer_mesh, degree)
    control_volume_form = fd.TestFunction(control_space) * fd.dx

    layer_dimension = layer_mesh.topological_dimension
    similarity_scales = _similarity_scales(coordinate_data, mesh.comm)

    control_volumes = fd.assemble(control_volume_form)
    reference_values = control_volumes.dat.data_ro.copy()

    # On the common similarity path, discard the potentially very large
    # interpolated coordinate array before allocating the result.
    all_layers_similar = np.all(np.isfinite(similarity_scales))
    if all_layers_similar:
        del coordinate_data
        del coordinates

    result = fd.Function(output_space, name=name)
    result_data = result.dat.data.reshape((-1, level_count))
    if result_data.shape[0] != reference_values.shape[0]:
        raise RuntimeError(
            "The layer control-volume and extruded output spaces do not have "
            "matching horizontal nodal layouts."
        )

    if all_layers_similar:
        factors = np.abs(similarity_scales) ** layer_dimension
        np.multiply(reference_values[:, None], factors[None, :], out=result_data)
        return result

    result_data[:, 0] = reference_values

    for level in range(1, level_count):
        scale = similarity_scales[level]
        if np.isfinite(scale):
            result_data[:, level] = reference_values * abs(scale) ** layer_dimension
            continue

        layer_coordinates.dat.data[:] = coordinate_data[:, level, :]
        fd.assemble(control_volume_form, tensor=control_volumes)
        result_data[:, level] = control_volumes.dat.data_ro

    return result
