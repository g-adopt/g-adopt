r"""This module provides several classes and functions to perform a number of pre-, syn-,
and post-processing tasks. Users incorporate utility as required in their code,
depending on what they would like to achieve.

"""

from firedrake import outer, ds_v, ds_t, ds_b, CellDiameter, CellVolume, dot, JacobianInverse
from firedrake import sqrt, Function, FiniteElement, TensorProductElement, FunctionSpace, VectorFunctionSpace
from firedrake import as_vector, SpatialCoordinate, Constant, max_value, min_value, dx, assemble, tanh
from firedrake import VectorElement, DirichletBC, interpolate, conditional
from firedrake.ufl_expr import extract_unique_domain
import pyop3 as op3
import ufl
import time
import weakref
from ufl.corealg.traversal import traverse_unique_terminals
from firedrake.petsc import PETSc
from firedrake.utils import get_device_type as fd_get_device_type
from functools import cached_property
from mpi4py import MPI
import numpy as np
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # NOQA
import os
from scipy.linalg import solveh_banded
from types import SimpleNamespace
from pyadjoint.tape import annotate_tape

try:
    from firedrake import MeshSequenceGeometry
except ImportError:
    MeshSequenceGeometry = None

# TBD: do we want our own set_log_level and use logging module with handlers?
log_level = logging.getLevelName(os.environ.get("GADOPT_LOGLEVEL", "INFO").upper())

valid_devices = ("HOST", "CUDA", "HIP")


def log(*args, level=None):
    """Log output to stdout from root processor only.

    If level is provided, the message is only printed when that level is
    enabled by the GADOPT_LOGLEVEL environment variable.
    """
    if level is not None and level < log_level:
        return
    PETSc.Sys.Print(*args)


class ParameterLog:
    def __init__(self, filename, mesh):
        self.comm = mesh.comm
        if self.comm.rank == 0:
            self.f = open(filename, 'w')

    def log_str(self, str):
        if self.comm.rank == 0:
            self.f.write(str + "\n")
            self.f.flush()

    def close(self):
        if self.comm.rank == 0:
            self.f.close()


class TimestepAdaptor:
    """Computes timestep based on CFL condition for provided velocity field

    Arguments:
      dt_const: Constant whose value will be updated by the timestep adaptor
      u: Velocity to base CFL condition on
      V: FunctionSpace for reference velocity, usually velocity space
      target_cfl: CFL number to target with chosen timestep
      increase_tolerance: Maximum tolerance timestep is allowed to change by
      maximum_timestep: Maximum allowable timestep

    """

    def __init__(self, dt_const, u, V, target_cfl=1.0, increase_tolerance=1.5, maximum_timestep=None):
        self.dt_const = dt_const
        self.u = u
        self.target_cfl = target_cfl
        self.increase_tolerance = increase_tolerance
        self.maximum_timestep = maximum_timestep
        self.mesh = V.mesh()

        # J^-1 u is a discontinuous expression, using op3.MAX_RW it takes the maximum value
        # in all adjacent elements when interpolating it to a continuous function space
        # We do need to ensure we reset ref_vel to zero, as it also takes the max with any previous values
        self.ref_vel_interpolate = interpolate(abs(dot(JacobianInverse(self.mesh), self.u)), V, access=op3.MAX_RW)

    def compute_timestep(self):
        max_ts = float(self.dt_const)*self.increase_tolerance
        if self.maximum_timestep is not None:
            max_ts = min(max_ts, self.maximum_timestep)

        # need to reset ref_vel to avoid taking max with previous values
        ref_vel = assemble(self.ref_vel_interpolate)
        local_maxrefvel = ref_vel.dat.data.max()
        max_refvel = self.mesh.comm.allreduce(local_maxrefvel, MPI.MAX)
        # NOTE; we're incorparating max_ts here before dividing by max. ref. vel. as it may be zero
        ts = self.target_cfl / max(max_refvel, self.target_cfl / max_ts)

        return ts

    def update_timestep(self):
        self.dt_const.assign(self.compute_timestep())
        return float(self.dt_const)


def is_cartesian(mesh):
    if MeshSequenceGeometry is not None and isinstance(mesh, MeshSequenceGeometry):
        return mesh.unique().cartesian
    else:
        return mesh.cartesian


def upward_normal(mesh):
    if is_cartesian(mesh):
        n = mesh.geometric_dimension
        return as_vector([0]*(n-1) + [1])
    else:
        X = SpatialCoordinate(mesh)
        r = sqrt(sum([x ** 2 for x in X]))
        return X/r


def vertical_component(u):
    mesh = extract_unique_domain(u)

    if is_cartesian(mesh):
        return u[u.ufl_shape[0]-1]
    else:
        n = upward_normal(mesh)
        return sum([n_i * u_i for n_i, u_i in zip(n, u)])


def horizontal_components(u):
    """Returns expression for vector u with vertical component removed, i.e. returns the vector UFL expression orthogonal to gravity."""
    mesh = extract_unique_domain(u)
    return u - vertical_component(u) * upward_normal(mesh)


def ensure_constant(f):
    if isinstance(f, float) or isinstance(f, int):
        return Constant(f)
    else:
        return f


def get_device_type(gpu_options: dict | None) -> str | None:
    """Determine the available GPU device type.

    Checks for a GPU device in the following order of priority:
    1. The `GADOPT_DEVICE_OVERRIDE` environment variable.
    2. The `device_type` key in the `gpu_options` dictionary.
    3. Firedrake's `get_device_type()` function.

    Returns:
        The detected device type (e.g., `CUDA`, `HIP`). Returns `None` if the `HOST`
        device is detected (i.e. no GPU) or if the device set in `gpu_options` or
        returned by Firedrake's `get_device_type()` is `None`.

    Raises:
        ValueError: If `GADOPT_DEVICE_OVERRIDE` or `gpu_options["device_type"]`
        specifies a device type not in `valid_devices`.
    """
    if "GADOPT_DEVICE_OVERRIDE" in os.environ:
        device_type = os.environ["GADOPT_DEVICE_OVERRIDE"]
        if device_type not in valid_devices:
            raise ValueError(
                f"Invalid device specified in GADOPT_DEVICE_OVERRIDE. Valid values are {','.join(valid_devices)}"
            )
    elif gpu_options is not None and "device_type" in gpu_options:
        device_type = gpu_options["device_type"]
        if device_type is not None and device_type not in valid_devices:
            raise ValueError(
                f"Invalid device specified in 'device_type' key. Valid values are {','.join(valid_devices)} or None"
            )
    else:
        # Set the PETSc error handler so that Firedrake can correctly intercept the
        # exception from PETSc if a GPU-enabled build calls this on a non-GPU enabled
        # system - add this to Firedrake. This function returns None for a GPU-enabled
        # build or HOST on a non-GPU enabled build when GPUs are not available.
        PETSc.Sys.pushErrorHandler("python")
        device_type = fd_get_device_type()
        PETSc.Sys.popErrorHandler()
    if device_type == "HOST":
        return None
    return device_type


class CombinedSurfaceMeasure(ufl.Measure):
    """
    A surface measure that combines ds_v, the integral over vertical boundary facets, and ds_t and ds_b,
    the integral over horizontal top and bottom facets. The vertical boundary facets are identified with
    the same surface ids as ds_v. The top and bottom surfaces are identified via the "top" and "bottom" ids."""

    def __init__(self, domain, degree):
        self.ds_v = ds_v(domain=domain, degree=degree)
        self.ds_t = ds_t(domain=domain, degree=degree)
        self.ds_b = ds_b(domain=domain, degree=degree)

    def __call__(self, subdomain_id, **kwargs):
        if subdomain_id == 'top':
            return self.ds_t(**kwargs)
        elif subdomain_id == 'bottom':
            return self.ds_b(**kwargs)
        else:
            return self.ds_v(subdomain_id, **kwargs)

    def __rmul__(self, other):
        """This is to handle terms to be integrated over all surfaces in the form of other*ds.
        Here the CombinedSurfaceMeasure ds is not called, instead we just split it up as below."""
        return other*self.ds_v + other*self.ds_t + other*self.ds_b


def _get_element(ufl_or_element):
    if isinstance(ufl_or_element, ufl.indexed.Indexed):
        expr, multiindex = ufl_or_element.ufl_operands
        V = expr.ufl_function_space()
        for i in multiindex:
            comp_to_subspace = tuple(j for j, W in enumerate(V) for k in range(W.value_size))
            V = V.sub(comp_to_subspace[int(i)])
        ufl_or_element = V

    if isinstance(ufl_or_element, ufl.AbstractFiniteElement):
        return ufl_or_element
    else:
        return ufl_or_element.ufl_element()


def is_continuous(expr):
    if isinstance(expr, ufl.tensors.ListTensor):
        return all(is_continuous(x) for x in expr.ufl_operands)

    elem = _get_element(expr)

    return elem in ufl.H1


def depends_on(ufl_expr, terminal):
    """Does ufl_expr depend on terminal (Function/Constant/...)?"""
    return terminal in traverse_unique_terminals(ufl_expr)


def normal_is_continuous(expr):
    # if we get some list expression, we can't guarantee its normal is continuous
    # unless all components are
    if isinstance(expr, ufl.tensors.ListTensor):
        return is_continuous(expr)

    elem = _get_element(expr)

    return elem in ufl.HDiv


def cell_size(mesh):
    if hasattr(mesh.ufl_cell(), 'sub_cells'):
        return CellVolume(mesh) ** (1/mesh.topological_dimension)
    else:
        return CellDiameter(mesh)


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    $$"jump"(bb u, bb n) = (bb u^+ bb n^+) + (bb u^- bb n^-)$$

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator `ufl.jump` which represents div(u).
    The equivalent of nabla_grad(u) is given by tensor_jump(n, u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


def extend_function_to_3d(func, mesh_extruded):
    """Returns a view of a `Function` on the base mesh as a `Function` on the extruded mesh.

    The extended function lives in the V x R space, where V is the space of
    `func`. Its horizontal element is the element of `func`, and it shares its
    data with `func`: a change to one shows in the other. The source must be a
    scalar or vector continuous (H1) function on the base mesh of `mesh_extruded`.
    Mixed spaces, their subspaces and restricted spaces are not supported.

    Arguments:
      func: Source `Function` on the base mesh.
      mesh_extruded: Mesh extruded from the mesh of `func`.

    Returns:
      The extended `Function`. Its `source` attribute is `func`.

    Raises:
      ValueError: If `func` does not satisfy the conditions above.
    """
    fs = func.function_space()
    ufl_elem = fs.ufl_element()
    is_vector = isinstance(ufl_elem, VectorElement)
    scalar_elem = ufl_elem.sub_elements[0] if is_vector else ufl_elem

    # These conditions depend only on the function space, so they are the same on every rank.
    # A subspace of a mixed space shares the buffer of the whole mixed function, and a
    # restricted space numbers its nodes differently from the unrestricted V x R space.
    if len(fs) != 1 or fs.parent is not None or fs.boundary_set:
        raise ValueError("extend_function_to_3d: the source must not be a mixed, sub or restricted space")
    if fs.mesh().topology is not mesh_extruded.topology._base_mesh:
        raise ValueError("extend_function_to_3d: the source mesh must be the base mesh of mesh_extruded")
    # The V x R space numbers its nodes in the same order as V only for continuous elements.
    if ufl_elem.sobolev_space != ufl.H1:
        raise ValueError(f"extend_function_to_3d: unsupported element {ufl_elem}; use a continuous element")

    # Use the source element itself in the horizontal, so that both spaces have the same dofs.
    r0 = FiniteElement("R", mesh_extruded.ufl_cell().sub_cells[1], 0)
    elem = TensorProductElement(scalar_elem, r0)
    if is_vector:
        fs_extended = VectorFunctionSpace(mesh_extruded, elem, dim=ufl_elem.num_sub_elements)
    else:
        fs_extended = FunctionSpace(mesh_extruded, elem)

    # The buffer of the source must have the size and the halo exchange pattern (star forest) of
    # the extended space. Both can differ between ranks, so all ranks agree before any raises.
    # The size test comes first, and np.array_equal is used, because StarForest.__eq__ fails on
    # arrays of different lengths.
    sf, sf_extended = func.dat.buffer.sf, fs_extended.axes.unindexed.sf
    ok = func.dat.buffer.size == fs_extended.axes.unindexed.local_max_size and type(sf) is type(sf_extended)
    if ok and isinstance(sf, op3.sf.StarForest) and sf is not sf_extended:
        ok = np.array_equal(sf.ilocal, sf_extended.ilocal) and np.array_equal(sf.iremote, sf_extended.iremote)
    if not mesh_extruded.comm.allreduce(ok, op=MPI.LAND):
        raise ValueError("extend_function_to_3d: the source and extended spaces have different data layouts")

    # A view with the axes of the extended space on the buffer of the source.
    func_extended = Function(fs_extended, name=func.name(), val=func.dat.with_axes(fs_extended.axes))
    func_extended.source = func
    return func_extended


class ExtrudedFunction(Function):
    """A 2D `Function` that provides a 3D view on the extruded domain.

    The 3D function can be accessed as `ExtrudedFunction.view_3d`.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.

    Arguments:
      mesh_3d: Extruded 3D mesh where the function will be extended to.

    """

    def __init__(self, *args, mesh_3d=None, **kwargs):
        # create the 2d function
        super().__init__(*args, **kwargs)
        if mesh_3d is not None:
            self.view_3d = extend_function_to_3d(self, mesh_3d)


def get_functionspace(mesh, h_family, h_degree, v_family=None, v_degree=None,
                      vector=False, hdiv=False, variant=None, v_variant=None,
                      **kwargs):
    cell_dim = mesh.cell_dimension()
    assert cell_dim in [2, (2, 1), (1, 1)], 'Unsupported cell dimension'
    hdiv_families = [
        'RT', 'RTF', 'RTCF', 'RAVIART-THOMAS',
        'BDM', 'BDMF', 'BDMCF', 'BREZZI-DOUGLAS-MARINI',
    ]
    if variant is None:
        if h_family.upper() in hdiv_families:
            if h_family in ['RTCF', 'BDMCF']:
                variant = 'equispaced'
            else:
                variant = 'integral'
        else:
            variant = 'equispaced'
    if v_variant is None:
        v_variant = 'equispaced'
    # Extruded meshes (2-D or 1-D base) take a tensor-product element.
    if cell_dim in ((2, 1), (1, 1)):
        if v_family is None:
            v_family = h_family
        if v_degree is None:
            v_degree = h_degree
        h_cell, v_cell = mesh.ufl_cell().sub_cells
        h_elt = FiniteElement(h_family, h_cell, h_degree, variant=variant)
        v_elt = FiniteElement(v_family, v_cell, v_degree, variant=v_variant)
        elt = TensorProductElement(h_elt, v_elt)
        if hdiv:
            elt = ufl.HDivElement(elt)
    else:
        elt = FiniteElement(h_family, mesh.ufl_cell(), h_degree, variant=variant)

    constructor = VectorFunctionSpace if vector else FunctionSpace
    return constructor(mesh, elt, **kwargs)


class LayerAveraging:
    """A manager for computing a vertical profile of horizontal layer averages.

    Arguments:
      mesh: The mesh over which to compute averages
      r1d: An array of either depth coordinates or radii,
             at which to compute layer averages. If not provided, and
             mesh is extruded, it uses the same layer heights. If mesh
             is not extruded, r1d is required.

    """

    def __init__(self, mesh, r1d=None, quad_degree=None):
        self.mesh = mesh
        XYZ = SpatialCoordinate(mesh)

        if is_cartesian(mesh):
            self.r = XYZ[len(XYZ)-1]
        else:
            self.r = sqrt(dot(XYZ, XYZ))

        self.dx = dx
        if quad_degree is not None:
            self.dx = dx(degree=quad_degree)

        if r1d is not None:
            self.r1d = r1d
        else:
            try:
                nlayers = mesh.layers
            except AttributeError:
                raise ValueError("For non-extruded mesh need to specify depths array r1d.")
            CG1 = FunctionSpace(mesh, "CG", 1)
            r_func = Function(CG1).interpolate(self.r)
            self.r1d = r_func.dat.data[:nlayers]

        self.mass = np.zeros((2, len(self.r1d)))
        self.rhs = np.zeros(len(self.r1d))
        self._assemble_mass()

    def _assemble_mass(self):
        # main diagonal of mass matrix
        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.)

        # radial P1 hat function in rp < r < rn with maximum at rc
        phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)

        for i, rin in enumerate(self.r1d[1:]):
            rn.assign(rin)
            self.mass[0, i] = assemble(phi**2 * self.dx)

            # shuffle coefficients for next iteration
            rp.assign(rc)
            rc.assign(rn)

        phi = max_value(min_value(1, (r - rp) / (rn - rp)), 0)
        self.mass[0, -1] = assemble(phi**2 * self.dx)

        # compute off-diagonal (symmetric)
        rp = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])

        # overlapping product between two basis functions in rp < r < rn
        overlap = max_value((rn - r) / (rn - rp), 0) * max_value((r - rp) / (rn - rp), 0) * self.dx

        for i, rin in enumerate(self.r1d[1:]):
            rn.assign(rin)
            self.mass[1, i] = assemble(overlap)

            # shuffle coefficients for next iteration
            rp.assign(rn)

    def _assemble_rhs(self, T):
        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.)

        phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)

        for i, rin in enumerate(self.r1d[1:]):
            rn.assign(rin)
            self.rhs[i] = assemble(phi * T * self.dx)

            rp.assign(rc)
            rc.assign(rn)

        phi = max_value(min_value(1, (r - rp) / (rn - rp)), 0)
        self.rhs[-1] = assemble(phi * T * self.dx)

    def get_layer_average(self, T):
        """Compute the layer averages of `Function` T at the predefined depths.

        Returns:
          A numpy array containing the averages.
        """

        self._assemble_rhs(T)
        return solveh_banded(self.mass, self.rhs, lower=True)

    def extrapolate_layer_average(self, u, avg):
        """Given an array of layer averages avg, extrapolate to `Function` u
        """

        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.)

        u.assign(0.0)

        phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)
        val = Constant(0.)

        for a, rin in zip(avg[:-1], self.r1d[1:]):
            val.assign(a)
            rn.assign(rin)
            # reconstruct this layer according to the basis function
            u.interpolate(u + val * phi)

            rp.assign(rc)
            rc.assign(rn)

        phi = max_value(min_value(1, (r - rp) / (rn - rp)), 0)
        val.assign(avg[-1])
        u.interpolate(u + val * phi)


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log(f"Time taken for {func.__name__}: {elapsed_time} seconds")
        return result
    return wrapper


class InteriorBC(DirichletBC):
    """DirichletBC applied to anywhere that is *not* on the specified boundary"""
    @cached_property
    def _nodes(self):
        # `node_count` includes the block size for vector and tensor spaces,
        # so take the local size (with ghosts) of the node axis.
        num_nodes = self._function_space.nodes.local_size
        return np.setdiff1d(np.arange(num_nodes), super()._nodes)


def absv(u):
    """Component-wise absolute value of vector for SU stabilisation"""
    return as_vector([abs(ui) for ui in u])


def step_func(r, centre, mag, increasing=True, sharpness=50):
    # A step function designed to design viscosity jumps
    # Build a step centred at "centre" with given magnitude
    # Increase with radius if "increasing" is True
    return mag * (
        0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
    )


def node_coordinates(function: Function) -> Function:
    """Interpolates mesh coordinates at each node of the provided function.

    Args:
      function: A Firedrake function

    Returns:
      A Firedrake function for the interpolated mesh coordinates
    """
    vec_space = VectorFunctionSpace(function.ufl_domain(), function.ufl_element())

    return Function(vec_space).interpolate(SpatialCoordinate(function))


def interpolate_1d_profile(function: Function, one_d_filename: str):
    """
    Assign a one-dimensional profile to a Function `function` from a file.

    The function reads a one-dimensional profile (e.g., viscosity) together with the
    radius/height input for the profile, from a file, broadcasts the two arrays to all
    processes, and then interpolates this array onto the function space of `function`.

    Args:
        function: The function onto which the 1D profile will be assigned
        one_d_filename: The path to the file containing the 1D radial profile

    Note:
        - This is designed to read a file with one process and distribute in parallel with MPI.
        - The input file should contain an array of radius/height and an array of values, separated by a comma.
    """
    mesh = extract_unique_domain(function)

    if mesh.comm.rank == 0:
        rshl, one_d_data = np.loadtxt(one_d_filename, unpack=True, delimiter=",")
        if rshl[1] < rshl[0]:
            rshl = rshl[::-1]
            one_d_data = one_d_data[::-1]
        if not np.all(np.diff(rshl) > 0):
            raise ValueError("Data must be strictly monotonous.")
    else:
        one_d_data = None
        rshl = None

    one_d_data = mesh.comm.bcast(one_d_data, root=0)
    rshl = mesh.comm.bcast(rshl, root=0)

    X = SpatialCoordinate(mesh)

    upward_coord = vertical_component(X)

    rad = Function(function.function_space()).interpolate(upward_coord)

    averager = LayerAveraging(mesh, rshl if mesh.layers is None else None)
    interpolated_visc = np.interp(averager.get_layer_average(rad), rshl, one_d_data)
    averager.extrapolate_layer_average(function, interpolated_visc)


class BoundaryIDNamespace(SimpleNamespace):
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, item):
        return item in self.values()

    def __iter__(self):
        return self.values().__iter__()


def get_boundary_ids(mesh) -> BoundaryIDNamespace:
    # PETSc creates these labels when loading meshes from files, Firedrake imitates it
    # in its own mesh creation functions.

    if mesh.topology_dm.hasLabel("Face Sets"):
        axis_extremes_order = [["left", "right"], ["bottom", "top"]]
        dim = mesh.geometric_dimension
        plex_dim = mesh.topology_dm.getCoordinateDim()  # In an extruded mesh, this is different to the
        # firedrake-assigned geometric_dimension
        if dim == 3:
            # For 3D meshes, we label dim[1] (y) as "front", "back" and dim[2] (z) as "bottom","top"
            axis_extremes_order.insert(1, ["front", "back"])
        bounding_box = mesh.topology_dm.getBoundingBox()
        boundary_tol = [abs(dim[1] - dim[0]) * 1e-6 for dim in bounding_box]
        if dim > 3:
            raise ValueError(f"Cannot handle {dim} dimensional mesh")
        # Recover the boundary ids Firedrake has inserted
        coords = mesh.topology_dm.getCoordinatesLocal()
        coord_sec = mesh.topology_dm.getCoordinateSection()
        mesh.topology_dm.markBoundaryFaces("TEMP_LABEL", 1)
        if mesh.topology_dm.getStratumSize("TEMP_LABEL", 1) > 0:
            boundary_cells = [
                (i, mesh.topology_dm.getLabelValue("Face Sets", i))
                for i in mesh.topology_dm.getStratumIS("TEMP_LABEL", 1).getIndices()
            ]
        else:
            boundary_cells = []
        identified = dict.fromkeys([i[1] for i in boundary_cells])
        nvert = -1
        for cell, face_id in boundary_cells:
            # -1 marks a boundary face that is not in "Face Sets"
            if face_id != -1 and identified[face_id] is None:
                face_coords = mesh.topology_dm.vecGetClosure(coord_sec, coords, cell)
                if nvert == -1:
                    nvert = len(face_coords) // plex_dim
                # Flattened version of axis_extremes_order
                tmp_bdys = [
                    axis_extremes_order[idim][ibdy]
                    for idim in range(plex_dim)
                    for ibdy in range(2)
                    if all(
                        abs(face_coords[idim + plex_dim * i] - bounding_box[idim][ibdy]) < boundary_tol[idim]
                        for i in range(nvert)
                    )
                ]
                # Found a cell unambiguously on one boundary
                if len(tmp_bdys) == 1:
                    identified[face_id] = tmp_bdys[0]
        # Not every MPI rank has every boundary of the mesh, gather all boundaries
        # seen by all MPI ranks
        gathered_boundaries = mesh.comm.allgather(identified)
        mesh.topology_dm.removeLabel("TEMP_LABEL")
        kwargs = {
            proc_bdy.get(face_id): face_id  # invert boundary mapping
            for proc_bdy in reversed(gathered_boundaries)  # keep value on lowest comm rank
            for face_id in set().union(*gathered_boundaries)  # all gathered face_ids
        }
        kwargs.pop(None, None)  # remove None entry
        # Firedrake names the top and bottom of an extruded mesh "top" and "bottom".
        # Use mesh.extruded, not the local labels, so that all ranks agree.
        if mesh.extruded:
            kwargs["top"] = "top"
            kwargs["bottom"] = "bottom"
        # Get remaining dimensions (if any)
        for idim in range(plex_dim, dim):
            for axis_label in axis_extremes_order[idim]:
                kwargs[axis_label] = axis_label
    # Integer subdomain meshes are only assigned by petsc or the firedrake utility mesh
    # module. If the "Face Sets" label is missing from the mesh, it was not created by
    # either of these and therefore will only have the "bottom" and "top" labels
    # utilised by 'CombinedSurfaceMeasure' above
    else:
        kwargs = {"bottom": "bottom", "top": "top"}
    return BoundaryIDNamespace(**kwargs)


def extruded_layer_heights(
        DG0_layers: int | list[int], radii: list[float]) -> list[float]:

    """Calculates layer heights for extruded mesh using rheological boundary

    Args:
      DG0_layers:
        Number of layers per rheological layer
      radii:
        Radii of rheological boundaries to match when extruding the mesh.
        Assumes sequence starts with outer radii -> inner radii.

    Returns:
        list of layer heights
        """

    layer_heights = []

    nz_layers = [DG0_layers] * len(radii) if isinstance(DG0_layers, int) else DG0_layers
    assert len(nz_layers) == len(radii)

    # starting at the bottom radius, work outwards
    for i in range(len(radii) - 2, -1, -1):
        dz = (radii[i] - radii[i + 1]) / nz_layers[i]
        for _ in range(nz_layers[i]):
            layer_heights.append(dz)
    return layer_heights


def initialise_background_field(
        f: Function,
        background_values: list[float],
        X: ufl.geometry.SpatialCoordinate,
        radii: list[float],
        shift: float = 0.0,):
    """Initialises discontinuous field with sharp jumps at rheological boundaries

    Args:
      f:
        Function to interpolate background values into. N.b. `f` is modified in place.
      background_values:
        Discrete values to interpolate into `f`
      X:
        Spatial coordinates associated with function `f`
      radii:
        Radii of rheological discontinuities. N.b. length of `background_values` must be one less
        than length of `radii`.
      shift:
        Shift vertical radii by a constant, e.g. when mesh is offset from `radii` values.
    """

    assert len(background_values) == len(radii)-1
    for i in range(len(background_values)):
        f.interpolate(
            conditional(vertical_component(X) + shift > radii[i+1],
                        conditional(vertical_component(X) + shift <= radii[i],
                                    background_values[i], f), f))


# -- Keywords of the adjoint and tangent solvers of an annotated variational solve --
#
# Two Firedrakes are in use. On Firedrake `main` an annotated variational solve
# puts a `NonlinearVariationalSolveBlock` on the tape, and the keywords of its
# adjoint solver are given to `solve` as `adj_kwargs`. Firedrake PR 4638 ("NLVS:
# cache more adjoint solvers/forms and fix Hessian") puts a `CachedSolverBlock`
# on the tape instead. Its forward, adjoint and tangent solvers are built once,
# at the first annotated solve, from a namespace that the solver constructor
# fills (`solver._ad_args_kwargs`), and `solve` refuses `adj_kwargs`. The
# constructor of that branch does not accept `adj_kwargs` or `tlm_kwargs`
# either: it removes them from the keywords only after the solver has been
# initialised, so they raise `TypeError`. The namespace is therefore the only
# place where these keywords can be given on that Firedrake. The write into the
# private `solver._ad_args_kwargs` is to be replaced by passing `adj_kwargs`
# and `tlm_kwargs` to the solver constructor once firedrakeproject/firedrake
# #5666 ("Take adj/tlm kwargs out before NonlinearVariationalSolver.__init__")
# is merged into Firedrake PR #4638.
try:
    from firedrake.adjoint_utils.blocks import CachedSolverBlock  # noqa: F401
    CACHED_SOLVER_BLOCKS = True
except ImportError:
    CACHED_SOLVER_BLOCKS = False

#: The names under which Firedrake PR 4638 caches the replay, tangent and
#: adjoint solvers of a variational solver (`functools.cached_property`, so
#: each one is in the instance `__dict__` once it is built).
_DERIVATIVE_SOLVER_CACHES = ("_ad_forward_cache", "_ad_tangent_cache",
                             "_ad_adjoint_cache", "_ad_hessian_cache")


#: The attribute `derivative_solve_keywords` sets on a solver once it has
#: written the derivative keywords into the solver's namespace.
_DERIVATIVE_KEYWORDS_WRITTEN = "_gadopt_derivative_keywords_written"


def derivative_solve_keywords(solver, adj_kwargs: dict,
                              tlm_kwargs: dict | None = None) -> dict:
    """Give the adjoint (and tangent) solver keywords to the running Firedrake.

    Call this before every solve of `solver` and pass the result to `solve`.
    Nothing is done unless the tape is recording, because only an annotated
    solve builds derivative solvers.

    With the stock solve block (Firedrake `main`) the adjoint keywords go to
    `solve`, which takes them on the first annotated solve and ignores them
    after that. The stock block builds its tangent solver itself, so
    `tlm_kwargs` is not used there.

    With cached solver blocks (Firedrake PR 4638) the keywords are written into
    the namespace from which the solver builds its cached adjoint and tangent
    solvers, and `solve` gets nothing. They are written only while those
    solvers do not exist yet, which is the same rule as on the stock path: the
    first annotated solve decides. `tlm_kwargs` defaults to a copy of
    `adj_kwargs`, which is what Firedrake does when only `adj_kwargs` is given
    to the constructor. A solver whose cached derivative solvers exist but
    were not built from these keywords was solved under annotation before
    this function saw it; its adjoint would run with the forward Newton
    options and without the application context, so that raises.

    Args:
      solver: The `NonlinearVariationalSolver` (or subclass) to be solved. An
        object without Firedrake's cached-solver namespace (a test stand-in
        that solves by hand) builds no derivative solvers and gets nothing.
      adj_kwargs: Constructor keywords of the adjoint `LinearVariationalSolver`.
      tlm_kwargs: Constructor keywords of the tangent `LinearVariationalSolver`
        (cached solver blocks only).

    Returns:
      The keywords to pass to `solver.solve`.

    Raises:
      RuntimeError: With cached solver blocks, if the derivative solvers of
        `solver` already exist and were built without these keywords.
    """
    if not annotate_tape():
        return {}
    if not CACHED_SOLVER_BLOCKS:
        return {"adj_kwargs": adj_kwargs}
    namespace = getattr(solver, "_ad_args_kwargs", None)
    if namespace is None:
        return {}
    built = vars(solver)
    if any(name in built for name in _DERIVATIVE_SOLVER_CACHES):
        if not built.get(_DERIVATIVE_KEYWORDS_WRITTEN, False):
            raise RuntimeError(
                "The cached adjoint and tangent solvers of this solver already "
                "exist and were built without the G-ADOPT derivative keywords: "
                "an annotated solve of the inner Firedrake solver ran before "
                "the G-ADOPT solve. Its adjoint would use the forward Newton "
                "options and no application context. Solve through the G-ADOPT "
                "solver (its `solve` method) from the first annotated solve on.")
        return {}
    # A write into a private namespace of Firedrake. To be replaced by
    # constructor keywords once firedrakeproject/firedrake#5666 is merged
    # into Firedrake PR #4638.
    namespace.adj_kwargs = dict(adj_kwargs)
    namespace.tlm_kwargs = dict(adj_kwargs if tlm_kwargs is None else tlm_kwargs)
    setattr(solver, _DERIVATIVE_KEYWORDS_WRITTEN, True)
    return {}


def weak_callback(method):
    """A PETSc callback that calls the bound `method` through a weak reference.

    A Firedrake solver keeps its callbacks (`pre_jacobian_callback`,
    `post_function_callback` and the others) in its `_SNESContext`. A Python
    preconditioner context, `firedrake.AssembledPC` for example, also holds
    that `_SNESContext`, and PETSc holds the Python context from C. A bound
    method holds its object strongly. If that object also holds the solver,
    the references form a cycle that goes through C. Python's garbage
    collector cannot see a reference held by C, so the solver, its PETSc
    objects and every matrix factorisation in them stay alive to the end of
    the process.

    The callback that this function returns holds only a `weakref.WeakMethod`
    to `method`. The cycle then has no strong link back to the object, and
    the object is collected when nothing else holds it. The owner of the
    callbacks (the user's solver object, or a tape block that holds it) must
    hold the object while the PETSc solver is in use.

    Args:
      method: A bound method.

    Returns:
      A function with the same arguments and return value as `method`. Its
      attribute `target` is the `weakref.WeakMethod` of `method`.

    Raises:
      ReferenceError: When the returned function is called after the object
        of `method` was collected.
    """
    reference = weakref.WeakMethod(method)
    name = method.__qualname__

    def callback(*args, **kwargs):
        bound = reference()
        if bound is None:
            raise ReferenceError(
                f"The callback {name} was called after its object was "
                "garbage-collected. Keep the solver that owns it alive while "
                "its PETSc solvers are in use.")
        return bound(*args, **kwargs)

    callback.__qualname__ = f"weak_callback({name})"
    #: The `weakref.WeakMethod`; calling it gives the bound method, or None
    #: once its object is gone. A test can check which method a solver
    #: carries as its callback with `callback.target() == obj.method`.
    callback.target = reference
    return callback


def interior_facet_integrand(expr):
    """Restrict a quantity of a mesh on that mesh's interior facets: `avg(expr)`.

    UFL's rule: in an integral over interior facets, every `Argument` and
    every `Coefficient` of a discontinuous element must carry a restriction
    (`'+'`, `'-'` or `avg`); a coefficient of a continuous element is given a
    default side. A `Real` field counts as continuous, so a `Real`
    coefficient is accepted unrestricted in a form, but the derivative with
    respect to it is a `Real` `Argument`, which UFL refuses unrestricted
    (`Discontinuous type Argument must be restricted`). pyadjoint builds
    exactly that derivative when the coefficient is a control. Firedrake
    passes `Real` arguments to UFL as they are, so the rule applies to them.
    Every quantity of the mesh whose facets are interior facets of the
    integral therefore goes through this helper where the integrand is
    built: test and trial functions, `Real` fields and `Real` coefficients
    that can be controls (row scalings such as `B_mu`, `Lambda`, `G`, a
    sheet scale), and continuous fields such as the potential.

    `avg` and not `'+'`: which cell UFL calls `'+'` is the mesh's cell
    ordering. `avg` of a field that is continuous across the facet is the
    field itself, so the value of the integrand does not change. For a `Real`
    function, whose one value is the same on both sides, `avg(r) = r`
    exactly, and the derivative with respect to `r` adds half of the facet
    integral from each side to its one global entry, once in total.

    Only quantities of the mesh whose interior facets these are go through
    here. On a facet-to-facet measure of a submesh (the submesh's exterior
    facets, intersected with the parent's interior facets) the fields of the
    submesh must stay unrestricted: their facets are exterior facets, and UFL
    refuses a restriction there (`Inconsistent restrictions`). A coupling
    parameter of the GIA solver can live on either mesh;
    `gadopt.gia_couplings.CouplingContext.on_coupling_facet` reads its mesh
    and calls this helper only for the parent.

    Args:
      expr: a UFL expression of the mesh whose interior facets are integrated.

    Returns:
      `avg(expr)`.
    """
    return ufl.avg(expr)


def scalar_value(value) -> float:
    """`float()` of one scalar carrier: a number, a `Constant` or a `Real` Function.

    For checks and for numbers that a PETSc callback needs. A value that
    enters a form stays UFL. `float()` of a composite UFL expression raises
    when a factor is a `Real` Function (`Division.__float__ returned
    non-float`), so a composite is computed from its factors by the caller.
    """
    try:
        return float(value)
    except (TypeError, NotImplementedError):
        pass
    # A `Real` value is owned by one rank and is a ghost entry on the others,
    # so the array with ghosts is read.
    import numpy as np
    return float(np.asarray(value.dat.data_ro_with_halos,
                            dtype=float).reshape(-1)[0])
