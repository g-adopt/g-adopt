r"""This module provides several classes and functions to perform a number of pre-, syn-,
and post-processing tasks. Users incorporate utility as required in their code,
depending on what they would like to achieve.

"""

import os
from functools import cached_property
from logging import getLevelName

import numpy as np
from firedrake import *
from firedrake.__future__ import Interpolator
from firedrake.ufl_expr import extract_unique_domain
from mpi4py import MPI
from scipy.linalg import solveh_banded
from ufl.corealg.traversal import traverse_unique_terminals

# TBD: do we want our own set_log_level and use logging module with handlers?
log_level = getLevelName(os.environ.get("GADOPT_LOGLEVEL", "INFO").upper())


def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


class ParameterLog:
    def __init__(self, filename, mesh) -> None:
        self.comm = mesh.comm
        if self.comm.rank == 0:
            self.f = open(filename, "w")

    def log_str(self, str) -> None:
        if self.comm.rank == 0:
            self.f.write(str + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.comm.rank == 0:
            self.f.close()


class TimestepAdaptor:
    """Computes timestep based on CFL condition for provided velocity field

    Args:
      dt_const:
        Constant whose value will be updated by the timestep adaptor
      u:
        Velocity to base CFL condition on
      V:
        Function space for reference velocity, usually velocity space
      target_cfl:
        CFL number to target with chosen timestep
      increase_tolerance:
        Maximum tolerance timestep is allowed to change by
      maximum_timestep:
        Maximum allowable timestep

    """

    def __init__(
        self,
        dt_const,
        u,
        V,
        target_cfl: float = 1.0,
        increase_tolerance: float = 1.5,
        maximum_timestep: float | None = None,
    ) -> None:
        self.dt_const = dt_const
        self.u = u
        self.target_cfl = target_cfl
        self.increase_tolerance = increase_tolerance
        self.maximum_timestep = maximum_timestep
        self.mesh = V.mesh()

        # J^-1 u is a discontinuous expression, using op2.MAX it takes the maximum value
        # in all adjacent elements when interpolating it to a continuous function space
        # We do need to ensure we reset ref_vel to zero, as it also takes the max with any previous values
        self.ref_vel_interpolator = Interpolator(
            abs(dot(JacobianInverse(self.mesh), self.u)), V, access=op2.MAX
        )

    def compute_timestep(self) -> float:
        max_ts = float(self.dt_const) * self.increase_tolerance
        if self.maximum_timestep is not None:
            max_ts = min(max_ts, self.maximum_timestep)

        # need to reset ref_vel to avoid taking max with previous values
        ref_vel = assemble(self.ref_vel_interpolator.interpolate())
        local_maxrefvel = ref_vel.dat.data.max()
        max_refvel = self.mesh.comm.allreduce(local_maxrefvel, MPI.MAX)
        # NOTE; we're incorparating max_ts here before dividing by max. ref. vel. as it may be zero
        ts = self.target_cfl / max(max_refvel, self.target_cfl / max_ts)

        return ts

    def update_timestep(self) -> float:
        self.dt_const.assign(self.compute_timestep())
        return float(self.dt_const)


def upward_normal(mesh):
    if mesh.cartesian:
        n = mesh.geometric_dimension()
        return as_vector([0] * (n - 1) + [1])
    else:
        X = SpatialCoordinate(mesh)
        r = sqrt(inner(X, X))
        return X / r


def vertical_component(u):
    mesh = extract_unique_domain(u)

    if mesh.cartesian:
        return u[u.ufl_shape[0] - 1]
    else:
        n = upward_normal(mesh)
        return sum([n_i * u_i for n_i, u_i in zip(n, u)])


def ensure_constant(f):
    if isinstance(f, float) or isinstance(f, int):
        return Constant(f)
    else:
        return f


class CombinedSurfaceMeasure(ufl.Measure):
    """
    A surface measure that combines ds_v, the integral over vertical boundary facets, and ds_t and ds_b,
    the integral over horizontal top and bottom facets. The vertical boundary facets are identified with
    the same surface ids as ds_v. The top and bottom surfaces are identified via the "top" and "bottom" ids."""

    def __init__(self, domain, degree: int) -> None:
        self.ds_v = ds_v(domain=domain, degree=degree)
        self.ds_t = ds_t(domain=domain, degree=degree)
        self.ds_b = ds_b(domain=domain, degree=degree)

    def __call__(self, subdomain_id: int | str, **kwargs):
        if subdomain_id == "top":
            return self.ds_t(**kwargs)
        elif subdomain_id == "bottom":
            return self.ds_b(**kwargs)
        else:
            return self.ds_v(subdomain_id, **kwargs)

    def __rmul__(self, other):
        """This is to handle terms to be integrated over all surfaces in the form of other*ds.
        Here the CombinedSurfaceMeasure ds is not called, instead we just split it up as below."""
        return other * self.ds_v + other * self.ds_t + other * self.ds_b


def is_continuous(expr, normal: bool = False) -> bool:
    element_space = ufl.HDiv if normal else ufl.H1

    ufl_algebra = (ufl.algebra.Division, ufl.algebra.Product, ufl.algebra.Sum)
    compatible_types = (Function, ufl.indexed.Indexed, ufl.tensors.ListTensor)

    if isinstance(expr, AbstractFiniteElement):
        return expr in element_space
    elif isinstance(expr, (functionspaceimpl.WithGeometry, Function)):
        return expr.ufl_element() in element_space
    elif isinstance(expr, ufl.indexed.Indexed):
        expr, multiindex = expr.ufl_operands
        V = expr.ufl_function_space()

        for i in multiindex:
            comp_to_subspace = tuple(
                j for j, W in enumerate(V) for k in range(W.value_size)
            )
            V = V.sub(comp_to_subspace[int(i)])

        elem = V

        return elem in element_space
    elif isinstance(expr, ufl.tensors.ListTensor):
        return all(is_continuous(operand) for operand in expr.ufl_operands)
    elif isinstance(expr, ufl_algebra):
        continuity = []
        expr_list = [expr]

        while expr_list:
            for operand in expr_list.pop().ufl_operands:
                if isinstance(operand, compatible_types):
                    continuity.append(is_continuous(operand))
                elif isinstance(operand, ufl_algebra):
                    expr_list.append(operand)
                else:
                    continue

        return all(continuity)
    else:
        raise ValueError(f"Continuity test not implemented for {type(expr)}.")


def depends_on(ufl_expr, terminal) -> bool:
    """Does ufl_expr depend on terminal (Function/Constant/...)?"""
    if isinstance(ufl_expr, float) or isinstance(ufl_expr, int):
        return False
    else:
        return terminal in traverse_unique_terminals(ufl_expr)


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    $$"jump"(bb u, bb n) = (bb u^+ bb n^+) + (bb u^- bb n^-)$$

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator `ufl.jump` which represents div(u).
    The equivalent of nabla_grad(u) is given by tensor_jump(n, u).
    """
    return outer(v("+"), n("+")) + outer(v("-"), n("-"))


def extend_function_to_3d(func, mesh_extruded):
    """
    Returns a 3D view of a 2D `Function` on the extruded domain.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
    # assert fs.mesh().geometric_dimension() == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, VectorElement):
        # vector function space
        fs_extended = get_functionspace(
            mesh_extruded, family, degree, "R", 0, dim=2, vector=True
        )
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, "R", 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
    func_extended.source = func
    return func_extended


def get_functionspace(
    mesh,
    h_family,
    h_degree,
    v_family=None,
    v_degree=None,
    vector=False,
    hdiv=False,
    variant=None,
    v_variant=None,
    **kwargs,
):
    cell_dim = mesh.cell_dimension()
    print(cell_dim)
    assert cell_dim in [2, (2, 1), (1, 1)], "Unsupported cell dimension"
    hdiv_families = [
        "RT",
        "RTF",
        "RTCF",
        "RAVIART-THOMAS",
        "BDM",
        "BDMF",
        "BDMCF",
        "BREZZI-DOUGLAS-MARINI",
    ]
    if variant is None:
        if h_family.upper() in hdiv_families:
            if h_family in ["RTCF", "BDMCF"]:
                variant = "equispaced"
            else:
                variant = "integral"
        else:
            print("var = equi")
            variant = "equispaced"
    if v_variant is None:
        v_variant = "equispaced"
    if cell_dim == (2, 1) or (1, 1):
        if v_family is None:
            v_family = h_family
        if v_degree is None:
            v_degree = h_degree
        h_cell, v_cell = mesh.ufl_cell().sub_cells()
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

    Args:
      mesh: The mesh over which to compute averages
      r1d: An array of either depth coordinates or radii,
             at which to compute layer averages. If not provided, and
             mesh is extruded, it uses the same layer heights. If mesh
             is not extruded, r1d is required.

    """

    def __init__(self, mesh, r1d=None, quad_degree: int | None = None) -> None:
        self.mesh = mesh
        XYZ = SpatialCoordinate(mesh)

        if mesh.cartesian:
            self.r = XYZ[len(XYZ) - 1]
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
                raise ValueError(
                    "For non-extruded mesh need to specify depths array r1d."
                )
            CG1 = FunctionSpace(mesh, "CG", 1)
            r_func = Function(CG1).interpolate(self.r)
            self.r1d = r_func.dat.data[:nlayers]

        self.mass = np.zeros((2, len(self.r1d)))
        self.rhs = np.zeros(len(self.r1d))
        self._assemble_mass()

    def _assemble_mass(self) -> None:
        # main diagonal of mass matrix
        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.0)

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
        overlap = (
            max_value((rn - r) / (rn - rp), 0)
            * max_value((r - rp) / (rn - rp), 0)
            * self.dx
        )

        for i, rin in enumerate(self.r1d[1:]):
            rn.assign(rin)
            self.mass[1, i] = assemble(overlap)

            # shuffle coefficients for next iteration
            rp.assign(rn)

    def _assemble_rhs(self, T) -> None:
        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.0)

        phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)

        for i, rin in enumerate(self.r1d[1:]):
            rn.assign(rin)
            self.rhs[i] = assemble(phi * T * self.dx)

            rp.assign(rc)
            rc.assign(rn)

        phi = max_value(min_value(1, (r - rp) / (rn - rp)), 0)
        self.rhs[-1] = assemble(phi * T * self.dx)

    def get_layer_average(self, T) -> np.ndarray:
        """Compute the layer averages of `Function` T at the predefined depths.

        Returns:
          A numpy array containing the averages.
        """

        self._assemble_rhs(T)
        return solveh_banded(self.mass, self.rhs, lower=True)

    def extrapolate_layer_average(self, u, avg) -> None:
        """Given an array of layer averages avg, extrapolate to `Function` u"""

        r = self.r
        rc = Constant(self.r1d[0])
        rn = Constant(self.r1d[1])
        rp = Constant(0.0)

        u.assign(0.0)

        phi = max_value(min_value((r - rp) / (rc - rp), (rn - r) / (rn - rc)), 0)
        val = Constant(0.0)

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


class InteriorBC(DirichletBC):
    """DirichletBC applied to anywhere that is *not* on the specified boundary"""

    @cached_property
    def nodes(self) -> np.ndarray:
        return np.array(
            list(set(range(self._function_space.node_count)) - set(super().nodes))
        )


def absv(u):
    """Component-wise absolute value of vector for SU stabilisation"""
    return as_vector([abs(ui) for ui in u])


def step_func(r, centre, mag, increasing: bool = True, sharpness=50):
    """A step function designed to represent viscosity jumps.

    Builds a step centred at "centre" with given magnitude.
    Increases with radius if "increasing" is True.
    """
    sign = 1 if increasing else -1
    return mag * (0.5 * (1 + tanh(sign * (r - centre) * sharpness)))


def node_coordinates(function: Function) -> np.ndarray:
    """Interpolates mesh coordinates at each node of the provided function

    Args:
      function: A Firedrake function

    Returns:
      A NumPy array holding the interpolated mesh coordinates
    """
    vec_space = VectorFunctionSpace(function.ufl_domain(), function.ufl_element())

    return Function(vec_space).interpolate(SpatialCoordinate(function)).dat.data


def interpolate_1d_profile(function: Function, one_d_filename: str) -> None:
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
