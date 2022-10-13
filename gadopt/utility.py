"""
A module with utitity functions for gadopt
"""
from firedrake import outer, ds_v, ds_t, ds_b, CellDiameter, CellVolume, dot, JacobianInverse
from firedrake import sqrt, Function, FiniteElement, TensorProductElement, FunctionSpace, VectorFunctionSpace
from firedrake import as_vector, SpatialCoordinate, Constant
import ufl
from ufl.corealg.traversal import traverse_unique_terminals
from firedrake.petsc import PETSc
from mpi4py import MPI
import numpy as np
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # NOQA


# TBD: do we want our own set_log_level and use logging module with handlers?
log_level = INFO


def log(*args):
    """Log output to stdout from root processor only"""
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
        self.f.close()


class TimestepAdaptor:
    def __init__(self, dt_const, u_fs, target_cfl=1.0, increase_tolerance=1.5, maximum_timestep=None):
        self.dt_const = dt_const
        self.u_fs = u_fs
        self.target_cfl = target_cfl
        self.increase_tolerance = increase_tolerance
        self.maximum_timestep = maximum_timestep
        self.mesh = u_fs.mesh()
        self.ref_vel = Function(self.u_fs, name="Reference_Velocity")

    def compute_timestep(self, u):
        max_ts = float(self.dt_const)*self.increase_tolerance
        if self.maximum_timestep is not None:
            max_ts = min(max_ts, self.maximum_timestep)

        self.ref_vel.interpolate(dot(JacobianInverse(self.mesh), u))
        # NOTE; we're incorparing max_ts here before divinging by max. ref. vel. as it may be zero
        ts = self.target_cfl / max(self.mesh.comm.allreduce(np.abs(self.ref_vel.dat.data.max()), MPI.MAX), self.target_cfl / max_ts)

        return ts

    def update_timestep(self, u):
        self.dt_const.assign(self.compute_timestep(u))
        return float(self.dt_const)


def upward_normal(mesh, cartesian):
    if cartesian:
        n = mesh.geometric_dimension()
        return as_vector([0]*(n-1) + [1])
    else:
        X = SpatialCoordinate(mesh)
        r = sqrt(dot(X, X))
        return X/r


def vertical_component(u, cartesian):
    if cartesian:
        return u[u.ufl_shape[0]-1]
    else:
        n = upward_normal(u.ufl_domain(), cartesian)
        return dot(n, u)


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
    if isinstance(ufl_or_element, ufl.FiniteElementBase):
        return ufl_or_element
    else:
        return ufl_or_element.ufl_element()


def is_continuous(expr):
    if isinstance(expr, ufl.tensors.ListTensor):
        return all(is_continuous(x) for x in expr.ufl_operands)

    if isinstance(expr, ufl.indexed.Indexed):
        elem = expr.ufl_operands[0].ufl_element()
        if isinstance(elem, ufl.MixedElement):
            # the second operand is a MultiIndex
            assert(len(expr.ufl_operands[1]) == 1)
            sub_element_index, _ = elem.extract_subelement_component(int(expr.ufl_operands[1][0]))
            elem = elem.sub_elements()[sub_element_index]
    else:
        elem = _get_element(expr)

    family = elem.family()
    if family == 'Lagrange' or family == 'Q':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    elif isinstance(elem, ufl.HCurlElement) or isinstance(elem, ufl.HDivElement):
        return False
    elif family == 'TensorProductElement':
        elem_h, elem_v = elem.sub_elements()
        return is_continuous(elem_h) and is_continuous(elem_v)
    elif family == 'EnrichedElement':
        return all(is_continuous(e) for e in elem._elements)
    else:
        raise NotImplementedError("Unknown finite element family")


def depends_on(ufl_expr, terminal):
    """Does ufl_expr depend on terminal (Function/Constant/...)?"""
    return terminal in traverse_unique_terminals(ufl_expr)


def normal_is_continuous(expr):
    # if we get some list expression, we can't guarantee its normal is continuous
    # unless all components are
    if isinstance(expr, ufl.tensors.ListTensor):
        return is_continuous(expr)

    elem = _get_element(expr)

    family = elem.family()
    if family == 'Lagrange' or family == 'Q':
        return True
    elif family == 'Discontinuous Lagrange' or family == 'DQ':
        return False
    elif isinstance(elem, ufl.HCurlElement):
        return False
    elif isinstance(elem, ufl.HDivElement):
        return True
    elif family == 'TensorProductElement':
        elem_h, elem_v = elem.sub_elements()
        return normal_is_continuous(elem_h) and normal_is_continuous(elem_v)
    elif family == 'EnrichedElement':
        return all(normal_is_continuous(e) for e in elem._elements)
    else:
        raise NotImplementedError("Unknown finite element family")


def cell_size(mesh):
    if hasattr(mesh.ufl_cell(), 'sub_cells'):
        return sqrt(CellVolume(mesh))
    else:
        return CellDiameter(mesh)


def cell_edge_integral_ratio(mesh, p):
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2
    for facets f, elements e and polynomials u of degree p.

    See eqn. (3.7) ad table 3.1 from Hillewaert's thesis: https://www.researchgate.net/publication/260085826
    and its appendix C for derivation."""
    cell_type = mesh.ufl_cell().cellname()
    if cell_type == "triangle":
        return (p+1)*(p+2)/2.
    elif cell_type == "quadrilateral" or cell_type == "interval * interval":
        return (p+1)**2
    elif cell_type == "triangle * interval":
        return (p+1)**2
    elif cell_type == "quadrilateral * interval":
        # if e is a wedge and f is a triangle: (p+1)**2
        # if e is a wedge and f is a quad: (p+1)*(p+2)/2
        # here we just return the largest of the the two (for p>=0)
        return (p+1)**2
    elif cell_type == "tetrahedron":
        return (p+1)*(p+3)/3
    else:
        raise NotImplementedError("Unknown cell type in mesh: {}".format(cell_type))


def tensor_jump(v, n):
    r"""
    Jump term for vector functions based on the tensor product

    .. math::
        \text{jump}(\mathbf{u}, \mathbf{n}) = (\mathbf{u}^+ \mathbf{n}^+) +
        (\mathbf{u}^- \mathbf{n}^-)

    This is the discrete equivalent of grad(u) as opposed to the
    vectorial UFL jump operator :meth:`ufl.jump` which represents div(u).
    The equivalent of nabla_grad(u) is given by tensor_jump(n, u).
    """
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


def extend_function_to_3d(func, mesh_extruded):
    """
    Returns a 3D view of a 2D :class:`Function` on the extruded domain.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
#    assert fs.mesh().geometric_dimension() == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, ufl.VectorElement):
        # vector function space
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0, dim=2, vector=True)
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
    func_extended.source = func
    return func_extended


class ExtrudedFunction(Function):
    """
    A 2D :class:`Function` that provides a 3D view on the extruded domain.
    The 3D function can be accessed as `ExtrudedFunction.view_3d`.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function."""
    def __init__(self, *args, mesh_3d=None, **kwargs):
        """
        Create a 2D :class:`Function` with a 3D view on extruded mesh.
        :arg mesh_3d: Extruded 3D mesh where the function will be extended to.
        """
        # create the 2d function
        super().__init__(*args, **kwargs)
        print(*args)
        if mesh_3d is not None:
            self.view_3d = extend_function_to_3d(self, mesh_3d)


def get_functionspace(mesh, h_family, h_degree, v_family=None, v_degree=None,
                      vector=False, hdiv=False, variant=None, v_variant=None,
                      **kwargs):
    cell_dim = mesh.cell_dimension()
    print(cell_dim)
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
            print("var = equi")
            variant = 'equispaced'
    if v_variant is None:
        v_variant = 'equispaced'
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
            elt = ufl.HDiv(elt)
    else:
        elt = FiniteElement(h_family, mesh.ufl_cell(), h_degree, variant=variant)

    constructor = VectorFunctionSpace if vector else FunctionSpace
    return constructor(mesh, elt, **kwargs)
