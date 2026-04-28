r"""This module provides several classes and functions to perform a number of pre-, syn-,
and post-processing tasks. Users incorporate utility as required in their code,
depending on what they would like to achieve.

"""

from firedrake import outer, ds_v, ds_t, ds_b, CellDiameter, CellVolume, dot, grad, JacobianInverse
from firedrake import sqrt, Function, FiniteElement, TensorProductElement, FunctionSpace, VectorFunctionSpace
from firedrake import as_vector, SpatialCoordinate, Constant, max_value, min_value, dx, assemble, tanh
from firedrake import op2, VectorElement, DirichletBC, interpolate, conditional
from firedrake.ufl_expr import extract_unique_domain
import ufl
import time
from ufl.corealg.traversal import traverse_unique_terminals
from firedrake.petsc import PETSc
from functools import cached_property
from mpi4py import MPI
import numpy as np
import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # NOQA
import os
from scipy.linalg import solveh_banded
from types import SimpleNamespace

try:
    from firedrake import MeshSequenceGeometry
except ImportError:
    MeshSequenceGeometry = None

# TBD: do we want our own set_log_level and use logging module with handlers?
log_level = logging.getLevelName(os.environ.get("GADOPT_LOGLEVEL", "INFO").upper())


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

        # J^-1 u is a discontinuous expression, using op2.MAX it takes the maximum value
        # in all adjacent elements when interpolating it to a continuous function space
        # We do need to ensure we reset ref_vel to zero, as it also takes the max with any previous values
        self.ref_vel_interpolate = interpolate(abs(dot(JacobianInverse(self.mesh), self.u)), V, access=op2.MAX)

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


class RichardsTimestepAdaptor:
    """Three-tier adaptive timestep for the Richards equation.

    Computes the minimum of three dt proposals at each step:

    1. **Advective CFL**: from gravity-driven Darcy flux q = K(h) * n_down,
       using the same |J^{-1} q| pattern as TimestepAdaptor.
    2. **Diffusive CFL**: from hydraulic diffusivity D(h) = K(h) / C(h),
       with dt ~ dx^2 / D scaling.
    3. **SNES heuristic**: scales dt up when Newton converges quickly and
       down when it struggles.

    The final dt is clamped by increase_tolerance (relative to previous dt)
    and by maximum_timestep / minimum_timestep bounds.

    Arguments:
      dt_const: Constant whose value will be updated by the adaptor
      richards_solver: RichardsSolver instance (read-only access for solution,
        soil curve, mesh, and SNES iteration count)
      target_cfl: Target CFL number for advective criterion
      target_diffusive_cfl: Target CFL number for diffusive criterion
      snes_target_its: Target SNES iteration count; dt scales up below this
      snes_scale_up: Factor to multiply dt when snes_its <= snes_target_its
      snes_scale_down: Factor to multiply dt when snes_its > snes_target_its
      increase_tolerance: Maximum factor dt can increase between steps
      maximum_timestep: Upper bound on dt (None for no limit)
      minimum_timestep: Lower bound on dt (None for no limit)
      c_floor: Floor for |C(h)| to prevent division by zero in diffusivity.
        Only affects DOFs where h < 0 (unsaturated); DOFs with h >= 0 are
        excluded from the diffusive bound entirely.

    """

    def __init__(
        self,
        dt_const,
        richards_solver,
        target_cfl=0.5,
        target_diffusive_cfl=0.5,
        snes_target_its=3,
        snes_scale_up=1.5,
        snes_scale_down=0.5,
        increase_tolerance=1.5,
        maximum_timestep=None,
        minimum_timestep=None,
        c_floor=1e-10,
    ):
        self.dt_const = dt_const
        self.solver = richards_solver
        self.target_cfl = target_cfl
        self.target_diffusive_cfl = target_diffusive_cfl
        self.snes_target_its = snes_target_its
        self.snes_scale_up = snes_scale_up
        self.snes_scale_down = snes_scale_down
        self.increase_tolerance = increase_tolerance
        self.maximum_timestep = maximum_timestep
        self.minimum_timestep = minimum_timestep
        self._has_solved = False

        h = richards_solver.solution
        soil_curve = richards_solver.soil_curve
        V = richards_solver.solution_space
        self.mesh = richards_solver.mesh
        dim = self.mesh.geometric_dimension
        self._dim = dim

        # Gravity direction (same convention as richards_gravity_term)
        x = SpatialCoordinate(self.mesh)
        n_down = grad(x[dim - 1])

        # Tier 1: advective CFL — |J^{-1} (K(h) * n_down)|
        K_expr = soil_curve.relative_permeability(h)
        q = K_expr * n_down
        J_inv_q = dot(JacobianInverse(self.mesh), q)
        ref_speed = sqrt(dot(J_inv_q, J_inv_q))
        self._adv_interpolate = interpolate(ref_speed, V, access=op2.MAX)

        # Tier 2: diffusive CFL — D(h) / dx^2 where D = K / C.
        # Only meaningful in the unsaturated regime (h < 0) where C > 0.
        # DG solutions can overshoot to h >= 0 where C = 0 by definition,
        # making D singular; those DOFs are excluded (D set to 0) so they
        # do not pollute the max.  The advective CFL and SNES heuristic
        # still protect against timestep issues in saturated regions.
        C_expr = soil_curve.water_retention(h)
        D_unsat = K_expr / max_value(abs(C_expr), c_floor)
        D_expr = conditional(h < 0, D_unsat, 0)
        ref_diff_expr = D_expr / CellDiameter(self.mesh) ** 2
        self._diff_interpolate = interpolate(ref_diff_expr, V, access=op2.MAX)

    def _get_snes_iterations(self):
        """Read SNES iteration count from the solver's most recent solve.

        Returns None if no solve has been performed yet.
        """
        try:
            return self.solver.ts.stepper.solver.snes.getIterationNumber()
        except AttributeError:
            return None

    def _compute_advective_dt(self):
        """Tier 1: dt from advective CFL on gravity-driven Darcy flux."""
        ref_vel = assemble(self._adv_interpolate)
        local_max = ref_vel.dat.data.max()
        max_refvel = self.mesh.comm.allreduce(local_max, MPI.MAX)
        if max_refvel < 1e-30:
            return float('inf')
        return self.target_cfl / max_refvel

    def _compute_diffusive_dt(self):
        """Tier 2: dt from diffusive CFL on hydraulic diffusivity D(h)."""
        ref_diff = assemble(self._diff_interpolate)
        local_max = ref_diff.dat.data.max()
        max_ref_diff = self.mesh.comm.allreduce(local_max, MPI.MAX)
        if max_ref_diff < 1e-30:
            return float('inf')
        return self.target_diffusive_cfl / (2.0 * self._dim * max_ref_diff)

    def _compute_snes_dt(self, snes_its):
        """Tier 3: scale current dt based on SNES iteration count."""
        current_dt = float(self.dt_const)
        if snes_its <= self.snes_target_its:
            return current_dt * self.snes_scale_up
        else:
            return current_dt * self.snes_scale_down

    def compute_timestep(self):
        """Compute the adaptive timestep from all three tiers.

        Returns:
          The adapted timestep value (float).
        """
        dt_adv = self._compute_advective_dt()
        dt_diff = self._compute_diffusive_dt()
        snes_its = self._get_snes_iterations()

        dt_proposed = min(dt_adv, dt_diff)
        if dt_adv <= dt_diff:
            controlling = "advective CFL"
        else:
            controlling = "diffusive CFL"

        if snes_its is not None:
            dt_snes = self._compute_snes_dt(snes_its)
            if dt_snes < dt_proposed:
                controlling = f"SNES ({snes_its} its)"
            dt_proposed = min(dt_proposed, dt_snes)
        else:
            dt_snes = None

        # Clamp by increase_tolerance relative to current dt
        max_ts = float(self.dt_const) * self.increase_tolerance
        if self.maximum_timestep is not None:
            max_ts = min(max_ts, self.maximum_timestep)
        if dt_proposed > max_ts:
            controlling = "increase tolerance"
            dt_proposed = max_ts

        if self.minimum_timestep is not None and dt_proposed < self.minimum_timestep:
            log(f"RichardsTimestepAdaptor: dt {dt_proposed:.3e} below "
                f"minimum {self.minimum_timestep:.3e}, clamping.")
            controlling = "minimum timestep"
            dt_proposed = self.minimum_timestep

        snes_info = (f", snes={dt_snes:.3e} [{snes_its} its]"
                     if dt_snes is not None else "")
        log(f"RichardsTimestepAdaptor: dt={dt_proposed:.3e} "
            f"(adv={dt_adv:.3e}, diff={dt_diff:.3e}"
            f"{snes_info}) controlled by {controlling}")

        return dt_proposed

    def update_timestep(self):
        """Compute and assign the adapted timestep.

        Returns:
          The new timestep value (float).
        """
        self.dt_const.assign(self.compute_timestep())
        return float(self.dt_const)

    def solve(self, t=None, max_retries=5):
        """Solve one timestep with automatic retry on SNES failure.

        If the nonlinear solve diverges, restores the previous solution,
        halves dt, and retries up to max_retries times.  On success,
        adapts dt for the next step via the three-tier heuristic.

        Args:
          t: Current simulation time (passed to solver.solve).
          max_retries: Maximum number of halving retries before raising.

        Returns:
          The timestep that was actually used (float).
        """
        from firedrake import ConvergenceError

        # On the very first call, set dt from CFL bounds before solving
        if not self._has_solved:
            self.update_timestep()
            self._has_solved = True

        for attempt in range(max_retries + 1):
            try:
                self.solver.solve(t)
                break
            except (ConvergenceError, Exception) as e:
                if not (isinstance(e, ConvergenceError)
                        or "SNES" in str(e) or "error code" in str(e)):
                    raise
                if attempt == max_retries:
                    raise
                # Restore pre-solve solution and halve dt
                self.solver.solution.assign(self.solver.ts.solution_old)
                new_dt = float(self.dt_const) * self.snes_scale_down
                if self.minimum_timestep is not None:
                    new_dt = max(new_dt, self.minimum_timestep)
                self.dt_const.assign(new_dt)
                log(f"RichardsTimestepAdaptor: SNES diverged, "
                    f"retrying with dt={new_dt:.3e} "
                    f"(attempt {attempt + 1}/{max_retries})")

        dt_used = float(self.dt_const)
        self.update_timestep()
        return dt_used


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
    """
    Returns a 3D view of a 2D `Function` on the extruded domain.
    The 3D function resides in V x R function space, where V is the function
    space of the source function. The 3D function shares the data of the 2D
    function.
    """
    fs = func.function_space()
#    assert fs.mesh().geometric_dimension == 2, 'Function must be in 2D space'
    ufl_elem = fs.ufl_element()
    family = ufl_elem.family()
    degree = ufl_elem.degree()
    name = func.name()
    if isinstance(ufl_elem, VectorElement):
        # vector function space
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0, dim=2, vector=True)
    else:
        fs_extended = get_functionspace(mesh_extruded, family, degree, 'R', 0)
    func_extended = Function(fs_extended, name=name, val=func.dat._data)
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
    def nodes(self):
        return np.array(list(set(range(self._function_space.node_count)) - set(super().nodes)))


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
            if identified[face_id] is None:
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
        kwargs = dict(
            [
                (proc_bdy.get(face_id), face_id)  # invert boundary mapping
                for proc_bdy in reversed(gathered_boundaries)  # keep value on lowest comm rank
                for face_id in set().union(*gathered_boundaries)  # all gathered face_ids
            ]
        )
        kwargs.pop(None, None)  # remove None entry
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
