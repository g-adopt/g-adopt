import firedrake as fd
import numpy as np
import shapely as sl

from gadopt.approximations import BoussinesqApproximation
from gadopt.equations import BaseEquation, BaseTerm
from gadopt.level_set_solver import LevelSetSolver
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import SSPRK33
from gadopt.utility import TimestepAdaptor, ensure_constant, is_continuous


class ProjectionTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        u = fields["target"]
        n = self.n

        ufl_element = u * fd.div(test) * self.dx
        # integration by parts leads to boundary term
        ufl_bc = u * fd.dot(test, n) * self.ds

        return -ufl_element + ufl_bc


class ProjectionEquation(BaseEquation):
    terms = [ProjectionTerm]

    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class ProjectionSolver:
    def __init__(self, field, target, solver_parameters=None):
        self.field = field
        self.func_space = field.function_space()
        self.mesh = self.func_space.mesh()
        self.test = fd.TestFunction(self.func_space)
        self.trial = fd.TrialFunction(self.func_space)
        self.eq = ProjectionEquation(self.func_space, self.func_space)
        self.fields = {"target": target}

        if solver_parameters is None:
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            self.solver_parameters = solver_parameters

        self.bilinear = self.eq.mass_term(self.test, self.trial)
        self.linear = self.eq.residual(self.test, None, None, self.fields, None)

        # solver is set up only last minute to enable overwriting of the parameters we
        # have set up here
        self._solver_setup = False

    def setup_solver(self):
        self.problem = fd.LinearVariationalProblem(
            self.bilinear, self.linear, self.field
        )
        self.solver = fd.LinearVariationalSolver(
            self.problem, solver_parameters=self.solver_parameters
        )
        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        self.solver.solve()


class ReinitialisationSharpenTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        return -trial * (1 - trial) * (1 - 2 * trial) * test * self.dx


class ReinitialisationBalanceTerm(BaseTerm):
    def residual(self, test, trial, trial_lagged, fields, bcs):
        gradient = fields["level_set_grad"]
        epsilon = fields["epsilon"]

        return (
            epsilon
            * (1 - 2 * trial)
            * fd.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
            * test
            * self.dx
        )


class ReinitialisationEquation(BaseEquation):
    terms = [ReinitialisationSharpenTerm, ReinitialisationBalanceTerm]

    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class ReinitialisationSolver:
    def __init__(
        self,
        level_set,
        level_set_grad,
        epsilon,
        dt,
        timestepper,
        bcs=None,
        solver_parameters=None,
    ):
        self.level_set = level_set
        self.func_space = level_set.function_space()
        self.mesh = self.func_space.mesh()
        self.dt = dt
        self.eq = ReinitialisationEquation(self.func_space, self.func_space)
        self.fields = {
            "level_set_grad": level_set_grad,
            "epsilon": ensure_constant(epsilon),
        }

        if solver_parameters is None:
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu",
            }
        else:
            self.solver_parameters = solver_parameters
        apply_strongly = is_continuous(level_set)
        self.strong_bcs = []
        self.weak_bcs = {}
        bcs = bcs or {}
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == "T":
                    if apply_strongly:
                        self.strong_bcs.append(
                            fd.DirichletBC(self.func_space, value, id)
                        )
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.timestepper = timestepper
        self.level_set_old = fd.Function(self.func_space)
        # solver is set up only last minute to enable overwriting of the parameters we
        # have set up here
        self._solver_setup = False

    def setup_solver(self):
        """Setup timestepper and associated solver"""
        self.ts = self.timestepper(
            self.eq,
            self.level_set,
            self.fields,
            self.dt,
            bnd_conditions=self.weak_bcs,
            solution_old=self.level_set_old,
            strong_bcs=self.strong_bcs,
            solver_parameters=self.solver_parameters,
        )
        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        t = 0  # not used atm
        self.ts.advance(t)


def initial_signed_distance(lx):
    interface_x = np.linspace(0, lx, 1000)
    interface_y = 0.2 + 0.02 * np.cos(np.pi * interface_x / 0.9142)
    curve = sl.LineString([*np.column_stack((interface_x, interface_y))])
    sl.prepare(curve)

    node_relation_to_curve = [
        (
            y > 0.2 + 0.02 * np.cos(np.pi * x / 0.9142),
            curve.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    node_sign_dist_to_curve = [
        dist if is_above else -dist for is_above, dist in node_relation_to_curve
    ]
    return node_sign_dist_to_curve


# Set up geometry
nx, ny = 32, 32
lx, ly = 0.9142, 1
mesh = fd.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Mesh boundary IDs

mesh_coords = fd.SpatialCoordinate(mesh)

conservative_level_set = True

level_set_func_space_deg = 2
func_space_dg = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
vec_func_space_cg = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = fd.Function(func_space_dg, name="level_set")
level_set_grad_proj = fd.Function(vec_func_space_cg, name="level_set_grad_proj")
node_coords_x = fd.Function(func_space_dg).interpolate(mesh_coords[0]).dat.data
node_coords_y = fd.Function(func_space_dg).interpolate(mesh_coords[1]).dat.data

node_sign_dist_to_interface = initial_signed_distance(lx)

if conservative_level_set:
    epsilon = 5e-3  # This needs to be parameterised
    level_set.dat.data[:] = (
        np.tanh(np.asarray(node_sign_dist_to_interface) / 2 / epsilon) + 1
    ) / 2
else:
    level_set.dat.data[:] = node_sign_dist_to_interface

# Set up Stokes function spaces - currently using the bilinear Q2Q1 element pair:
V = fd.VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = fd.FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = fd.MixedFunctionSpace([V, W])  # Mixed function space

z = fd.Function(Z)  # Field over the mixed function space Z
u, p = fd.split(z)  # Symbolic UFL expression for u and p

# Parameters; since they are included in UFL, they are wrapped inside Constant
g = fd.Constant(10)
Ra = fd.Constant(-1)
T = fd.Constant(1)

if conservative_level_set:
    rho = fd.conditional(level_set > 0.5, 1 / g, 0 / g)
    mu = fd.conditional(level_set > 0.5, 1, 0.1)
else:
    rho = fd.conditional(level_set > 0, 1 / g, 0 / g)
    mu = fd.conditional(level_set > 0, 1, 0.1)

approximation = BoussinesqApproximation(Ra, g=g, rho=rho)
Z_nullspace = create_stokes_nullspace(Z)
stokes_bcs = {
    bottom_id: {"ux": 0, "uy": 0},
    top_id: {"ux": 0, "uy": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    mu=mu,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
)

dt = fd.Constant(1)  # Initial time-step
dt_reini = fd.Constant(1e-4)
# Empirical calibration. Should it be included at all?
target_cfl = 2 * (ly / ny) ** (1 / 10) / (level_set_func_space_deg + 1) ** (5 / 3)
t_adapt = TimestepAdaptor(dt, u, V, target_cfl=target_cfl, maximum_timestep=5)

level_set_solver = LevelSetSolver(level_set, u, dt, SSPRK33)
projection_solver = ProjectionSolver(level_set_grad_proj, level_set)
reinitialisation_solver = ReinitialisationSolver(
    level_set, level_set_grad_proj, epsilon, dt_reini, SSPRK33
)

time_now, time_end = 0, 2000
dump_counter, dump_period = 0, 10
output_file = fd.File("level_set/output.pvd", target_degree=level_set_func_space_deg)

# Extract individual velocity and pressure fields and rename them for output
u_, p_ = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

# Perform the time loop
step = 0
while time_now < time_end:
    if time_now >= dump_counter * dump_period:
        dump_counter += 1
        output_file.write(level_set, u_, p_, level_set_grad_proj)

    dt = t_adapt.update_timestep()
    time_now += dt

    stokes_solver.solve()

    level_set_solver.solve()

    projection_solver.solve()

    # if step > 0 and step % 5 == 0:
    #     if conservative_level_set:
    # for reini_iter in range(100):
    reinitialisation_solver.solve()

    step += 1
output_file.write(level_set, u_, p_, level_set_grad_proj)
