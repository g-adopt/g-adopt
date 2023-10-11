import gadopt as ga
import numpy as np
import shapely as sl
from gadopt.approximations import BoussinesqApproximation
from gadopt.scalar_equation import ScalarAdvectionEquation
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import SSPRK33
from gadopt.utility import TimestepAdaptor, is_continuous


class LevelSetEquation(ScalarAdvectionEquation):
    def __init__(self, test_space, trial_space, quad_degree=None):
        super().__init__(test_space, trial_space, quad_degree=quad_degree)

    def mass_term(self, test, trial):
        return super().mass_term(test, trial)


class LevelSetSolver:
    def __init__(self, level_set, u, dt, timestepper, bcs=None, solver_parameters=None):
        self.level_set = level_set
        self.Q = level_set.function_space()
        self.mesh = self.Q.mesh()
        self.dt = dt
        self.eq = LevelSetEquation(self.Q, self.Q)
        self.fields = {"velocity": u}

        if solver_parameters is None:
            self.solver_parameters = {
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
                        self.strong_bcs.append(ga.DirichletBC(self.Q, value, id))
                    else:
                        weak_bc["q"] = value
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.timestepper = timestepper
        self.level_set_old = ga.Function(self.Q)
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


# Set up geometry
nx, ny = 32, 32
lx, ly = 0.9142, 1
mesh = ga.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Mesh boundary IDs

mesh_coords = ga.SpatialCoordinate(mesh)

level_set_func_space_deg = 1
func_space_dg = ga.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
level_set = ga.Function(func_space_dg, name="level_set")
level_set_step = ga.Function(func_space_dg)
node_coords_x = ga.Function(func_space_dg).interpolate(mesh_coords[0]).dat.data
node_coords_y = ga.Function(func_space_dg).interpolate(mesh_coords[1]).dat.data

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

level_set.dat.data[:] = node_sign_dist_to_curve

# Set up Stokes function spaces - currently using the bilinear Q2Q1 element pair:
V = ga.VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = ga.FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = ga.MixedFunctionSpace([V, W])  # Mixed function space

z = ga.Function(Z)  # Field over the mixed function space Z
u, p = ga.split(z)  # Symbolic UFL expression for u and p

# Parameters; since they are included in UFL, they are wrapped inside Constant
g = ga.Constant(-10)
Ra = ga.Constant(1)
T = ga.Constant(1)

rho = ga.conditional(level_set > 0, 1 / g, 0 / g)
mu = ga.conditional(level_set > 0, 1, 0.1)

approximation = BoussinesqApproximation(Ra, rho=rho)
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

dt = ga.Constant(1)  # Initial time-step
t_adapt = TimestepAdaptor(dt, u, V, target_cfl=0.2, maximum_timestep=5)

level_set_solver = LevelSetSolver(level_set, u, dt, SSPRK33)

u_, p_ = z.subfunctions  # extract individual velocity and pressure fields
# Rename for output
u_.rename("Velocity")
p_.rename("Pressure")

time_now, time_end = 0, 2000
dump_counter, dump_period = 0, 10
output_file = ga.File("van_keken/output.pvd", target_degree=level_set_func_space_deg)

# Perform the time loop
while time_now < time_end:
    if time_now > dump_counter * dump_period:
        dump_counter += 1
        output_file.write(level_set, u_, p_)

    dt = t_adapt.update_timestep()
    time_now += dt

    stokes_solver.solve()

    level_set_solver.solve()
output_file.write(level_set, u_, p_)
