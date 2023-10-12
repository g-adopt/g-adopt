import firedrake as fd
import numpy as np
import shapely as sl
from gadopt.approximations import BoussinesqApproximation
from gadopt.level_set_solver import LevelSetSolver
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import SSPRK33
from gadopt.utility import TimestepAdaptor


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
nx, ny = 64, 64
lx, ly = 0.9142, 1
mesh = fd.RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Mesh boundary IDs

mesh_coords = fd.SpatialCoordinate(mesh)

conservative_level_set = True

level_set_func_space_deg = 2
func_space_dg = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
level_set = fd.Function(func_space_dg, name="level_set")
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
t_adapt = TimestepAdaptor(dt, u, V, target_cfl=0.2, maximum_timestep=5)

level_set_solver = LevelSetSolver(level_set, u, dt, SSPRK33)

time_now, time_end = 0, 2000
dump_counter, dump_period = 0, 10
output_file = fd.File("level_set/output.pvd", target_degree=level_set_func_space_deg)

# Extract individual velocity and pressure fields and rename them for output
u_, p_ = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

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
