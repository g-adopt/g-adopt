import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import shapely as sl
from mpi4py import MPI

from gadopt.approximations import BoussinesqApproximation
from gadopt.diagnostics import domain_volume
from gadopt.level_set_tools import (
    LevelSetEquation,
    ProjectionSolver,
    ReinitialisationEquation,
    TimeStepperSolver,
)
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import SSPRK33
from gadopt.utility import TimestepAdaptor


def initial_signed_distance(lx, node_coords_x, node_coords_y):
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
vec_func_space_cg = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = fd.Function(func_space_dg, name="level_set")
level_set_grad_proj = fd.Function(vec_func_space_cg, name="level_set_grad_proj")
node_coords_x = fd.Function(func_space_dg).interpolate(mesh_coords[0]).dat.data
node_coords_y = fd.Function(func_space_dg).interpolate(mesh_coords[1]).dat.data

node_sign_dist_to_interface = initial_signed_distance(lx, node_coords_x, node_coords_y)

if conservative_level_set:
    epsilon = fd.Constant(min(lx / nx, ly / ny) / 4)  # Loose guess
    level_set.dat.data[:] = (
        np.tanh(np.asarray(node_sign_dist_to_interface) / 2 / epsilon.values().item())
        + 1
    ) / 2
    level_set_contour = 0.5
else:
    level_set.dat.data[:] = node_sign_dist_to_interface
    level_set_contour = 0

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

rho = fd.conditional(level_set > level_set_contour, 1 / g, 0 / g)
mu = fd.conditional(level_set > level_set_contour, 1, 1)

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
# Empirical calibration
target_cfl = 2 * (ly / ny) ** (1 / 10) / (level_set_func_space_deg + 1) ** (5 / 3)
t_adapt = TimestepAdaptor(dt, u, V, target_cfl=target_cfl, maximum_timestep=5)

level_set_fields = {"velocity": u}
projection_fields = {"target": level_set}
reinitialisation_fields = {"level_set_grad": level_set_grad_proj, "epsilon": epsilon}

# Forces gradient norm to satisfy reinitialisation equation
projection_boundary_conditions = fd.DirichletBC(
    vec_func_space_cg,
    level_set * (1 - level_set) / epsilon * fd.Constant((1, 0)),
    "on_boundary",
)

level_set_solver = TimeStepperSolver(
    level_set, level_set_fields, dt, SSPRK33, LevelSetEquation
)
projection_solver = ProjectionSolver(
    level_set_grad_proj, projection_fields, bcs=projection_boundary_conditions
)
reinitialisation_solver = TimeStepperSolver(
    level_set,
    reinitialisation_fields,
    dt / 50,  # Loose guess
    SSPRK33,
    ReinitialisationEquation,
    coupled_solver=projection_solver,
)

time_now, time_end = 0, 2000
dump_counter, dump_period = 0, 10
output_file = fd.File("level_set/output.pvd", target_degree=level_set_func_space_deg)

# Extract individual velocity and pressure fields and rename them for output
u_, p_ = z.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

output_time = []
rms_velocity = []
entrainment = []

# Perform the time loop
step = 0
reini_frequency = 1
reini_iterations = 5
while time_now < time_end:
    if time_now >= dump_counter * dump_period:
        dump_counter += 1
        output_file.write(level_set, u_, p_, level_set_grad_proj)

    dt = t_adapt.update_timestep()
    time_now += dt

    stokes_solver.solve()

    level_set_solver.solve()

    if step > reini_frequency:
        reinitialisation_solver.ts.solution_old.assign(level_set)

    if step > 0 and step % reini_frequency == 0:
        if conservative_level_set:
            for reini_step in range(reini_iterations):
                reinitialisation_solver.solve()

        level_set_solver.ts.solution_old.assign(level_set)

    output_time.append(time_now)
    rms_velocity.append(fd.norm(u) / fd.sqrt(domain_volume(mesh)))
    entrainment.append(
        fd.assemble(fd.conditional(mesh_coords[1] >= 0.2, 1 - rho * g, 0) * fd.dx)
        / lx
        / 0.2
    )

    step += 1
output_file.write(level_set, u_, p_, level_set_grad_proj)

if MPI.COMM_WORLD.rank == 0:
    np.savez(
        "output", time=output_time, rms_velocity=rms_velocity, entrainment=entrainment
    )

    fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

    ax[0].set_xlabel("Time (non-dimensional)")
    ax[1].set_xlabel("Time (non-dimensional)")
    ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
    ax[1].set_ylabel("Entrainment (non-dimensional)")

    ax[0].plot(output_time, rms_velocity)
    ax[1].plot(output_time, entrainment)

    fig.savefig("rms_velocity_and_entrainment.pdf", dpi=300, bbox_inches="tight")
