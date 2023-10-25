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


def initial_signed_distance():
    """Determine the signed distance to the material interface for each mesh node"""

    def get_interface_y(interface_x):
        return layer_interface_y + interface_deflection * np.cos(
            np.pi * interface_x / domain_length_x
        )

    interface_x = np.linspace(0, domain_length_x, 1000)
    interface_y = get_interface_y(interface_x)
    curve = sl.LineString([*np.column_stack((interface_x, interface_y))])
    sl.prepare(curve)

    node_relation_to_curve = [
        (
            node_coord_y > get_interface_y(node_coord_x),
            curve.distance(sl.Point(node_coord_x, node_coord_y)),
        )
        for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y)
    ]
    node_sign_dist_to_curve = [
        dist if is_above else -dist for is_above, dist in node_relation_to_curve
    ]
    return node_sign_dist_to_curve


# Set up geometry
mesh_elements_x, mesh_elements_y = 64, 64
domain_length_x, domain_length_y = 0.9142, 1
mesh = fd.RectangleMesh(
    mesh_elements_x,
    mesh_elements_y,
    domain_length_x,
    domain_length_y,
    quadrilateral=True,
)
mesh_coords = fd.SpatialCoordinate(mesh)

# Set up function spaces and functions used as part of the level-set approach
level_set_func_space_deg = 2
func_space_dg = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
vec_func_space_cg = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = fd.Function(func_space_dg, name="level_set")
level_set_grad_proj = fd.Function(vec_func_space_cg, name="level_set_grad_proj")
node_coords_x = fd.Function(func_space_dg).interpolate(mesh_coords[0]).dat.data
node_coords_y = fd.Function(func_space_dg).interpolate(mesh_coords[1]).dat.data

# Initial interface layout
layer_interface_y = domain_length_y / 5
interface_deflection = domain_length_y / 50
signed_dist_to_interface = initial_signed_distance()

# Initialise level set
conservative_level_set = True
if conservative_level_set:
    epsilon = fd.Constant(
        min(domain_length_x / mesh_elements_x, domain_length_y / mesh_elements_y) / 4
    )  # Loose guess
    level_set.dat.data[:] = (
        np.tanh(np.asarray(signed_dist_to_interface) / 2 / epsilon.values().item()) + 1
    ) / 2
    level_set_contour = 0.5
else:
    level_set.dat.data[:] = signed_dist_to_interface
    level_set_contour = 0

# Set up Stokes function spaces corresponding to the bilinear Q2Q1 element pair
vel_func_space = fd.VectorFunctionSpace(mesh, "CG", 2)
pres_func_space = fd.FunctionSpace(mesh, "CG", 1)
stokes_func_space = fd.MixedFunctionSpace([vel_func_space, pres_func_space])

stokes_function = fd.Function(stokes_func_space)
u, p = fd.split(stokes_function)  # Symbolic UFL expression for velocity and pressure

# Physical parameters; since they are included in UFL, they are wrapped inside Constant
g = fd.Constant(10)
Ra = fd.Constant(-1)
T = fd.Constant(1)

# Physical fields depending on the level set
rho = fd.conditional(level_set > level_set_contour, 1 / g, 0 / g)
mu = fd.conditional(level_set > level_set_contour, 1, 1)

# Set up Stokes solver
approximation = BoussinesqApproximation(Ra, g=g, rho=rho)
stokes_nullspace = create_stokes_nullspace(stokes_func_space)
stokes_bcs = {
    1: {"ux": 0},
    2: {"ux": 0},
    3: {"ux": 0, "uy": 0},
    4: {"ux": 0, "uy": 0},
}  # RectangleMesh boundary mapping: {1: left, 2: right, 3: bottom, 4: top}
stokes_solver = StokesSolver(
    stokes_function,
    T,
    approximation,
    bcs=stokes_bcs,
    mu=mu,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Set up timestepping objects
dt = fd.Constant(1)  # Initial time-step
target_cfl = (
    2
    * (domain_length_y / mesh_elements_y) ** (1 / 10)
    / (level_set_func_space_deg + 1) ** (5 / 3)
)  # Empirical calibration
t_adapt = TimestepAdaptor(
    dt, u, vel_func_space, target_cfl=target_cfl, maximum_timestep=5
)

# Additional fields required for each level-set solver
level_set_fields = {"velocity": u}
projection_fields = {"target": level_set}
reinitialisation_fields = {"level_set_grad": level_set_grad_proj, "epsilon": epsilon}

# Force projected gradient norm to satisfy the reinitialisation equation
projection_boundary_conditions = fd.DirichletBC(
    vec_func_space_cg,
    level_set * (1 - level_set) / epsilon * fd.Constant((1, 0)),
    "on_boundary",
)

# Set up level-set solvers
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

# Time-loop objects
time_now, time_end = 0, 2000
dump_counter, dump_period = 0, 10
output_file = fd.File("level_set/output.pvd", target_degree=level_set_func_space_deg)

# Extract individual velocity and pressure fields and rename them for output
u_, p_ = stokes_function.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

# Relevant post-processing fields
output_time = []
rms_velocity = []
entrainment = []

# Perform the time loop
step = 0
reini_frequency = 1
reini_iterations = 5
while time_now < time_end:
    if time_now >= dump_counter * dump_period:  # Write to output file
        dump_counter += 1
        output_file.write(level_set, u_, p_, level_set_grad_proj)

    # Update time and timestep
    dt = t_adapt.update_timestep()
    time_now += dt

    # Solve Stokes system
    stokes_solver.solve()

    # Solve level-set advection
    level_set_solver.solve()

    if step > reini_frequency:  # Update stored function given previous solve
        reinitialisation_solver.ts.solution_old.assign(level_set)

    if step > 0 and step % reini_frequency == 0:  # Solve level-set reinitialisation
        if conservative_level_set:
            for reini_step in range(reini_iterations):
                reinitialisation_solver.solve()

        # Update stored function given previous solve
        level_set_solver.ts.solution_old.assign(level_set)

    # Calculate and store post-processing fields
    output_time.append(time_now)
    rms_velocity.append(fd.norm(u) / fd.sqrt(domain_volume(mesh)))
    entrainment.append(
        fd.assemble(
            fd.conditional(mesh_coords[1] >= layer_interface_y, 1 - rho * g, 0) * fd.dx
        )
        / domain_length_x
        / layer_interface_y
    )

    # Increment time-loop step counter
    step += 1
# Write final simulation state to output file
output_file.write(level_set, u_, p_, level_set_grad_proj)

if MPI.COMM_WORLD.rank == 0:  # Save post-processing fields and produce graphs
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
