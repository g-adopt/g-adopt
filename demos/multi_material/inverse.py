from importlib import import_module
from pathlib import Path
from gadopt import *
from gadopt.inverse import *

Simulation = import_module("benchmarks.schmalholz_2011").Simulation

# Loading the necessary fields
checkpoint_path = Path(__file__).resolve().parent / "schmalholz_2011/checkpoint_1.h5"
with CheckpointFile(str(checkpoint_path), "r",) as fi:
    mesh = fi.load_mesh("firedrake_default")
    level_set_obs = fi.load_function(mesh, "Level set #0", idx=25)
    level_set_obs.rename("Level set Reference")


# Velocity and Pressure
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])
z = Function(Z, name="stokes")

# temperature
Q = FunctionSpace(mesh, "CG", 2)
T = Function(Q, name="Temperature").assign(0.0)

# initiating level-set
level_set = Function(level_set_obs.function_space(), name="Level set #0")
control = Control(level_set)
level_set.assign(level_set_obs)

# Extract velocity and pressure from the Stokes function
u, p = split(z)  # UFL expressions
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# Thickness of the hyperbolic tangent profile in the conservative level-set approach
local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)
# epsilon = Constant(3867.21543934)

R = FunctionSpace(mesh, "R", 0)
timestep = Function(R, name="timestep")
timestep.assign(1e5)

# Set up fields that depend on the material interface
ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless = density_RaB(
    Simulation, [level_set], Q)

viscosity_ufl = field_interface(
    [level_set],
    [material.viscosity(u, T) for material in Simulation.materials],
    method="sharp" if "Schmalholz_2011" in Simulation.name else "geometric",
)
viscosity = Function(Q, name="Viscosity").interpolate(viscosity_ufl)

int_heat_rate_ufl = field_interface(
    [level_set],
    [material.internal_heating_rate() for material in Simulation.materials],
    method="geometric",
)
int_heat_rate = Function(Q, name="Internal heating rate")
int_heat_rate.interpolate(int_heat_rate_ufl)

# Set up energy and Stokes solvers
approximation = BoussinesqApproximation(
    Simulation.Ra,
    rho=ref_dens,
    alpha=1,
    g=Simulation.g,
    T0=0,
    RaB=RaB_ufl,
    delta_rho=dens_diff,
    kappa=1,
    H=int_heat_rate_ufl,
)

energy_solver = EnergySolver(
    T,
    u,
    approximation,
    timestep,
    ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = create_stokes_nullspace(
    z.function_space(), **Simulation.stokes_nullspace_args
)
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=Simulation.stokes_bcs,
    mu=viscosity_ufl,
    quad_degree=None,
    solver_parameters=Simulation.stokes_solver_params,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Set up level-set solvers
ls_solver = LevelSetSolver(level_set, u, timestep, eSSPRKs10p3, Simulation.subcycles, epsilon)
ls_solver.reini_params["tstep"] *= 10

# Perform the time loop
step = 0
time_now = Function(R, name="time")
time_now.assign(0.)

for i in range(0, 100):
    # Solve Stokes system
    stokes_solver.solve()

    # Solve energy system
    energy_solver.solve(t=float(time_now))

    # Advect each level set
    ls_solver.solve(step)

    time_now.assign(time_now + timestep)

    step += 1


# Defining the objective functional
objective = assemble((level_set - level_set_obs) ** 2 * dx)

log(f"Value of the objective from forward {objective}")

# We do not want to annotate anymore
stop_annotating()

#
reduced_functional = ReducedFunctional(objective, control)
log(f"Value of the reduced functional call {reduced_functional([level_set])}")

# reduced_functional.derivative()
#
# ls_lower_bound = Function(ls_control[0].function_space()).assign(0)
# ls_upper_bound = Function(ls_control[0].function_space()).assign(1)
#
# minimisation_problem = MinimizationProblem(
#     reduced_functional, bounds=(ls_lower_bound, ls_upper_bound)
# )
#
# optimiser = LinMoreOptimiser(minimisation_problem, minimisation_parameters)
# optimiser.run()
