from importlib import import_module
import pdb
from pathlib import Path
from gadopt import *
from gadopt.inverse import *

Simulation = import_module("benchmarks.schmalholz_2011").Simulation

# Loading the necessary fields
checkpoint_path = Path(__file__).resolve().parent / "schmalholz_2011/checkpoint_1.h5"
with CheckpointFile(str(checkpoint_path), "r",) as fi:
    mesh = fi.load_mesh("firedrake_default")
    time_now_dump = fi.load_function(mesh, "Time", idx=25)
    timestep_dump = fi.load_function(mesh, "Time step", idx=25)
    level_set_obs = fi.load_function(mesh, "Level set #0", idx=25)

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
# local_min_mesh_size = mesh.cell_sizes.dat.data.min()
# epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)
epsilon = Constant(3867.21543934)

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
    un,
    approximation,
    timestep,
    ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = create_stokes_nullspace(
    stokes_function.function_space(), **Simulation.stokes_nullspace_args
)
stokes_solver = StokesSolver(
    stokes_function,
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
# level_set_solver[0].reini_params["tstep"] *= 10
# level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

update_forcings = None

# Perform the time loop
step = 0
while True:
    # Solve Stokes system
    stokes_solver.solve()

    # Solve energy system
    energy_solver.solve(t=float(time_now), update_forcings=update_forcings)

    # Advect each level set
    ls_solver.solve(step)

    step += 1

objective = assemble((level_set - level_set_obs) ** 2 * dx)

reduced_functional = ReducedFunctional(objective, control)
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
