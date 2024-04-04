from gadopt import *
from gadopt.inverse import *
import sys
sys.path.append("../benchmarks/")
from schmalholz_2011 import Simulation
import firedrake as fd
from mpi4py import MPI

# constructing mesh
mesh = Mesh('../benchmarks/schmalholz_2011.msh')
# Set up Stokes function spaces corresponding to the mixed Q2Q1 Taylor-Hood element
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])
z = Function(Z, name="Stokes")

# Define temperature function space and initialise temperature
Q = FunctionSpace(mesh, "CG", 2)
temp = Function(Q, name="Temperature")
controls = [Control(temp)]

# # TODO: initialise
# Simulation.initialise_temperature(temp)

# Set up function spaces and functions used in the level-set approach
LS = FunctionSpace(mesh, "DQ", Simulation.level_set_func_space_deg)
level_set = [
    Function(LS, name=f"Level set #{i}")
    for i in range(len(Simulation.materials) - 1)
]
for ctrl in [Control(ls) for ls in level_set]:
    controls.append(ctrl)

local_min_mesh_size = mesh.cell_sizes.dat.data.min()
epsilon = Constant(mesh.comm.allreduce(local_min_mesh_size, MPI.MIN) / 4)

# Initialise level set
signed_dist_to_interface = fd.Function(level_set[0].function_space())
for ls, isd, params in zip(
    level_set, Simulation.initialise_signed_distance, Simulation.isd_params
):
    signed_dist_to_interface.dat.data[:] = isd(params, ls)
    ls.interpolate((1 + fd.tanh(signed_dist_to_interface / 2 / epsilon)) / 2)

time_output = Function(W, name="Time")
time_now = 0
dump_counter = 0

# Set up fields that depend on the material interface
func_space_interp = FunctionSpace(mesh, "CG", Simulation.level_set_func_space_deg)


ref_dens, dens_diff, density, RaB_ufl, RaB, dimensionless = density_RaB(
    Simulation, level_set, func_space_interp
)


u, p = split(z)  # UFL expressions
u_, p_ = z.subfunctions  # Associated Firedrake functions
u_.rename("Velocity")
p_.rename("Pressure")

viscosity_ufl = field_interface(
    level_set,
    [
        material.viscosity(u, temp)/1e21
        for material in Simulation.materials
    ],
    method="sharp" if "Schmalholz_2011" in Simulation.name else "geometric",
)
viscosity = Function(func_space_interp, name="Viscosity").interpolate(viscosity_ufl)

int_heat_rate_ufl = field_interface(
    level_set,
    [material.internal_heating_rate() for material in Simulation.materials],
    method="geometric",
)
int_heat_rate = Function(
    func_space_interp, name="Internal heating rate"
).interpolate(int_heat_rate_ufl)

# Timestep object
timestep = Constant(1e-9)

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
    temp,
    u,
    approximation,
    timestep,
    ImplicitMidpoint,
    bcs=Simulation.temp_bcs,
)
stokes_nullspace = create_stokes_nullspace(
    Z, **Simulation.stokes_nullspace_args
)
stokes_solver = StokesSolver(
    z,
    temp,
    approximation,
    bcs=Simulation.stokes_bcs,
    mu=viscosity_ufl,
    quad_degree=None,
    solver_parameters="newton",
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
)

# Parameters involved in level-set reinitialisation
reini_params = {
    "epsilon": epsilon,
    "tstep": 1e-2,
    "tstep_alg": eSSPRKs3p3,
    "frequency": 5,
    "iterations": 0,
}

# Set up level-set solvers
level_set_solver = [
    LevelSetSolver(
        ls, u, timestep, eSSPRKs10p3, Simulation.subcycles, reini_params
    )
    for ls in level_set
]
level_set_grad_proj = [ls_solv.level_set_grad_proj for ls_solv in level_set_solver]

output_file = fd.output.VTKFile(
    "test.pvd",
    target_degree=Simulation.level_set_func_space_deg,
)


# Objects used to calculate simulation diagnostics
diag_vars = {
    "epsilon": epsilon,
    "level_set": level_set,
    "level_set_grad_proj": level_set_grad_proj,
    "density": density,
    "viscosity": viscosity,
    "int_heat_rate": int_heat_rate,
}

update_forcings = None
# Perform the time loop
step = 0
for step in range(5):
    output_file.write(u_, p_, temp, *level_set, viscosity)

    # Solve Stokes system
    stokes_solver.solve()
    # Solve energy system
    energy_solver.solve(t=time_now, update_forcings=update_forcings)

    # Advect each level set
    for ls_solv in level_set_solver:
        ls_solv.solve(step)
    viscosity.interpolate(viscosity_ufl)

objective = assemble(0.5 * level_set[0] ** 2 * dx)
print(objective)
rf = ReducedFunctional(
    objective,
    controls)
print(rf([temp, *level_set]))
print(rf([temp, *level_set]))
