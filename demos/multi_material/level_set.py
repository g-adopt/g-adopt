import firedrake as fd
import numpy as np
from mpi4py import MPI

from gadopt.approximations import BoussinesqApproximation
from gadopt.energy_solver import EnergySolver
from gadopt.level_set_tools import (
    LevelSetEquation,
    ProjectionSolver,
    ReinitialisationEquation,
    TimeStepperSolver,
)
from gadopt.stokes_integrators import StokesSolver, create_stokes_nullspace
from gadopt.time_stepper import ImplicitMidpoint, eSSPRKs3p3, eSSPRKs10p3
from gadopt.utility import TimestepAdaptor

import flib
from helper import node_coordinates
from initial_signed_distance import initialise_signed_distance
from initial_temperature import initialise_temperature
from post_processing import diagnostics, save_and_plot


def non_linear_viscosity(u):
    strain_rate = fd.sym(fd.grad(u))
    strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2)

    return fd.min_value(
        fd.max_value(
            visc_coeff * strain_rate_sec_inv ** (1 / stress_exponent - 1), 1e21
        ),
        1e25,
    )


def update_internal_heating_rate(int_heat_rate, simu_time):
    node_coords_x, node_coords_y = node_coordinates(int_heat_rate)

    analytical_values = []
    for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y):
        analytical_values.append(
            flib.h_python(
                node_coord_x,
                node_coord_y,
                simu_time,
                domain_length_x,
                k,
                layer_interface_y,
                Ra,
                RaB,
            )
        )

    int_heat_rate.dat.data[:] = analytical_values


def write_output(dump_counter):
    time_output.assign(time_now)
    composition.interpolate(compo_field)
    viscosity.interpolate(visc)

    output_file.write(
        time_output,
        level_set,
        level_set_grad_proj,
        composition,
        velocity,
        pressure,
        viscosity,
        T,
        int_heat_rate,
    )

    return dump_counter + 1


# Reinitialisation is not implemented for regular signed-distance level set
conservative_level_set = True

benchmark = "Trim_2023"
output_dir = benchmark.lower()
match benchmark:
    case "van_Keken_1997_isothermal":
        domain_length_x, domain_length_y = 0.9142, 1
        layer_interface_y = domain_length_y / 5
        interface_deflection = domain_length_y / 50
        isd_params = (domain_length_x, layer_interface_y, interface_deflection)

        compo_0, compo_1 = 0, 1
        visc_0, visc_1 = 1, 1

        Ra = fd.Constant(0)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        RaB = fd.Constant(1)
        dens_diff = fd.Constant(1)

        ini_temp_params = None

        temp_bcs = None
        stokes_bcs = {
            1: {"ux": 0},
            2: {"ux": 0},
            3: {"ux": 0, "uy": 0},
            4: {"ux": 0, "uy": 0},
        }

        dt = fd.Constant(1)
        subcycles = 1
        time_end = 2000
        dump_period = 10

        diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
        entrainment_height = domain_length_y / 5
        diag_params = {
            "domain_length_x": domain_length_x,
            "layer_interface_y": layer_interface_y,
            "entrainment_height": entrainment_height,
        }
    case "van_Keken_1997_thermochemical":
        domain_length_x, domain_length_y = 2, 1
        layer_interface_y = domain_length_y / 40
        interface_deflection = 0
        isd_params = (domain_length_x, layer_interface_y, interface_deflection)

        compo_0, compo_1 = 1, 0
        visc_0, visc_1 = 1, 1

        Ra = fd.Constant(3e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        RaB = fd.Constant(4.5e5)
        dens_diff = fd.Constant(1)

        ini_temp_params = (Ra.values().item(), domain_length_x)

        temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 5
        time_end = 0.05
        dump_period = 1e-4

        diag_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
        entrainment_height = domain_length_y / 5
        diag_params = {
            "domain_length_x": domain_length_x,
            "layer_interface_y": layer_interface_y,
            "entrainment_height": entrainment_height,
        }
    case "Gerya_2003":
        domain_length_x, domain_length_y = 5e5, 5e5
        ref_vertex_x = 2e5
        ref_vertex_y = 3.5e5
        edge_length = 1e5
        isd_params = (ref_vertex_x, ref_vertex_y, edge_length)

        compo_0, compo_1 = 0, 1
        visc_0, visc_1 = 1e21, 1e21

        Ra = fd.Constant(1)
        ref_dens = fd.Constant(3200)
        g = fd.Constant(9.8)
        RaB = fd.Constant(1)
        dens_diff = fd.Constant(100)

        ini_temp_params = None

        temp_bcs = None
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e11)
        subcycles = 1
        time_end = 9.886e6 * 365.25 * 8.64e4
        dump_period = 1e5 * 365.25 * 8.64e4

        diag_fields = {}
        diag_params = {}
    case "Schmalholz_2011":
        domain_length_x, domain_length_y = 1e6, 6.6e5
        isd_params = None

        compo_0, compo_1 = 0, 1
        visc_0 = 1e21
        visc_coeff = 4.75e11
        stress_exponent = 4
        slab_length = 2.5e5

        Ra = fd.Constant(1)
        ref_dens = fd.Constant(3150)
        g = fd.Constant(9.81)
        RaB = fd.Constant(1)
        dens_diff = fd.Constant(150)

        ini_temp_params = None

        temp_bcs = None
        stokes_bcs = {
            1: {"ux": 0, "uy": 0},
            2: {"ux": 0, "uy": 0},
            3: {"uy": 0},
            4: {"uy": 0},
        }

        dt = fd.Constant(1e11)
        subcycles = 1
        time_end = 25e6 * 365.25 * 8.64e4
        dump_period = 5e5 * 365.25 * 8.64e4

        diag_fields = {}
        diag_params = {}
        characteristic_time = (
            4 * visc_coeff / dens_diff.values().item() / g.values().item() / slab_length
        ) ** stress_exponent
    case "Robey_2019":
        domain_length_x, domain_length_y = 3, 1
        layer_interface_y = domain_length_y / 2
        interface_deflection = 0
        isd_params = (domain_length_x, layer_interface_y, interface_deflection)

        compo_0, compo_1 = 1, 0
        visc_0, visc_1 = 1, 1

        Ra = fd.Constant(1e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        RaB = fd.Constant(1e4)
        dens_diff = fd.Constant(1)

        A = 0.05
        k = 1.5
        ini_temp_params = (A, k)

        temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 1
        time_end = 0.025
        dump_period = 1e-4

        diag_fields = {}
        diag_params = {}
    case "Trim_2023":  # Requires https://github.com/seantrim/exact-thermochem-solution
        domain_length_x, domain_length_y = 1, 1
        layer_interface_y = domain_length_y / 2
        interface_deflection = 0
        isd_params = (domain_length_x, layer_interface_y, interface_deflection)

        compo_0, compo_1 = 1, 0
        visc_0, visc_1 = 1, 1

        Ra = fd.Constant(1e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        RaB = fd.Constant(5e4)
        dens_diff = fd.Constant(1)

        a = 100
        b = 100
        t = 0
        f = a * np.sin(np.pi * b * t)
        k = 35

        ini_temp_params = (
            Ra.values().item(),
            RaB.values().item(),
            f,
            k,
            layer_interface_y,
            domain_length_x,
        )

        temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 1
        time_end = 0.01
        dump_period = 1e-4

        diag_fields = {}
        diag_params = {}
    case _:
        raise ValueError("Unknown benchmark.")

# Set up geometry
# Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
# the level-set approach. Insufficient mesh refinement leads to the vanishing of the
# material interface during advection and to unwanted motion of the material interface
# during reinitialisation.
mesh_elements_x, mesh_elements_y = 32, 32
mesh = fd.RectangleMesh(
    mesh_elements_x,
    mesh_elements_y,
    domain_length_x,
    domain_length_y,
    quadrilateral=True,
)  # RectangleMesh boundary mapping: {1: left, 2: right, 3: bottom, 4: top}

# Set up function spaces and functions used as part of the level-set approach
# Running "Schmalholz_2011" using DQ2 does not work.
level_set_func_space_deg = 2
func_space_dq = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
vec_func_space_cg = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = fd.Function(func_space_dq, name="Level set")
level_set_grad_proj = fd.Function(
    vec_func_space_cg, name="Level-set gradient (projection)"
)

# Initialise level set
signed_dist_to_interface = initialise_signed_distance(level_set, benchmark, isd_params)
if conservative_level_set:
    epsilon = fd.Constant(
        min(
            domain_length_x / mesh_elements_x,
            domain_length_y / mesh_elements_y,
        )
        / 4
    )  # Empirical calibration that seems to be robust
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
# Define Stokes functions on the mixed function space
stokes_function = fd.Function(stokes_func_space)
u, p = fd.split(stokes_function)  # Symbolic UFL expression for velocity and pressure
velocity, pressure = stokes_function.subfunctions  # Associated UFL functions
velocity.rename("Velocity")
pressure.rename("Pressure")

# Set up fields that depend on the material interface
compo_field = fd.conditional(level_set > level_set_contour, compo_1, compo_0)
composition = fd.Function(pres_func_space, name="Composition")
if benchmark == "Schmalholz_2011":
    visc_1 = non_linear_viscosity(u)
visc = fd.conditional(level_set > level_set_contour, visc_1, visc_0)
viscosity = fd.Function(pres_func_space, name="Viscosity")

# Temperature function and associated function space
temp_func_space = fd.FunctionSpace(mesh, "CG", 2)
T = fd.Function(temp_func_space, name="Temperature")
initialise_temperature(T, benchmark, ini_temp_params)
# Internal heating rate
int_heat_rate = fd.Function(temp_func_space, name="Internal heating rate")
if benchmark == "Trim_2023":
    update_internal_heating_rate(int_heat_rate, 0)

# Set up energy and Stokes solvers
approximation = BoussinesqApproximation(
    Ra, rho=ref_dens, g=g, RaB=RaB, delta_rho=dens_diff, H=int_heat_rate
)
energy_solver = EnergySolver(T, u, approximation, dt, ImplicitMidpoint, bcs=temp_bcs)
stokes_nullspace = create_stokes_nullspace(stokes_func_space)
stokes_solver = StokesSolver(
    stokes_function,
    T,
    approximation,
    bcs=stokes_bcs,
    mu=visc,
    C=compo_field,
    nullspace=stokes_nullspace,
    transpose_nullspace=stokes_nullspace,
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
    level_set, level_set_fields, dt / subcycles, eSSPRKs10p3, LevelSetEquation
)
projection_solver = ProjectionSolver(
    level_set_grad_proj, projection_fields, bcs=projection_boundary_conditions
)
reinitialisation_solver = TimeStepperSolver(
    level_set,
    reinitialisation_fields,
    2e-2,  # Trade-off between actual reinitialisation and unwanted interface motion
    eSSPRKs3p3,
    ReinitialisationEquation,
    coupled_solver=projection_solver,
)

# Time-loop objects
t_adapt = TimestepAdaptor(dt, u, vel_func_space, target_cfl=subcycles * 0.2)
time_output = fd.Function(pres_func_space, name="Time")
time_now, step, dump_counter = 0, 0, 0
reini_frequency, reini_iterations = 1, 2
output_file = fd.File(
    f"{output_dir}/output.pvd", target_degree=level_set_func_space_deg
)

diag_vars = {
    "level_set": level_set,
    "level_set_grad_proj": level_set_grad_proj,
    "composition": composition,
    "velocity": u,
    "pressure": p,
    "viscosity": viscosity,
    "temperature": T,
    "int_heat_rate": int_heat_rate,
}

# Perform the time loop
while time_now < time_end:
    if time_now >= dump_counter * dump_period:  # Write to output file
        dump_counter = write_output(dump_counter)

    if benchmark == "Trim_2023":
        update_internal_heating_rate(int_heat_rate, time_now)

    # Update time and timestep
    dt = t_adapt.update_timestep()
    time_now += dt

    # Solve Stokes system
    stokes_solver.solve()

    # Solve energy system
    energy_solver.solve()

    for subcycle in range(subcycles):
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

    diagnostics(time_now, benchmark, diag_fields, diag_vars, diag_params)

    # Increment time-loop step counter
    step += 1
# Write final simulation state to output file
dump_counter = write_output(dump_counter)

if MPI.COMM_WORLD.rank == 0:  # Save post-processing fields and produce graphs
    save_and_plot(benchmark, output_dir, diag_fields)
