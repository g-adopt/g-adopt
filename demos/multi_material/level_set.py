import firedrake as fd
import flib
import matplotlib.pyplot as plt
import numpy as np
import shapely as sl
from mpi4py import MPI
from scipy.special import erf

from gadopt.approximations import BoussinesqApproximation
from gadopt.diagnostics import domain_volume
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


def initial_signed_distance_simple_curve():
    """Set up the initial signed distance function to the material interface"""

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


def initial_signed_distance_gerya():
    """Set up the initial signed distance function to the material interface"""

    square = sl.Polygon(
        [(2e5, 4.5e5), (3e5, 4.5e5), (3e5, 3.5e5), (2e5, 3.5e5), (2e5, 4.5e5)]
    )
    sl.prepare(square)

    node_relation_to_square = [
        (
            square.contains(sl.Point(x, y)) or square.boundary.contains(sl.Point(x, y)),
            square.boundary.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    node_sign_dist_to_square = [
        -dist if is_inside else dist for is_inside, dist in node_relation_to_square
    ]
    return node_sign_dist_to_square


def initial_signed_distance_schmalholz():
    interface_x = np.array([0, 4.6e5, 4.6e5, 5.4e5, 5.4e5, 1e6])
    interface_y = np.array([5.8e5, 5.8e5, 3.3e5, 3.3e5, 5.8e5, 5.8e5])
    curve = sl.LineString([*np.column_stack((interface_x, interface_y))])
    sl.prepare(curve)

    rectangle_lith = sl.Polygon(
        [(0, 6.6e5), (1e6, 6.6e5), (1e6, 5.8e5), (0, 5.8e5), (0, 6.6e5)]
    )
    sl.prepare(rectangle_lith)
    rectangle_slab = sl.Polygon(
        [(4.6e5, 5.8e5), (5.4e5, 5.8e5), (5.4e5, 3.3e5), (4.6e5, 3.3e5), (4.6e5, 5.8e5)]
    )
    sl.prepare(rectangle_slab)
    polygon_lith = sl.union(rectangle_lith, rectangle_slab)
    sl.prepare(polygon_lith)

    node_relation_to_curve = [
        (
            polygon_lith.contains(sl.Point(x, y))
            or polygon_lith.boundary.contains(sl.Point(x, y)),
            curve.distance(sl.Point(x, y)),
        )
        for x, y in zip(node_coords_x, node_coords_y)
    ]
    node_sign_dist_to_curve = [
        -dist if is_inside else dist for is_inside, dist in node_relation_to_curve
    ]
    return node_sign_dist_to_curve


def initialise_temperature(benchmark):
    """Set up the initial temperature field"""
    # Temperature function and associated function space
    temp_func_space = fd.FunctionSpace(mesh, "CG", 2)
    T = fd.Function(temp_func_space, name="Temperature")
    node_coords_x = fd.Function(temp_func_space).interpolate(mesh_coords[0]).dat.data
    node_coords_y = fd.Function(temp_func_space).interpolate(mesh_coords[1]).dat.data

    match benchmark:
        case "van_Keken_1997_isothermal":
            pass
        case "van_Keken_1997_thermochemical":
            u0 = (
                domain_length_x ** (7 / 3)
                / (1 + domain_length_x**4) ** (2 / 3)
                * (Ra.values().item() / 2 / np.sqrt(np.pi)) ** (2 / 3)
            )
            v0 = u0
            Q = 2 * np.sqrt(domain_length_x / np.pi / u0)
            Tu = erf((1 - node_coords_y) / 2 * np.sqrt(u0 / node_coords_x)) / 2
            Tl = 1 - 1 / 2 * erf(
                node_coords_y / 2 * np.sqrt(u0 / (domain_length_x - node_coords_x))
            )
            Tr = 1 / 2 + Q / 2 / np.sqrt(np.pi) * np.sqrt(
                v0 / (node_coords_y + 1)
            ) * np.exp(-(node_coords_x**2) * v0 / (4 * node_coords_y + 4))
            Ts = 1 / 2 - Q / 2 / np.sqrt(np.pi) * np.sqrt(
                v0 / (2 - node_coords_y)
            ) * np.exp(
                -((domain_length_x - node_coords_x) ** 2) * v0 / (8 - 4 * node_coords_y)
            )

            T.dat.data[:] = Tu + Tl + Tr + Ts - 3 / 2
            fd.DirichletBC(temp_func_space, 1, 3).apply(T)
            fd.DirichletBC(temp_func_space, 0, 4).apply(T)
            T.interpolate(fd.max_value(fd.min_value(T, 1), 0))
        case "Gerya_2003":
            pass
        case "Schmalholz_2011":
            pass
        case "Robey_2019":
            k = 1.5
            A = 0.05

            mask_bottom = node_coords_y <= 1 / 10
            mask_top = node_coords_y >= 9 / 10

            T.dat.data[:] = 0.5
            T.dat.data[mask_bottom] = (
                1
                - 5 * node_coords_y[mask_bottom]
                + A
                * np.sin(10 * np.pi * node_coords_y[mask_bottom])
                * (1 - np.cos(2 / 3 * k * np.pi * node_coords_x[mask_bottom]))
            )
            T.dat.data[mask_top] = (
                5
                - 5 * node_coords_y[mask_top]
                + A
                * np.sin(10 * np.pi * node_coords_y[mask_top])
                * (1 - np.cos(2 / 3 * k * np.pi * node_coords_x[mask_top] + np.pi))
            )
        case "Trim_2023":
            a = 100
            b = 100
            t = 0
            f = a * np.sin(np.pi * b * t)
            k = 35
            C0 = 1 / (1 + np.exp(-2 * k * (layer_interface_y - node_coords_y)))

            T.dat.data[:] = (
                -np.pi**3
                * (domain_length_x**2 + 1) ** 2
                / domain_length_x**3
                * np.cos(np.pi * node_coords_x / domain_length_x)
                * np.sin(np.pi * node_coords_y)
                * f
                + abs(Rb.values().item()) * C0
                + (Ra.values().item() - abs(Rb.values().item())) * (1 - node_coords_y)
            ) / Ra.values().item()

    return T


def write_output(dump_counter):
    time_output.assign(time_now)
    composition.interpolate(compo_field)
    viscosity.interpolate(visc)
    output_file.write(
        time_output,
        level_set,
        level_set_grad_proj,
        composition,
        u_,
        p_,
        viscosity,
        T,
        H,
    )

    dump_counter += 1
    return dump_counter


def diagnostics(benchmark, fields):
    if "van_Keken" in benchmark:
        fields["output_time"].append(time_now)
        fields["rms_velocity"].append(fd.norm(u) / fd.sqrt(domain_volume(mesh)))
        fields["entrainment"].append(
            fd.assemble(
                fd.conditional(mesh_coords[1] >= entrainment_height, compo_field, 0)
                * fd.dx
            )
            / domain_length_x
            / layer_interface_y
        )


def save_and_plot(benchmark, fields):
    if "van_Keken" in benchmark:
        np.savez(f"{benchmark.lower()}/output", fields=fields)

        fig, ax = plt.subplots(1, 2, figsize=(18, 10), constrained_layout=True)

        ax[0].set_xlabel("Time (non-dimensional)")
        ax[1].set_xlabel("Time (non-dimensional)")
        ax[0].set_ylabel("Root-mean-square velocity (non-dimensional)")
        ax[1].set_ylabel("Entrainment (non-dimensional)")

        ax[0].plot(fields["output_time"], fields["rms_velocity"])
        ax[1].plot(fields["output_time"], fields["entrainment"])

        fig.savefig(
            f"{benchmark.lower()}/rms_velocity_and_entrainment.pdf",
            dpi=300,
            bbox_inches="tight",
        )


benchmark = "Trim_2023"
match benchmark:
    case "van_Keken_1997_isothermal":
        domain_length_x, domain_length_y = 0.9142, 1
        layer_interface_y = domain_length_y / 5
        interface_deflection = domain_length_y / 50

        visc_ref, visc_compo = 1, 1

        Ra = fd.Constant(0)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        Rb = fd.Constant(-1)
        dens_diff = fd.Constant(1)

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

        post_process_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
        entrainment_height = domain_length_y / 5
    case "van_Keken_1997_thermochemical":
        domain_length_x, domain_length_y = 2, 1
        layer_interface_y = domain_length_y / 40
        interface_deflection = 0

        visc_ref, visc_compo = 1, 1

        Ra = fd.Constant(3e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        Rb = fd.Constant(4.5e5)
        dens_diff = fd.Constant(1)

        temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 5
        time_end = 0.05
        dump_period = 1e-4

        post_process_fields = {"output_time": [], "rms_velocity": [], "entrainment": []}
        entrainment_height = domain_length_y / 5
    case "Gerya_2003":
        domain_length_x, domain_length_y = 5e5, 5e5

        visc_ref, visc_compo = 1e21, 1e21

        Ra = fd.Constant(1)
        ref_dens = fd.Constant(3200)
        g = fd.Constant(9.8)
        Rb = fd.Constant(1)
        dens_diff = fd.Constant(100)

        temp_bcs = None
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e11)
        subcycles = 1
        time_end = 9.886e6 * 365.25 * 8.64e4
        dump_period = 1e5 * 365.25 * 8.64e4

        post_process_fields = {}
    case "Schmalholz_2011":
        domain_length_x, domain_length_y = 1e6, 6.6e5

        visc_ref = 1e21
        visc_coeff = 4.75e11
        stress_exponent = 4
        slab_length = 2.5e5

        Ra = fd.Constant(1)
        ref_dens = fd.Constant(3150)
        g = fd.Constant(9.81)
        Rb = fd.Constant(1)
        dens_diff = fd.Constant(150)

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

        post_process_fields = {}
        characteristic_time = (
            4 * visc_coeff / dens_diff.values().item() / g.values().item() / slab_length
        ) ** stress_exponent
    case "Robey_2019":
        domain_length_x, domain_length_y = 3, 1
        layer_interface_y = domain_length_y / 2
        interface_deflection = 0

        visc_ref, visc_compo = 1, 1

        Ra = fd.Constant(1e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        Rb = fd.Constant(2e4)
        dens_diff = fd.Constant(1)

        temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 1
        time_end = 0.025
        dump_period = 1e-4

        post_process_fields = {}
    case "Trim_2023":
        domain_length_x, domain_length_y = 1, 1
        layer_interface_y = domain_length_y / 2
        interface_deflection = 0

        visc_ref, visc_compo = 1, 1

        Ra = fd.Constant(1e5)
        ref_dens = fd.Constant(1)
        g = fd.Constant(1)
        Rb = fd.Constant(-5e4)
        dens_diff = fd.Constant(1)

        stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

        dt = fd.Constant(1e-6)
        subcycles = 1
        time_end = 0.01
        dump_period = 1e-4

        post_process_fields = {}
    case _:
        raise ValueError("Unknown benchmark.")

# Set up geometry
# Mesh resolution should be sufficient to capture the smaller-scale dynamics tracked by
# the level-set approach. Insufficient mesh refinement leads to the vanishing of the
# material interface during advection and to unwanted motion of the material interface
# during reinitialisation.
mesh_elements_x, mesh_elements_y = 16, 16
mesh = fd.RectangleMesh(
    mesh_elements_x,
    mesh_elements_y,
    domain_length_x,
    domain_length_y,
    quadrilateral=True,
)  # RectangleMesh boundary mapping: {1: left, 2: right, 3: bottom, 4: top}
mesh_coords = fd.SpatialCoordinate(mesh)

# Set up function spaces and functions used as part of the level-set approach
# Running "Schmalholz_2011" using DQ2 does not work.
level_set_func_space_deg = 1
func_space_dg = fd.FunctionSpace(mesh, "DQ", level_set_func_space_deg)
vec_func_space_cg = fd.VectorFunctionSpace(mesh, "CG", level_set_func_space_deg)
level_set = fd.Function(func_space_dg, name="Level set")
level_set_grad_proj = fd.Function(
    vec_func_space_cg, name="Level-set gradient (projection)"
)

# Initial interface layout
node_coords_x = fd.Function(func_space_dg).interpolate(mesh_coords[0]).dat.data
node_coords_y = fd.Function(func_space_dg).interpolate(mesh_coords[1]).dat.data
match benchmark:
    case "van_Keken_1997_isothermal":
        signed_dist_to_interface = initial_signed_distance_simple_curve()
    case "van_Keken_1997_thermochemical":
        signed_dist_to_interface = initial_signed_distance_simple_curve()
    case "Gerya_2003":
        signed_dist_to_interface = initial_signed_distance_gerya()
    case "Schmalholz_2011":
        signed_dist_to_interface = initial_signed_distance_schmalholz()
    case "Robey_2019":
        signed_dist_to_interface = initial_signed_distance_simple_curve()
    case "Trim_2023":
        signed_dist_to_interface = initial_signed_distance_simple_curve()

# Initialise level set
conservative_level_set = True
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

stokes_function = fd.Function(stokes_func_space)
u, p = fd.split(stokes_function)  # Symbolic UFL expression for velocity and pressure

if benchmark == "Schmalholz_2011":
    strain_rate = fd.sym(fd.grad(u))
    strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2)
    visc_compo = fd.max_value(
        fd.min_value(
            visc_coeff * strain_rate_sec_inv ** (1 / stress_exponent - 1), 1e25
        ),
        1e21,
    )
compo_field = fd.conditional(level_set > level_set_contour, 0, 1)
visc = fd.conditional(level_set > level_set_contour, visc_ref, visc_compo)
composition = fd.Function(pres_func_space, name="Composition")
viscosity = fd.Function(pres_func_space, name="Viscosity")

T = initialise_temperature(benchmark)

temp_func_space = T.function_space()
H = fd.Function(temp_func_space, name="Internal heating rate")
if benchmark == "Trim_2023":
    node_coords_x = fd.Function(temp_func_space).interpolate(mesh_coords[0]).dat.data[:]
    node_coords_y = fd.Function(temp_func_space).interpolate(mesh_coords[1]).dat.data[:]

    k = 35

    H_values = []
    for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y):
        H_values.append(
            flib.h_python(
                node_coord_x,
                node_coord_y,
                0,
                domain_length_x,
                k,
                layer_interface_y,
                Ra.values().item(),
                abs(Rb.values().item()),
            )
        )
    H.dat.data[:] = H_values

    C0_0 = 1 / (1 + np.exp(-2 * k * (layer_interface_y - 0)))
    C0_1 = 1 / (1 + np.exp(-2 * k * (layer_interface_y - 1)))

    temp_bcs = {
        3: {"T": abs(Rb.values().item()) / Ra.values().item() * (C0_0 - 1) + 1},
        4: {"T": abs(Rb.values().item()) / Ra.values().item() * C0_1},
    }

# Set up energy and Stokes solvers
approximation = BoussinesqApproximation(
    Ra, rho=ref_dens, g=g, Rb=Rb, delta_rho=dens_diff, H=H
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
    1e-2,  # Trade-off between actual reinitialisation and unwanted interface motion
    eSSPRKs3p3,
    ReinitialisationEquation,
    coupled_solver=projection_solver,
)

# Time-loop objects
t_adapt = TimestepAdaptor(dt, u, vel_func_space, target_cfl=subcycles * 0.1)
time_output = fd.Function(pres_func_space, name="Time")
time_now, step, dump_counter = 0, 0, 0
reini_frequency, reini_iterations = 1, 2
output_file = fd.File(
    f"{benchmark.lower()}/output.pvd",
    target_degree=level_set_func_space_deg,
)

# Extract individual velocity and pressure fields and rename them for output
u_, p_ = stokes_function.subfunctions
u_.rename("Velocity")
p_.rename("Pressure")

# Perform the time loop
while time_now < time_end:
    if time_now >= dump_counter * dump_period:  # Write to output file
        dump_counter = write_output(dump_counter)

    if benchmark == "Trim_2023":
        H_values.clear()
        for node_coord_x, node_coord_y in zip(node_coords_x, node_coords_y):
            H_values.append(
                flib.h_python(
                    node_coord_x,
                    node_coord_y,
                    time_now,
                    domain_length_x,
                    k,
                    layer_interface_y,
                    Ra.values().item(),
                    abs(Rb.values().item()),
                )
            )
        H.dat.data[:] = H_values

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

    diagnostics(benchmark, post_process_fields)

    # Increment time-loop step counter
    step += 1
# Write final simulation state to output file
dump_counter = write_output(dump_counter)

if MPI.COMM_WORLD.rank == 0:  # Save post-processing fields and produce graphs
    save_and_plot(benchmark, post_process_fields)
