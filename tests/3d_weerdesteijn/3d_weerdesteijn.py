# Test case based on Weerdesteijn et al 2023 benchmark.
# A 100 km thick ice sheet is placed on the corner of a
# 1500 km x 1500 km x 2891 km box. There is a short
# (200 yr) and long loading scenario (110 kyr).
# The rheological properties are based on the Spada et
# al. 2011 testcases. There is also a case with a low
# viscosity region situated under the ice sheet.
# results within G-ADOPT see Scott et al. 2025.

# Weerdesteijn, M. F., Naliboff, J. B., Conrad,
# C. P., Reusen, J. M., Steffen, R., Heister, T., &
# Zhang, J. (2023). Modeling viscoelastic solid
# earth deformation due to ice age and contemporary
# glacial mass changes in ASPECT. Geochemistry,
# Geophysics, Geosystems, 24(3), e2022GC010813.

# Spada, G., Barletta, V. R., Klemann, V., Riva, R. E. M.,
# Martinec, Z., Gasperini, P., ... & King, M. A. (2011).
# A benchmark study for glacial isostatic adjustment codes.
# Geophysical Journal International, 185(1), 106-132.

# Automated forward and adjoint modelling of
# viscoelastic deformation of the solid Earth.
# Scott, W.; Hoggard, M.; Duvernay, T.;
# Ghelichkhan, S.; Gibson, A.; Roberts, D.;
# Kramer, S. C.; and Davies, D. R.
# EGUsphere, 2025: 1–43. 2025.

from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
from gadopt.utility import extruded_layer_heights
from gadopt.utility import initialise_background_field
from gadopt.utility import vertical_component as vc
import argparse
import numpy as np
from mpi4py import MPI
parser = argparse.ArgumentParser()
parser.add_argument("--dx", default=20, type=float,
                    help="Horizontal resolution in km", required=False)
parser.add_argument("--refined_surface", action='store_true',
                    help="Use refined surface mesh")
parser.add_argument("--const_aspect", action='store_true',
                    help="Also scale farfield dx for parallel scaling test to keep same aspect ratio")
parser.add_argument("--structured_dz", action='store_true',
                    help="Use constant vertical resolution")
parser.add_argument("--nz", default=100, type=int,
                    help="Number of vertical layers for structured dz")
parser.add_argument("--DG0_layers", default=5, type=int,
                    help="Number of cells per layer for DG0 discretisation of background profiles",
                    required=False)
parser.add_argument("--dt_years", default=1e3, type=float,
                    help="Timestep in years", required=False)
parser.add_argument("--dt_out_years", default=10e3, type=float,
                    help="Output timestep in years", required=False)
parser.add_argument("--Tend", default=110e3, type=float,
                    help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=100, type=float,
                    help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--Tstart", default=0, type=float,
                    help="Simulation start time in years", required=False)
parser.add_argument("--short_simulation", action='store_true',
                    help="Run simulation with short ice history from Weerdesteijn et a. 2023 testcase")
parser.add_argument("--lateral_viscosity", action='store_true',
                    help="Include low viscosity cylinder from Weerdesteijn et a. 2023 testcase")
parser.add_argument("--burgers", action='store_true',
                    help="Use a burgers rheology")
parser.add_argument("--viscosity_ratio", default=1, type=float,
                    help="Ratio of viscosity 2 / viscosity 1", required=False)
parser.add_argument("--write_output", action='store_true',
                    help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str,
                    help="Optional string to add to simulation name for outputs",
                    required=False)
parser.add_argument("--output_path", default="./", type=str,
                    help="Optional output path", required=False)
parser.add_argument("--gamg_threshold", default=0.01, type=float,
                    help="Gamg threshold")
parser.add_argument("--gamg_near_null_rot", action='store_true',
                    help="Use rotational gamg near nullspace")
args = parser.parse_args()

name = f"weerdesteijn-3d-iv-burgers{args.burgers}-{args.optional_name}"

# +
# Set up geometry:
L = 1500e3  # Length of the domain in m
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
D = radius_values[0]-radius_values[-1]
L_tilde = L / D
radius_values_tilde = np.array(radius_values)/D

if args.structured_dz:
    layer_heights = []
    dz = 1./args.nz
    for i in range(args.nz):
        layer_heights.append(dz)
    nz = args.nz

else:
    layer_heights = extruded_layer_heights(args.DG0_layers, radius_values_tilde)
    nz = f"{args.DG0_layers}perlayer"

if args.refined_surface:
    if args.const_aspect:
        surface_mesh = Mesh(f"./weerdesteijn_box_refined_surface_{round(args.dx)}km_nondim_constaspect.msh", name="surface_mesh")
    else:
        surface_mesh = Mesh(f"./weerdesteijn_box_refined_surface_{round(args.dx)}km_nondim.msh", name="surface_mesh")
else:
    nx = round(L / (args.dx*1e3))
    surface_mesh = SquareMesh(nx, nx, L_tilde)

mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_heights),
    layer_height=layer_heights,
)

vertical_component = 2

mesh.cartesian = True
boundary = get_boundary_ids(mesh)

ds = CombinedSurfaceMeasure(mesh, degree=6)

log("Volume of box: ", assemble(Constant(1) * dx(domain=mesh)))
log("Area of top: ", assemble(Constant(1) * ds(boundary.top, domain=mesh)))
log("Area of bottom: ", assemble(Constant(1) * ds(boundary.bottom, domain=mesh)))
log("Area of left side: ", assemble(Constant(1) * ds(boundary.left, domain=mesh)))
log("Area of right side: ", assemble(Constant(1) * ds(boundary.right, domain=mesh)))
log("Area of front side: ", assemble(Constant(1) * ds(boundary.front, domain=mesh)))
log("Area of back side: ", assemble(Constant(1) * ds(boundary.back, domain=mesh)))

# -

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
S = TensorFunctionSpace(mesh, "DQ", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG0 = FunctionSpace(mesh, "DG", 0)  # DG0 for 1d radial profiles
DG1 = FunctionSpace(mesh, "DG", 1)  # DG0 for 1d radial profiles
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)


Z = MixedFunctionSpace([V, S])  # Mixed function space.


# +
u = Function(V, name='displacement')  # A field over the mixed function space Z.
m = Function(S, name='internal variable 1')  # A field over the mixed function space Z.
m_list = [m]

if args.burgers:
    m2 = Function(S, name='internal variable 2')  # A field over the internal variable space.
    m_list.append(m2)

stress = Function(S, name='deviatoric stress')  # A field over the mixed function space Z.
power_factor = Function(DG1, name='viscosity factor')  # A field over the mixed function space Z.
dev_stress_2 = Function(DG1, name='2nd stress invariant')  # A field over the mixed function space Z.


# Output function space information:
log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable  DOF:", S.dim())
log("Number of Velocity and internal variable DOF:", V.dim()+S.dim())

X = SpatialCoordinate(mesh)

density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e40, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21

density_values_tilde = np.array(density_values)/density_scale

if args.burgers:
    shear_modulus_values_1_tilde = 0.5*np.array(shear_modulus_values)/shear_modulus_scale
    shear_modulus_values_2_tilde = 0.5*np.array(shear_modulus_values)/shear_modulus_scale
    viscosity_values_1_tilde = 0.5*np.array(viscosity_values)/viscosity_scale
    viscosity_values_2_tilde = args.viscosity_ratio*0.5*np.array(viscosity_values)/viscosity_scale
else:
    shear_modulus_values_tilde = np.array(shear_modulus_values)/shear_modulus_scale
    viscosity_values_tilde = np.array(viscosity_values)/viscosity_scale

density = Function(DG0, name="density")
initialise_background_field(
    density, density_values_tilde, X, radius_values_tilde,
    shift=radius_values_tilde[-1])

if args.burgers:
    shear_modulus_1 = Function(DG0, name="shear modulus 1")
    initialise_background_field(
        shear_modulus_1, shear_modulus_values_1_tilde, X, radius_values_tilde,
        shift=radius_values_tilde[-1])

    shear_modulus_2 = Function(DG0, name="shear modulus 2")
    initialise_background_field(
        shear_modulus_2, shear_modulus_values_2_tilde, X, radius_values_tilde,
        shift=radius_values_tilde[-1])
    shear_mod_list = [shear_modulus_1, shear_modulus_2]
else:
    shear_modulus = Function(DG0, name="shear modulus")
    initialise_background_field(
        shear_modulus, shear_modulus_values_tilde, X, radius_values_tilde,
        shift=radius_values_tilde[-1])
    shear_mod_list = [shear_modulus]

# if Pseudo incompressible set bulk modulus to a constant...
# Otherwise use same jumps from shear modulus multiplied by a factor

if args.bulk_shear_ratio > 10:
    bulk_modulus = Constant(1)
    compressible_buoyancy = False
    compressible_adv_hyd_pre = False
else:
    bulk_modulus = Function(DG0, name="bulk modulus")
    if args.burgers:
        initialise_background_field(
            bulk_modulus, 2*shear_modulus_values_1_tilde, X, radius_values_tilde,
            shift=radius_values_tilde[-1])
    else:
        initialise_background_field(
            bulk_modulus, shear_modulus_values_tilde, X, radius_values_tilde,
            shift=radius_values_tilde[-1])
    compressible_buoyancy = True
    compressible_adv_hyd_pre = True

if args.burgers:
    viscosity_1 = Function(DG0, name="viscosity 1")
    initialise_background_field(viscosity_2, viscosity_values_2_tilde)
    viscosity_2 = Function(DG0, name="viscosity 2")
    initialise_background_field(
        viscosity_2, viscosity_values_2_tilde, X, radius_values_tilde,
        shift=radius_values_tilde[-1])
else:
    viscosity = Function(DG0, name="viscosity")
    initialise_background_field(
        viscosity, viscosity_values_tilde, X, radius_values_tilde,
        shift=radius_values_tilde[-1])

# Next let's define the length of our time step. If we want to accurately resolve the
# elastic response we should choose a timestep lower than the Maxwell time,
# $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation
# to 'catch up' with the initial, instantaneous elastic deformation.
#
# Let's print out the Maxwell time for each layer

year_in_seconds = 8.64e4 * 365.25
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale
for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(layer_visc/layer_mu/year_in_seconds):.0f} years")
    log(f"Ratio to characteristic maxwell time: {float(layer_visc/layer_mu/characteristic_maxwell_time)}")


# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds / characteristic_maxwell_time)

dt_years = args.dt_years
dt = Constant(dt_years * year_in_seconds/characteristic_maxwell_time)
Tend_years = args.Tend
Tend = Constant(Tend_years * year_in_seconds/characteristic_maxwell_time)
dt_out_years = args.dt_out_years
dt_out = Constant(dt_out_years * year_in_seconds/characteristic_maxwell_time)

max_timesteps = round((Tend - Tstart * year_in_seconds/characteristic_maxwell_time) / dt)
log("max timesteps: ", max_timesteps)

output_frequency = round(dt_out / dt)
log("output_frequency:", output_frequency)
log(f"dt: {float(dt)} maxwell times")
log(f"dt: {float(dt * characteristic_maxwell_time / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} maxwell times")
log(f"Simulation end time: {Tend} maxwell times")
log(f"Simulation end time: {float(Tend * characteristic_maxwell_time / year_in_seconds)} years")
# -

# Initialise ice loading
rho_ice = 931 / density_scale
g = 9.815
B_mu = Constant(density_scale * D * g / shear_modulus_scale)
log("Ratio of buoyancy/shear = rho g D / mu = ", float(B_mu))

if args.short_simulation:
    Hice = 100 / D
    t1_load = 100 * year_in_seconds / characteristic_maxwell_time
    ramp = conditional(time < t1_load, time / t1_load, 1)
else:
    Hice = 1000 / D
    t1_load = 90e3 * year_in_seconds / characteristic_maxwell_time
    t2_load = 100e3 * year_in_seconds / characteristic_maxwell_time
    ramp_after_t1 = conditional(
        time < t2_load, 1 - (time - t1_load) / (t2_load - t1_load), 0
    )
    ramp = conditional(time < t1_load, time / t1_load, ramp_after_t1)

# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3 / D
disc_dx = 1e3 / D
k_disc = 1/disc_dx  # wavenumber for disk 2pi / lambda
r = pow(pow(X[0], 2) + pow(X[1], 2), 0.5)
disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
ice_load = ramp * B_mu * rho_ice * Hice * disc

if args.lateral_viscosity:
    upper_depth = -70e3 / D
    lower_depth = -170e3 / D
    cylinder_thickness = conditional(
        X[2] < upper_depth, conditional(X[2] > lower_depth, 1, 0),
        0
    )
    low_visc = 1e19 / viscosity_scale
    cylinder_mask = Function(DG0).interpolate(cylinder_thickness * disc)
    if args.burgers:
        low_visc *= 0.5
        viscosity_1.interpolate(cylinder_mask * low_visc + (1-cylinder_mask) * viscosity_1)
        viscosity_2.interpolate(cylinder_mask * low_visc + (1-cylinder_mask) * viscosity_2)
    else:
        viscosity.interpolate(cylinder_mask * low_visc + (1-cylinder_mask) * viscosity)

if args.burgers:
    visc_list = [viscosity_1, viscosity_2]
else:
    visc_list = [viscosity]
# +
# Setup boundary conditions
stokes_bcs = {
    boundary.bottom: {'uz': 0},
    boundary.top: {'normal_stress': ice_load, 'free_surface': {}},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
    boundary.front: {'uy': 0},
    boundary.back: {'uy': 0},
}


# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.

approximation = CompressibleInternalVariableApproximation(
    bulk_modulus=bulk_modulus, density=density, shear_modulus=shear_mod_list,
    viscosity=visc_list, B_mu=B_mu, bulk_shear_ratio=args.bulk_shear_ratio,
    compressible_buoyancy=compressible_buoyancy,
    compressible_adv_hyd_pre=compressible_adv_hyd_pre)

iterative_parameters = {"mat_type": "matfree",
                        "snes_monitor": None,
                        "snes_converged_reason": None,
                        "snes_type": "ksponly",
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-5,
                        "ksp_converged_reason": None,
                        "ksp_monitor": None,
                        "pc_type": "python",
                        "pc_fieldsplit_type": "firedrake.AssembledPC",
                        "pc_python_type": "firedrake.AssembledPC",
                        "assembled_pc_type": "gamg",
                        "assembled_mg_levels_pc_type": "sor",
                        "assembled_pc_gamg_threshold": args.gamg_threshold,
                        "assembled_pc_gamg_square_graph": 100,
                        "assembled_pc_gamg_coarse_eq_limit": 1000,
                        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
                        }

Z_nullspace = None  # Default: don't add nullspace for now
Z_near_nullspace = rigid_body_modes(V, rotational=args.gamg_near_null_rot,
                                    translations=[0, 1, 2])

coupled_solver = InternalVariableSolver(u, approximation, dt=dt, m_list=m_list, bcs=stokes_bcs,
                                        solver_parameters=iterative_parameters,
                                        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                        near_nullspace=Z_near_nullspace)


coupled_stage = PETSc.Log.Stage("coupled_solve")

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create output file
OUTPUT = args.write_output
if OUTPUT:
    output_file = VTKFile(args.output_path+f"output_{name}.pvd")
    output_file.write(u, m)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max uk_min"
)
gd = GeodynamicalDiagnostics(u, density, boundary.bottom, boundary.top)

checkpoint_filename = f"{args.output_path}{name}-refinedsurface{args.refined_surface}-dx{args.dx}km-nz{nz}-dt{dt_years}years-bulktoshear{args.bulk_shear_ratio}-compbuoy{compressible_buoyancy}-nondim-chk.h5"

displacement_filename = f"{args.output_path}displacement-{name}-refinedsurface{args.refined_surface}-dx{args.dx}km-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-compbuoy{compressible_buoyancy}-nondim.dat"

# Initial displacement at time zero is zero
displacement_min_array = [[0.0, 0.0]]

vertical_displacement = Function(V.sub(2), name="vertical displacement")  # Function to store vertical displacement for output
# -

# Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter.
# At each step we call `solve` to calculate the incremental displacement and pressure fields. This
# will update the displacement at the surface and stress values accounting for the time dependent
# Maxwell consitutive equation.

for timestep in range(1, max_timesteps+1):
    # update time first so that ice load begins
    time.assign(time+dt)
    # Solve Stokes sytem:
    with coupled_stage: coupled_solver.solve()

    # Log diagnostics:
    plog.log_str(f"{timestep} {time.dat.data[0]} {float(dt)} {gd.u_rms()} "
                 f"{gd.u_rms_top()} {gd.ux_max(boundary.top)} "
                 f"{gd.uk_min(boundary.top)}")
    # Compute diagnostics:
    # output dimensional vertical displacement
    vertical_displacement.interpolate(vc(u)*D)
    bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, boundary.top)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    # Minimum displacement at surface (should be top left corner with
    # greatest (-ve) deflection due to ice loading
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)
    log("Greatest (-ve) displacement", displacement_min)
    log("check Greatest (-ve) gd.log", D*gd.uk_min(boundary.top))
    displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
    displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)
    log("Greatest (+ve) displacement", displacement_max)
    displacement_min_array.append([float(characteristic_maxwell_time*time.dat.data[0]/year_in_seconds), displacement_min])

    disp_norm_L2surf = assemble((u[vertical_component])**2 * ds(boundary.top))
    log("L2 surface norm displacement", disp_norm_L2surf)

    disp_norm_L1surf = assemble(abs(u[vertical_component]) * ds(boundary.top))
    log("L1 surface norm displacement", disp_norm_L1surf)

    integrated_disp = assemble(u[vertical_component] * ds(boundary.top))
    log("Integrated displacement", integrated_disp)

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        if OUTPUT:
            output_file.write(u, m)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u, name="Stokes")
            checkpoint.save_function(m, name="Internal variable")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

plog.close()
