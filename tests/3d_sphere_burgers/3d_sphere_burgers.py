# Test case based on Spada et al. 2011 benchmark without
# gravity but including lateral heterogenities and
# compressible burgers rheology. Ice load is applied
# instanteously at the start of the simulation and the
# mantle evolves towards isostatic equilbrium
# For results within G-ADOPT see Scott et al. 2025.

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
import argparse
import numpy as np
import scipy
import math
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--reflevel", default=5, type=float,
                    help="Horizontal refinement level of surface cubed sphere mesh",
                    required=False)
parser.add_argument("--DG0_layers", default=5, type=int,
                    help="Number of cells per layer for DG0 discretisation of background profiles",
                    required=False)
parser.add_argument("--dt_years", default=1e3, type=float,
                    help="Timestep in years", required=False)
parser.add_argument("--Tend", default=10e3, type=float,
                    help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=1.94, type=float,
                    help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--viscosity_ratio", default=1, type=float,
                    help="Ratio of viscosity 2 / viscosity 1", required=False)
parser.add_argument("--lateral_visc", action='store_true',
                    help="Use a lateral viscosity field")
parser.add_argument("--load_checkpoint", action='store_true',
                    help="Load simulation data from a checkpoint file")
parser.add_argument("--checkpoint_file", default=None, type=str,
                    help="Checkpoint file name", required=False)
parser.add_argument("--Tstart", default=0, type=float,
                    help="Simulation start time in years", required=False)
parser.add_argument("--geometric_dt_steps", default=0, type=int,
                    help="No. of steps used for a geometric progression for increasing dt")
parser.add_argument("--split_dt_steps", default=0, type=int,
                    help="No. of steps used for a split timestep approach nsteps before and after characteristic maxwell time")
parser.add_argument("--write_output", action='store_true',
                    help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str,
                    help="Optional string to add to simulation name for outputs",
                    required=False)
parser.add_argument("--output_path",
                    default="./",
                    type=str, help="Optional output path", required=False)
args = parser.parse_args()

name = f"sphere-burgers-3d-internalvariable-{args.optional_name}"
if args.geometric_dt_steps:
    name = f"{name}-geomdt{args.geometric_dt_steps}"
elif args.split_dt_steps:
    name = f"{name}-splitdt{args.split_dt_steps}"
# +
# Set up geometry:
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
D = radius_values[0]-radius_values[-1]
radius_values_tilde = np.array(radius_values)/D

layer_height_list = []
DG0_layers = args.DG0_layers
nz_layers = [DG0_layers, DG0_layers, DG0_layers, DG0_layers]

for j in range(len(radius_values_tilde)-1):
    i = len(radius_values_tilde)-2 - j  # want to start at the bottom
    r = radius_values_tilde[i]
    h = r - radius_values_tilde[i+1]
    nz = nz_layers[i]
    dz = h / nz

    for i in range(nz):
        layer_height_list.append(dz)

# Construct a circle mesh and then extrude into a cylinder:
ncells = 8*2**(args.reflevel-1)
rmin = radius_values_tilde[-1]
surface_mesh = CubedSphereMesh(rmin, refinement_level=args.reflevel, degree=2, name='surface_mesh')

layer_heights = extruded_layer_heights(args.DG0_layers, radius_values_tilde)

mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_heights),
    layer_height=layer_heights,
    extrusion_type='radial'
)

mesh.cartesian = False
boundary = get_boundary_ids(mesh)
nz = f"{args.DG0_layers}perlayer"

ds = CombinedSurfaceMeasure(mesh, degree=6)

log("Volume of sphere: ", assemble(Constant(1) * dx(domain=mesh)))
log("Area of top: ", assemble(Constant(1) * ds(boundary.top, domain=mesh)))
log("Area of bottom: ", assemble(Constant(1) * ds(boundary.bottom, domain=mesh)))

# -

# Set up function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
S = TensorFunctionSpace(mesh, "DQ", 1)  # (Discontinuous) Stress tensor function space
DG0 = FunctionSpace(mesh, "DG", 0)  # (Discontinuous) function space
DG1 = FunctionSpace(mesh, "DG", 1)  # (Discontinuous) viscosity function space
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)
P1 = FunctionSpace(mesh, "CG", 1)


# +
# Function to store the solutions:
u = Function(V, name='displacement')  # A field over the displacement space.
m1 = Function(S, name='internal variable 1')  # A field over the internal variable space.
m2 = Function(S, name='internal variable 2')  # A field over the internal variable space.
stress = Function(S, name='deviatoric stress')  # A field over the mixed function space Z.

m_list = [m1, m2]
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable  DOF:", S.dim())

X = SpatialCoordinate(mesh)

# +
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e40, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale

density_values_tilde = np.array(density_values)/density_scale
shear_modulus_values_1_tilde = 0.5*np.array(shear_modulus_values)/shear_modulus_scale
shear_modulus_values_2_tilde = 0.5*np.array(shear_modulus_values)/shear_modulus_scale
viscosity_values_1_tilde_log = np.log10(0.5*np.array(viscosity_values)/viscosity_scale)
viscosity_values_2_tilde_log = np.log10(args.viscosity_ratio*0.5*np.array(viscosity_values)/viscosity_scale)


density = Function(DG0, name="density")
initialise_background_field(
    density, density_values_tilde, X, radius_values_tilde)

shear_modulus_1 = Function(DG0, name="shear modulus 1")
initialise_background_field(
    shear_modulus_1, shear_modulus_values_1_tilde, X, radius_values_tilde)

shear_modulus_2 = Function(DG0, name="shear modulus 2")
initialise_background_field(
    shear_modulus_2, shear_modulus_values_2_tilde, X, radius_values_tilde)

# if Pseudo incompressible set bulk modulus to a constant...
# Otherwise use same jumps from shear modulus multiplied by a factor

if args.bulk_shear_ratio > 10:
    bulk_modulus = Constant(1)
    approx = QuasiCompressibleInternalVariableApproximation
else:
    bulk_modulus = Function(DG0, name="bulk modulus")
    initialise_background_field(
        bulk_modulus, 2*shear_modulus_values_1_tilde, X, radius_values_tilde)
    approx = CompressibleInternalVariableApproximation

viscosity_1 = Function(DG1, name="viscosity")
initialise_background_field(
    viscosity_1, viscosity_values_1_tilde_log, X, radius_values_tilde)

viscosity_2 = Function(DG1, name="viscosity 2")
initialise_background_field(
    viscosity_2, viscosity_values_2_tilde_log, X, radius_values_tilde)

# -


# +
# Timestepping parameters
year_in_seconds = 8.64e4 * 365.25
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds / characteristic_maxwell_time)


if args.geometric_dt_steps:
    dt_years = start * characteristic_maxwell_time / year_in_seconds  # initial dt for writing filenames...
    Tend_years = args.Tend
    Tend = Tend_years * year_in_seconds/characteristic_maxwell_time
    t_list = np.geomspace(start, Tend, num=args.geometric_dt_steps, endpoint=True, dtype=None, axis=0)
    dt_list = np.diff(t_list)
    dt_list = np.insert(dt_list, 0, t_list[0])
    dt = Constant(dt_list[0])
    print(dt_list)
    max_timesteps = len(t_list)
    output_frequency = 1

elif args.split_dt_steps:
    elastic_dt = 1 / args.split_dt_steps
    Tend_years = args.Tend
    Tend = Tend_years * year_in_seconds/characteristic_maxwell_time
    viscous_dt = (Tend - 1) / args.split_dt_steps
    dt = Constant(elastic_dt)
    max_timesteps = round(2*args.split_dt_steps)
    output_frequency = 1
    dt_years = 'split'

else:
    dt_years = args.dt_years
    dt = Constant(dt_years * year_in_seconds/characteristic_maxwell_time)
    Tend_years = args.Tend
    Tend = Constant(Tend_years * year_in_seconds/characteristic_maxwell_time)
    dt_out_years = 1e3
    dt_out = Constant(dt_out_years * year_in_seconds/characteristic_maxwell_time)

    max_timesteps = round((Tend - Tstart * year_in_seconds/characteristic_maxwell_time) / dt)
    output_frequency = round(dt_out / dt)
log("max timesteps: ", max_timesteps)

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
Hice = 1000 / D

# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
surface_dx = 200*1e3
ncells = 2*pi*radius_values[0] / surface_dx
surface_resolution_radians = 2*pi / ncells
distance_from_rotation_axis = sqrt(X[0]**2 + X[1]**2)
colatitude = atan2(distance_from_rotation_axis, X[2])
disc1_centre = 0  # centre of disc1
disc = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))


OUTPUT = args.write_output

if args.lateral_visc:
    phi = atan2(X[1], X[0])  # (longitude )
    colatitude = atan2(sqrt(X[0]**2+X[1]**2), X[2])

    l, m, eps_c, eps_s = 4, 1, 1, 1
    Plm = Function(P1, name="P_lm")
    cos_colatitude = Function(P1).interpolate(cos(colatitude))
    Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_colatitude.dat.data_ro)  # Evaluate P_lm node-wise using scipy lpmv
    Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
    if m == 0:
        Plm.assign(Plm/math.sqrt(2))

    # This should be an order of mangitude change in viscosity
    viscosity_1.interpolate(10**(viscosity_1 + (eps_c*cos(m*phi) + eps_s*sin(m*phi)) * Plm))
    viscosity_2.interpolate(10**(viscosity_2 + (eps_c*cos(m*phi) + eps_s*sin(m*phi)) * Plm))
else:
    viscosity_1.interpolate(10**viscosity_1)
    viscosity_2.interpolate(10**viscosity_2)


if OUTPUT:
    discfunc = Function(P1).interpolate(disc)
    discfile = VTKFile(f"{args.output_path}discfile.pvd").write(discfunc)
    viscfile = VTKFile(f"{args.output_path}viscfile.pvd").write(viscosity_1, viscosity_2)

ice_load = B_mu * rho_ice * Hice * disc

# +
# Setup boundary conditions
stokes_bcs = {
    boundary.bottom: {'un': 0},
    boundary.top: {'normal_stress': ice_load, 'free_surface': {}},
}

# gd = GeodynamicalDiagnostics(z, density, boundary.bottom, boundary.top)
# -


# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.

approximation = approx(
    bulk_modulus=bulk_modulus, density=density,
    shear_modulus=[shear_modulus_1, shear_modulus_2],
    viscosity=[viscosity_1, viscosity_2], B_mu=B_mu,
    bulk_shear_ratio=args.bulk_shear_ratio)


iterative_parameters = {"mat_type": "matfree",
                        "snes_monitor": None,
                        "snes_converged_reason": None,
                        "snes_type": "ksponly",
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-5,
                        "ksp_converged_reason": None,
                        "ksp_monitor": None,
                        "pc_type": "python",
                        "pc_python_type": "firedrake.AssembledPC",
                        "assembled_pc_type": "gamg",
                        "assembled_mg_levels_pc_type": "sor",
                        "assembled_pc_gamg_threshold": 0.01,
                        "assembled_pc_gamg_square_graph": 100,
                        "assembled_pc_gamg_coarse_eq_limit": 1000,
                        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
                        }

Z_nullspace = rigid_body_modes(V, rotational=True)
Z_near_nullspace = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])

coupled_solver = InternalVariableSolver(
    u,
    approximation,
    dt=dt,
    internal_variables=m_list,
    bcs=stokes_bcs,
    solver_parameters=iterative_parameters,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
)

# +
# Create output file
vertical_displacement = Function(V.sub(2), name="radial displacement")
velocity = Function(V, name="velocity")  # Function to store velocity for output
old_disp = Function(V, name="old disp").interpolate(u)

if OUTPUT:
    output_file = VTKFile(f"{args.output_path}{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.pvd")
    output_file.write(u, *m_list, vertical_displacement, velocity, viscosity_1, viscosity_2)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max uv_min"
)
gd = GIADiagnostics(u, boundary.bottom, boundary.top)

checkpoint_filename = f"{args.output_path}{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulktoshear{args.bulk_shear_ratio}-nondim-chk.h5"

displacement_filename = f"{args.output_path}displacement-{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.dat"

# Initial displacement at time zero is zero
displacement_min_array = [[0.0, 0.0]]

# -

for timestep in range(1, max_timesteps+1):
    # update time first so that ice load begins
    if args.geometric_dt_steps:
        dt.assign(dt_list[timestep-1])
    elif args.split_dt_steps:
        dt.assign(conditional(time < 1, elastic_dt, viscous_dt))
    time.assign(time+dt)
    coupled_solver.solve()

    # Log diagnostics:
    plog.log_str(f"{timestep} {time.dat.data[0]} {float(dt)} {gd.u_rms()} "
                 f"{gd.u_rms_top()} {gd.ux_max(boundary.top)} "
                 f"{gd.uv_min(boundary.top)}")
    # Compute diagnostics:
    velocity.interpolate((u-old_disp)/dt)
    old_disp.interpolate(u)

    # output dimensional vertical displacement
    displacement_min = gd.uv_min(boundary.top)
    log("Greatest (-ve) displacement", displacement_min)
    log("Greatest (+ve) displacement", gd.uv_max(boundary.top))
    displacement_min_array.append([float(characteristic_maxwell_time*time.dat.data[0]/year_in_seconds),
                                   displacement_min])

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        if OUTPUT:
            output_file.write(u, *m_list, vertical_displacement, velocity, viscosity_1, viscosity_2)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u, name="Stokes")
            checkpoint.save_function(m1, name="Internal variable 1")
            checkpoint.save_function(m2, name="Internal variable 2")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

plog.close()
