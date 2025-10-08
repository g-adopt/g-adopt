# Test case based on simplified spada 2011 benchmark
# without gravity. A 10 deg ice load is placed on the
# North Pole instantaneously at the start of the simulation
# and the mantle evolves towards isostatic equilibrium.

from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
from gadopt.utility import vertical_component as vc
import argparse
import numpy as np
from mpi4py import MPI
parser = argparse.ArgumentParser()
parser.add_argument("--reflevel", default=5, type=float, help="Horizontal refinement level of surface cubed sphere mesh", required=False)
parser.add_argument("--DG0_layers", default=5, type=int, help="Number of cells per layer for DG0 discretisation of background profiles", required=False)
parser.add_argument("--dt_years", default=1e3, type=float, help="Timestep in years", required=False)
parser.add_argument("--Tend", default=10e3, type=float, help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=1.94, type=float, help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--load_checkpoint", action='store_true', help="Load simulation data from a checkpoint file")
parser.add_argument("--checkpoint_file", default=None, type=str, help="Checkpoint file name", required=False)
parser.add_argument("--Tstart", default=0, type=float, help="Simulation start time in years", required=False)
parser.add_argument("--geometric_dt_steps", default=0, type=int, help="No. of steps used for a geometric progression for increasing dt")
parser.add_argument("--split_dt_steps", default=0, type=int, help="No. of steps used for a split timestep approach nsteps before and after characteristic maxwell time")
parser.add_argument("--write_output", action='store_true', help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str, help="Optional string to add to simulation name for outputs", required=False)
parser.add_argument("--output_path", default="./", type=str, help="Optional output path", required=False)
args = parser.parse_args()

name = f"spada-3d-{args.optional_name}"
if args.geometric_dt_steps:
    name = f"{name}-geomdt{args.geometric_dt_steps}"
elif args.split_dt_steps:
    name = f"{name}-splitdt{args.split_dt_steps}"
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


mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_height_list),
    layer_height=layer_height_list,
    extrusion_type='radial'
)

mesh.cartesian = False
boundary = get_boundary_ids(mesh)
nz = f"{DG0_layers}perlayer"

ds = CombinedSurfaceMeasure(mesh, degree=6)

log("Volume of sphere: ", assemble(Constant(1) * dx(domain=mesh)))
log("Area of top: ", assemble(Constant(1) * ds(boundary.top, domain=mesh)))
log("Area of bottom: ", assemble(Constant(1) * ds(boundary.bottom, domain=mesh)))

# Set up function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
S = TensorFunctionSpace(mesh, "DQ", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG0 = FunctionSpace(mesh, "DG", 0)  # (Discontinuous) function space (tensor)
DG1 = FunctionSpace(mesh, "DG", 1)  # (Discontinuous)  function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)
P1 = FunctionSpace(mesh, "CG", 1)


# Function to store the solutions:
u = Function(V, name='displacement')  # A field over the displacement space.
m = Function(S, name='internal variable 1')  # A field over the internal variable space.
stress = Function(S, name='deviatoric stress')  # A field over the mixed function space Z.

m_list = [m]
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable  DOF:", S.dim())
log("Number of Velocity and internal variable DOF:", V.dim()+S.dim())

# Let's start initialising some parameters. First of all Firedrake has a helpful function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties.
# In this case the density, shear modulus and viscosity only vary in the vertical direction.
# The layer properties specified are from spada et al. (2011).
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e40, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21

density_values_tilde = np.array(density_values)/density_scale
shear_modulus_values_tilde = np.array(shear_modulus_values)/shear_modulus_scale
viscosity_values_tilde = np.array(viscosity_values)/viscosity_scale


def initialise_background_field(field, background_values):
    for i in range(0, len(background_values)):
        field.interpolate(conditional(vc(X) >= radius_values_tilde[i+1],
                          conditional(vc(X) <= radius_values_tilde[i],
                          background_values[i], field), field))


density = Function(DG0, name="density")
initialise_background_field(density, density_values_tilde)

shear_modulus = Function(DG0, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values_tilde)


# if Pseudo incompressible set bulk modulus to a constant...
# Otherwise use same jumps from shear modulus multiplied by a factor

if args.bulk_shear_ratio > 10:
    bulk_modulus = Constant(1)
    compressible_buoyancy = False
    compressible_adv_hyd_pre = False
else:
    bulk_modulus = Function(DG0, name="bulk modulus")
    initialise_background_field(bulk_modulus, shear_modulus_values_tilde)
    compressible_buoyancy = True
    compressible_adv_hyd_pre = True

viscosity = Function(DG0, name="viscosity")
initialise_background_field(viscosity, viscosity_values_tilde)


# -

# Next let's define the length of our time step. If we want to accurately resolve the elastic response we should choose a
# timestep lower than the Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation
# to 'catch up' with the initial, instantaneous elastic deformation.
#
# Let's print out the Maxwell time for each layer

year_in_seconds = 8.64e4 * 365.25
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale
for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(layer_visc/layer_mu/year_in_seconds):.0f} years")
    log(f"Ratio to characteristic maxwell time: {float(layer_visc/layer_mu/characteristic_maxwell_time)}")

# As we can see the shortest Maxwell time is given by the lower mantle and is about 280 years, i.e. it will take about 280
# years for the viscous deformation in that layer to catch up any instantaneous elastic deformation. Conversely the top layer,
# our lithosphere, has a Maxwell time of 6 million years. Given that our simulations only run for 110000 years the viscous
# deformation over the course of the simulation will always be negligible compared with the elastic deformation. For now let's
# choose a timestep of 250 years and an output frequency of 2000 years.

# +
# Timestepping parameters
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

# Next let's setup our ice load. Following the long test from Weeredesteijn et al 2023,
# during the first 90 thousand years of the simulation the ice sheet will grow to a thickness of 1 km.
# The ice thickness will rapidly shrink to ice free conditions in the next 10 thousand years. Finally,
# the simulation will run for a further 10 thousand years to allow the system to relax towards
# isostatic equilibrium. This is approximately the length of an interglacial-glacial cycle. The
# width of the ice sheet is 100 km and we have used a tanh function again to smooth out the
# transition from ice to ice-free regions.
#
# As the loading and unloading cycle only varies linearly in time, let's write the ice load as a symbolic expression.

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

ice_load = B_mu * rho_ice * Hice * disc

OUTPUT = args.write_output


if OUTPUT:
    discfunc = Function(P1).interpolate(disc)
    discfile = VTKFile(f"{args.output_path}discfile.pvd").write(discfunc)
    viscfile = VTKFile(f"{args.output_path}viscfile.pvd").write(viscosity)


# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and
# side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing
# the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as
# indicating this is a free surface.
#
# The `delta_rho_fs` option accounts for the density contrast across the free surface whether there
# is ice or air above a particular region of the mantle.

# +
# Setup boundary conditions
stokes_bcs = {
    boundary.bottom: {'un': 0},
    boundary.top: {'free_surface': {'normal_stress': ice_load}},
}

# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.

approximation = CompressibleInternalVariableApproximation(bulk_modulus=bulk_modulus, density=density, shear_modulus=[shear_modulus], viscosity=[viscosity], B_mu=B_mu, bulk_shear_ratio=args.bulk_shear_ratio, compressible_buoyancy=compressible_buoyancy, compressible_adv_hyd_pre=compressible_adv_hyd_pre)
# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields and various fields
# needed for the solve along with the approximation, timestep and boundary conditions.
#

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

V_nullspace = rigid_body_modes(V, rotational=True)
V_near_nullspace = rigid_body_modes(V, rotational=True, translations=[0, 1, 2])

coupled_solver = InternalVariableSolver(u, approximation, dt=dt, m_list=m_list, bcs=stokes_bcs,
                                        solver_parameters=iterative_parameters,
                                        nullspace=V_nullspace, transpose_nullspace=V_nullspace,
                                        near_nullspace=V_near_nullspace)


# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create output file
vertical_displacement = Function(V.sub(2), name="radial displacement")  # Function to store vertical displacement for output
velocity = Function(V, name="velocity")  # Function to store velocity for output
old_disp = Function(V, name="old disp").interpolate(u)  # Function to store velocity for output

if OUTPUT:
    output_file = VTKFile(f"{args.output_path}{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.pvd")
    output_file.write(u, *m_list, vertical_displacement, velocity)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max uk_min"
)
gd = GeodynamicalDiagnostics(u, density, boundary.bottom, boundary.top)


checkpoint_filename = f"{args.output_path}{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulktoshear{args.bulk_shear_ratio}-nondim-chk.h5"

displacement_filename = f"{args.output_path}displacement-{name}-reflevel{args.reflevel}-nz{nz}-dt{dt_years}years-bulk{args.bulk_shear_ratio}-nondim.dat"

# Initial displacement at time zero is zero
displacement_min_array = [[0.0, 0.0]]

# -

# Now let's run the simulation!
# At each step we call `solve` to calculate the incremental displacement and pressure fields. This
# will update the displacement at the surface and stress values accounting for the time dependent
# Maxwell consitutive equation.

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
                 f"{gd.uk_min(boundary.top)}")
    # Compute diagnostics:

    velocity.interpolate((u-old_disp)/dt)
    old_disp.interpolate(u)

    # output dimensional vertical displacement
    vertical_displacement.interpolate(vc(u)*D)
    bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, boundary.top)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (-ve) displacement", displacement_min)
    log(f"check gd log: {gd.uk_min(boundary.top)*D}")
    displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
    displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (+ve) displacement", displacement_max)
    displacement_min_array.append([float(characteristic_maxwell_time*time.dat.data[0]/year_in_seconds), displacement_min])

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        if OUTPUT:
            output_file.write(u, *m_list, vertical_displacement, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u, name="Stokes")
            checkpoint.save_function(m, name="Internal variable")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

plog.close()
