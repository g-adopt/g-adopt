# Idealised 2-D viscoelastic loading problem in a square box
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in a square box.
# Here we will focus purely on viscoelastic deformation by a surface load, i.e. a synthetic
# ice sheet!
#

from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
from gadopt.utility import vertical_component as vc
import argparse
import numpy as np
from mpi4py import MPI
parser = argparse.ArgumentParser()
parser.add_argument("--dx", default=50, type=float, help="Horizontal resolution in km", required=False)
parser.add_argument("--refined_surface", action='store_true', help="Use refined surface mesh")
parser.add_argument("--const_aspect", action='store_true', help="Also scale farfield dx for parallel scaling test to keep same aspect ratio")
parser.add_argument("--structured_dz", action='store_true', help="Use constant vertical resolution")
parser.add_argument("--nz", default=100, type=int, help="Number of vertical layers for structured dz")
parser.add_argument("--DG0_layers", default=5, type=int, help="Number of cells per layer for DG0 discretisation of background profiles", required=False)
parser.add_argument("--dt_years", default=1e3, type=float, help="Timestep in years", required=False)
parser.add_argument("--dt_out_years", default=10e3, type=float, help="Output timestep in years", required=False)
parser.add_argument("--Tend", default=110e3, type=float, help="Simulation end time in years", required=False)
parser.add_argument("--bulk_shear_ratio", default=100, type=float, help="Ratio of Bulk modulus / Shear modulus", required=False)
parser.add_argument("--load_checkpoint", action='store_true', help="Load simulation data from a checkpoint file")
parser.add_argument("--checkpoint_file", default=None, type=str, help="Checkpoint file name", required=False)
parser.add_argument("--Tstart", default=0, type=float, help="Simulation start time in years", required=False)
parser.add_argument("--short_simulation", action='store_true', help="Run simulation with short ice history from Weerdesteijn et a. 2023 testcase")
parser.add_argument("--lateral_viscosity", action='store_true', help="Include low viscosity cylinder from Weerdesteijn et a. 2023 testcase")
parser.add_argument("--write_output", action='store_true', help="Write out Paraview VTK files")
parser.add_argument("--optional_name", default="", type=str, help="Optional string to add to simulation name for outputs", required=False)
parser.add_argument("--output_path", default="/g/data/xd2/ws9229/viscoelastic/3d_weerdesteijn_displacement/", type=str, help="Optional output path", required=False)
parser.add_argument("--gamg_threshold", default=0.01, type=float, help="Gamg threshold")
parser.add_argument("--gamg_near_null_rot", action='store_true', help="Use rotational gamg near nullspace")
args = parser.parse_args()

name = f"weerdesteijn-3d-internalvariable-{args.optional_name}"
# Next we need to create a mesh of the mantle region we want to simulate. The Weerdesteijn test case is a 3D box 1500 km wide horizontally and
# 2891 km deep. To speed up things for this first demo, we consider a 2D domain, i.e. taking a vertical cross section through the 3D box.
#
# For starters let's use one of the default meshes provided by Firedrake, `RectangleMesh`. We have chosen 40 quadrilateral elements in the $x$
# direction and 40 quadrilateral elements in the $y$ direction. It is worth emphasising that the setup has coarse grid resolution so that the
# demo is quick to run! For real simulations we can use fully unstructured meshes to accurately resolve important features in the model, for
# instance near coastlines or sharp discontinuities in mantle properties.  We can print out the grid resolution using `log`, a utility provided by
# G-ADOPT. (N.b. `log` is equivalent to python's `print` function, except that it simplifies outputs when running simulations in parallel.)
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points
# in the negative z-direction. This attribute is used by G-ADOPT specifically, not
# Firedrake. By contrast, a non-Cartesian geometry is assumed to have gravity
# pointing in the radially inward direction.
#
# Boundaries are automatically tagged by the built-in meshes supported by Firedrake. For the `RectangleMesh` being used here, tag 1
# corresponds to the plane $x=0$; 2 to the $x=L$ plane; 3 to the $y=0$ plane; and 4 to the $y=D$ plane. For convenience, we can
# rename these to `left_id`, `right_id`, `bottom_id` and `boundary.top`.

# +
# Set up geometry:
L = 1500e3  # Length of the domain in m
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
D = radius_values[0]-radius_values[-1]
L_tilde = L / D
radius_values_tilde = np.array(radius_values)/D
 
layer_height_list = []

if args.structured_dz:
    dz = 1./args.nz
    for i in range(args.nz):
        layer_height_list.append(dz)
    nz = args.nz

else:
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
    nz = f"{DG0_layers}perlayer"

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
    layers=len(layer_height_list),
    layer_height=layer_height_list,
)

vertical_component = 2
mesh.coordinates.dat.data[:, vertical_component] -= 1

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
# We now need to choose finite element function spaces. `V` , `W`, `S` and `R` are symbolic
# variables representing function spaces. They also contain the
# function space's computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis. We will choose Q2-Q1 for the mixed incremental displacement-pressure similar to our mantle convection demos.
# This is a Taylor-Hood element pair which has good properties for Stokes modelling. We also initialise a discontinuous tensor function
# space that wil store our previous values of the deviatoric stress, as the gradient of the continous incremental displacement field will
# be discontinuous.

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
S = TensorFunctionSpace(mesh, "DQ", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG0 = FunctionSpace(mesh, "DG", 0)  # DG0 for 1d radial profiles
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the incremental displacement and pressure spaces to form
# a function space for the mixed Stokes problem, `Z`.

Z = MixedFunctionSpace([V, S])  # Mixed function space.

# We also specify functions to hold our solutions: `z` in the mixed
# function space, noting that a symbolic representation of the two
# parts – incremental displacement and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of `z` to *Incremental Displacement* and *Pressure*.
#
# We also need to initialise two functions `displacement` and `stress_old` that are used when timestepping the constitutive equation.

# +
z = Function(Z)  # A field over the mixed function space Z.
# Function to store the solutions:
u, m = split(z)  # Returns symbolic UFL expression for u and m
# Next rename for output:
z.subfunctions[0].rename("Displacement")
z.subfunctions[1].rename("Internal variable")
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
# We will approximate the series of layers using a smooth tanh function with a width of 20 km.
# The layer properties specified are from spada et al. (2011).
# N.b. that we have modified the viscosity of the Lithosphere viscosity from
# Spada et al. (2011) because we are using coarse grid resolution.


# +
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
        field.interpolate(conditional(vc(X) >= radius_values_tilde[i+1] - radius_values_tilde[0],
                          conditional(vc(X) <= radius_values_tilde[i] - radius_values_tilde[0],
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
else:
    bulk_modulus = Function(DG0, name="bulk modulus")
    initialise_background_field(bulk_modulus, shear_modulus_values_tilde)
    compressible_buoyancy = True

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
time = Function(R).assign(Tstart * year_in_seconds/ characteristic_maxwell_time)

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
Vi = Constant(density_scale * D * g / shear_modulus_scale)
log("Ratio of buoyancy/shear = rho g D / mu = ", float(Vi))

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
ice_load = ramp * Vi * rho_ice * Hice * disc

if args.lateral_viscosity:
    upper_depth = -70e3 / D
    lower_depth = -170e3 / D
    cylinder_thickness = conditional(
        X[2] < upper_depth, conditional(X[2] > lower_depth, 1, 0),
        0
    )
    low_visc = 1e19 / viscosity_scale 
    cylinder_mask = Function(DG0).interpolate(cylinder_thickness * disc)
    viscosity.interpolate(cylinder_mask * low_visc + (1-cylinder_mask) * viscosity)


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
    boundary.bottom: {'uz': 0},
    boundary.top: {'normal_stress': ice_load, 'free_surface': {}},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
    boundary.front: {'uy': 0},
    boundary.back: {'uy': 0},
}

gd = GeodynamicalDiagnostics(z, density, boundary.bottom, boundary.top)
# -


# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.

approximation = CompressibleInternalVariableApproximation(bulk_modulus=bulk_modulus, density=density, shear_modulus=shear_modulus, viscosity=viscosity, Vi=Vi, bulk_shear_ratio=args.bulk_shear_ratio, compressible_buoyancy=compressible_buoyancy, compressible_adv_hyd_pre=False)

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields `z` and various fields
# needed for the solve along with the approximation, timestep and boundary conditions.
#

direct_stokes_solver_parameters = {
    "snes_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

iterative_parameters = {"mat_type": "matfree",
                        "snes_type": "ksponly",
                        "ksp_type": "gmres",
                        "ksp_rtol": 1e-4,
                        "ksp_converged_reason": None,
                        "ksp_monitor": None,
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "symmetric_multiplicative",

                        "fieldsplit_0_ksp_converged_reason": None,
                        "fieldsplit_0_ksp_monitor": None,
#                        "fieldsplit_0_ksp_type": "cg",
                        "fieldsplit_0_ksp_type": "gmres",
                        "fieldsplit_0_pc_type": "python",
#                        "fieldsplit_0_pc_python_type": "gadopt.SPDAssembledPC",
                        "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                        "fieldsplit_0_assembled_pc_type": "gamg",
                        "fieldsplit_0_assembled_mg_levels_pc_type": "sor",
                        "fieldsplit_0_ksp_rtol": 1e-5,
                        "fieldsplit_0_assembled_pc_gamg_threshold": args.gamg_threshold,
                        "fieldsplit_0_assembled_pc_gamg_square_graph": 100,
                        "fieldsplit_0_assembled_pc_gamg_coarse_eq_limit": 1000,
                        "fieldsplit_0_assembled_pc_gamg_mis_k_minimum_degree_ordering": True,

                        "fieldsplit_1_ksp_converged_reason": None,
                        "fieldsplit_1_ksp_monitor": None,
                        "fieldsplit_1_ksp_type": "cg",
                        "fieldsplit_1_pc_type": "python",
                        "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
                        "fieldsplit_1_assembled_pc_type": "sor",
                        "fieldsplit_1_ksp_rtol": 1e-5,
                        }
Z_nullspace = None  # Default: don't add nullspace for now
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=args.gamg_near_null_rot, translations=[0, 1, 2])

coupled_solver = InternalVariableSolver(z, approximation, coupled_dt=dt, bcs=stokes_bcs,
                                        solver_parameters=iterative_parameters,
                                        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                        near_nullspace=Z_near_nullspace)


coupled_stage = PETSc.Log.Stage("coupled_solve")

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create output file
OUTPUT = args.write_output
if OUTPUT:
    output_file = VTKFile(args.output_path+"output.pvd")
    output_file.write(*z.subfunctions)

plog = ParameterLog(args.output_path+"params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)

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
    # Compute diagnostics:
    # output dimensional vertical displacement
    vertical_displacement.interpolate(vc(z.subfunctions[0])*D)
    bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, boundary.top)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (-ve) displacement", displacement_min)
    displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
    displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (+ve) displacement", displacement_max)
    displacement_min_array.append([float(characteristic_maxwell_time*time/year_in_seconds), displacement_min])

    disp_norm_L2surf = assemble((z.subfunctions[0][vertical_component])**2 * ds(boundary.top))
    log("L2 surface norm displacement", disp_norm_L2surf)

    disp_norm_L1surf = assemble(abs(z.subfunctions[0][vertical_component]) * ds(boundary.top))
    log("L1 surface norm displacement", disp_norm_L1surf)

    integrated_disp = assemble(z.subfunctions[0][vertical_component] * ds(boundary.top))
    log("Integrated displacement", integrated_disp)

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        if OUTPUT:
            output_file.write(*z.subfunctions)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Stokes")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

        plog.log_str(f"{timestep} {float(time)} {float(dt)} "
                     f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} "
                     )

# Let's use the python package *PyVista* to plot the magnitude of the displacement field through time.
# We will use the calculated displacement to artifically scale the mesh. We have exaggerated the stretching
# by a factor of 1500, **BUT...** it is important to remember this is just for ease of visualisation -
# the mesh is not moving in reality!

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import pyvista as pv
#
# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
#
# # Open a gif
# plotter.open_gif("displacement_warp.gif")
#
# # Make a colour map
# boring_cmap = plt.get_cmap("viridis", 25)
#
# for i in range(len(reader.time_values)):
#     reader.set_active_time_point(i)
#     data = reader.read()[0]
#
#     # Artificially warp the output data in the vertical direction by the free surface height
#     # Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=1500)
#     arrows = data.glyph(orient="Incremental Displacement", scale="Incremental Displacement", factor=400000, tolerance=0.05)
#     plotter.add_mesh(arrows, color="white", lighting=False)
#
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="displacement",
#         component=None,
#         lighting=False,
#         show_edges=False,
#         clim=[0, 70],
#         cmap=boring_cmap,
#         scalar_bar_args={
#             "title": 'Displacement (m)',
#             "position_x": 0.8,
#             "position_y": 0.2,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#     )
#
#     # Fix camera in default position otherwise mesh appears to jump around!
#     plotter.camera_position = [(750000.0, 1445500.0, 6291991.008627122),
#                         (750000.0, 1445500.0, 0.0),
#                         (0.0, 1.0, 0.0)]
#     plotter.add_text(f"Time: {i*2000:6} years", name='time-label')
#     plotter.write_frame()
#
#     if i == len(reader.time_values)-1:
#         # Write end frame multiple times to give a pause before gif starts again!
#         for j in range(20):
#             plotter.write_frame()
#
#     plotter.clear()
#
# # Closes and finalizes movie
# plotter.close()
# -
# Looking at the animation, we can see that as the weight of the ice load builds up the mantle deforms,
# pushing up material away from the ice load. If we kept the ice load fixed this forebulge will
# eventually grow enough that it balances the weight of the ice, i.e the mantle is in isostatic
# equilbrium and the deformation due to the ice load stops. At 100 thousand years when the ice is removed
# the topographic highs associated with forebulges are now out of equilibrium so the flow of material
# in the mantle reverses back towards the previously glaciated region.

# ![SegmentLocal](displacement_warp.gif "segment")

# References
# ----------
# Cathles L.M. (1975). *Viscosity of the Earth's Mantle*, Princeton University Press.
#
# Dahlen F. A. and Tromp J. (1998). *Theoretical Global Seismology*, Princeton University Press.
#
# Ranalli, G. (1995). Rheology of the Earth. Springer Science & Business Media.
#
# Weerdesteijn, M. F., Naliboff, J. B., Conrad, C. P., Reusen, J. M., Steffen, R., Heister, T., &
# Zhang, J. (2023). *Modeling viscoelastic solid earth deformation due to ice age and contemporary
# glacial mass changes in ASPECT*. Geochemistry, Geophysics, Geosystems.
#
# Wu P., Peltier W. R. (1982). *Viscous gravitational relaxation*, Geophysical Journal International.
#
# Zhong, S., Paulson, A., & Wahr, J. (2003). Three-dimensional finite-element modelling of Earth’s
# viscoelastic deformation: effects of lateral variations in lithospheric thickness. Geophysical
# Journal International.
