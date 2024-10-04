from gadopt import *
from mpi4py import MPI
import numpy as np
from gadopt.utility import vertical_component as vc
from gadopt.utility import CombinedSurfaceMeasure, step_func
import pandas as pd

do_write = True

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
density_values = [3037, 3438, 3871, 4978, 10750]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0]
viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

# +
# Set up geometry:
dx = 10e3  # horizontal grid resolution in m
L = 1500e3  # length of the domain in m
nz = 80  # number of vertical cells
D = radius_values[0]-radius_values[-1]


nx = L/dx
dz = D / nz  # because of extrusion need to define dz after
surface_mesh = IntervalMesh(nx, L, name="surface_mesh")
mesh = ExtrudedMesh(surface_mesh, nz, layer_height=dz)
mesh.cartesian = True
vertical_component = 1
vertical_squashing = True
vertical_tanh_width = None
mesh.coordinates.dat.data[:, vertical_component] -= D

if vertical_squashing:
    # rescale vertical resolution
    X = SpatialCoordinate(mesh)
    a = Constant(4)
    b = Constant(0)
    depth_c = 500.0
    z_scaled = X[vertical_component] / D
    Cs = (1.-b) * sinh(a*z_scaled) / sinh(a) + b*(tanh(a*(z_scaled + 0.5))/(2*tanh(0.5*a)) - 0.5)
    Vc = mesh.coordinates.function_space()

    scaled_z_coordinates = [X[i] for i in range(vertical_component)]
    scaled_z_coordinates.append(depth_c*z_scaled + (D - depth_c)*Cs)
    f = Function(Vc).interpolate(as_vector(scaled_z_coordinates))
    mesh.coordinates.assign(f)

ds = CombinedSurfaceMeasure(mesh, degree=6)
X = SpatialCoordinate(mesh)
bottom_id, top_id = "bottom", "top"  # Boundary IDs for extruded meshes


# -

def initialise_background_field(field, background_values):
    if vertical_tanh_width is None:
        for i in range(0, len(background_values)-1):
            field.interpolate(conditional(X[vertical_component] >= radius_values[i+1] - radius_values[0],
                              conditional(X[vertical_component] <= radius_values[i] - radius_values[0],
                              background_values[i], field), field))
    else:
        profile = background_values[0]
        sharpness = 1 / vertical_tanh_width
        depth = initialise_depth()
        for i in range(1, len(background_values)-1):
            centre = radius_values[i] - radius_values[0]
            mag = background_values[i] - background_values[i-1]
            profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

        field.interpolate(profile)


# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
M = MixedFunctionSpace([V, W])  # Mixed function space.
M = M
TP1 = TensorFunctionSpace(mesh, "DG", 2)
R = FunctionSpace(mesh, "R", 0)

m = Function(M)  # a field over the mixed function space M.
u_, p_ = m.subfunctions
displacement = Function(V, name="displacement").assign(0)
deviatoric_stress = Function(TP1, name='deviatoric_stress')

u, p = split(m)  # Returns symbolic UFL expression for u and p

u_old = Function(V, name="u old")
u_old.assign(u_)
vertical_displacement = Function(V.sub(1), name="vertical displacement")  # Function to store vertical displacement for output

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

# Timing info:
stokes_stage = PETSc.Log.Stage("stokes_solve")
# -

rho_ice = 931
g = 9.8125  # there is also a list but Aspect doesnt use...

# +
viscosity = Function(W, name="viscosity")
initialise_background_field(viscosity, viscosity_values)


shear_modulus = Function(W, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)

density = Function(W, name="density")
initialise_background_field(density, density_values)

# +
# Timestepping parameters
year_in_seconds = Constant(3600 * 24 * 365.25)
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 50
dt = Constant(dt_years * year_in_seconds)
Tend_years = 110e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 10e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart*year_in_seconds)/dt)
log("max timesteps", max_timesteps)

dump_period = round(dt_out / dt)
log("dump_period", dump_period)
log("dt", dt.values()[0])
log(f"Simulation start time {Tstart} years")

# +
# Initialise ice loading
ice_load = Function(W)
T1_load = 90e3 * year_in_seconds
Hice = 1000

T2_load = 100e3 * year_in_seconds

# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3
disc_dx = 5e3
k_disc = 2*pi/(8*disc_dx)  # wavenumber for disk 2pi / lambda
r = X[0]
disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
ramp = Constant(0)

if do_write:
    File("ice.pvd").write(ice_load)
# -

# Setup boundary conditions
exterior_density = conditional(time < T2_load, rho_ice*disc, 0)
stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - exterior_density}},
    1: {'ux': 0},
    2: {'ux': 0},
}


approximation = SmallDisplacementViscoelasticApproximation(density, displacement, g=g)

stokes_solver = ViscoelasticStokesSolver(m, viscosity, shear_modulus, density,
                                         deviatoric_stress, displacement, approximation,
                                         dt, bcs=stokes_bcs)

# +
prefactor_prestress = Function(W, name='prefactor prestress').interpolate(stokes_solver.prefactor_prestress)
effective_viscosity = Function(W, name='effective viscosity').interpolate(stokes_solver.effective_viscosity)

if do_write:
    # Rename for output
    u_.rename("Incremental Displacement")
    p_.rename("Pressure")
    # Create output file
    output_file = VTKFile(f"viscoelastic_loading/out_dtout{dt_out_years}a.pvd")
    output_file.write(u_, u_old, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity, vertical_displacement)

displacement_min_array = []

# +

displacement_vom_matplotlib_df = pd.DataFrame()
surface_nodes = []
surface_nx = round(L / (0.5*dx))

for i in range(surface_nx):
    surface_nodes.append([i*0.5*dx, 0])

if mesh.comm.rank == 0:
    displacement_vom_matplotlib_df['surface_points'] = surface_nodes
surface_VOM = VertexOnlyMesh(mesh, surface_nodes, missing_points_behaviour='warn')
DG0_vom = VectorFunctionSpace(surface_VOM, "DG", 0)
displacement_vom = Function(DG0_vom)

DG0_vom_input_ordering = VectorFunctionSpace(surface_VOM.input_ordering, "DG", 0)
displacement_vom_input = Function(DG0_vom_input_ordering)


def displacement_vom_out():
    displacement_vom.interpolate(displacement)
    displacement_vom_input.interpolate(displacement_vom)
    if mesh.comm.rank == 0:
        log("check min displacement", displacement_vom_input.dat.data[:, 1].min(initial=0))
        log("check arg min displacement", displacement_vom_input.dat.data[:, 1].argmin())
        for i in range(mesh.geometric_dimension()):
            displacement_vom_matplotlib_df[f'displacement{i}_vom_array_{float(time/year_in_seconds):.0f}years'] = displacement_vom_input.dat.data[:, i]
        displacement_vom_matplotlib_df.to_csv(f"{name}/surface_displacement_arrays.csv")


# +
checkpoint_filename = "viscoelastic_loading-chk.h5"
displacement_filename = "displacement-weerdesteijn-2d.dat"

for timestep in range(1, max_timesteps+1):
    ramp.assign(conditional(time < T1_load, time / T1_load,
                            conditional(time < T2_load, 1 - (time - T1_load) / (T2_load - T1_load),
                                        0)
                            )
                )

    ice_load.interpolate(ramp * rho_ice * g * Hice * disc)

    stokes_solver.solve()

    time.assign(time+dt)
    # Compute diagnostics:
    vertical_displacement.interpolate(vc(displacement))
    bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, top_id)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (-ve) displacement", displacement_min)
    displacement_min_array.append([float(time/year_in_seconds), displacement_min])
    if timestep % dump_period == 0:
        log("timestep", timestep)

        if do_write:
            output_file.write(u_, u_old, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity, vertical_displacement)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u_, name="Incremental Displacement")
            checkpoint.save_function(p_, name="Pressure")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(deviatoric_stress, name="Deviatoric stress")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)

# -
