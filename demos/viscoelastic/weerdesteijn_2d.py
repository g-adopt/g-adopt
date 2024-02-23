# 2d box model based on weerdestejin et al 2023

from gadopt import *
from mpi4py import MPI
import numpy as np

OUTPUT = True
# Set up geometry:
dx = 5e3  # horizontal grid resolution
L = 1500e3  # length of the domain in m

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]

thickness_values = [70e3, 350e3, 250e3, 2221e3, 3480e3]

density_values = [3037, 3438, 3871, 4978, 10750]

shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0]

viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

nx = round(L/dx)

D = radius_values[0]-radius_values[-1]
nz = 80
dz = D / nz  # because of extrusion need to define dz after
nz = round(D/dz)

LOAD_CHECKPOINT = False

checkpoint_file = ".h5"


if LOAD_CHECKPOINT:
    with CheckpointFile(checkpoint_file, 'r') as afile:
        mesh = afile.load_mesh("surface_mesh_extruded")
else:
    surface_mesh = IntervalMesh(150, L, name="surface_mesh")
    mesh = ExtrudedMesh(surface_mesh, nz, layer_height=dz)
    mesh.coordinates.dat.data[:, 1] -= D
    x, z = SpatialCoordinate(mesh)

    # rescale vertical resolution
    a = Constant(4)
    b = Constant(0)
    depth_c = 500.0
    z_scaled = z / D
    Cs = (1.-b) * sinh(a*z_scaled) / sinh(a) + b*(tanh(a*(z_scaled + 0.5))/(2*tanh(0.5*a)) - 0.5)
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([x, depth_c*z_scaled + (D - depth_c)*Cs]))
    mesh.coordinates.assign(f)

x, z = SpatialCoordinate(mesh)

bottom_id, top_id = "bottom", "top"  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
M = MixedFunctionSpace([V, W])  # Mixed function space.
TP1 = TensorFunctionSpace(mesh, "DG", 2)

m = Function(M)  # a field over the mixed function space M.
# Function to store the solutions:
if LOAD_CHECKPOINT:
    with CheckpointFile(checkpoint_file, 'r') as afile:
        u_dump = afile.load_function(mesh, name="Incremental Displacement")
        p_dump = afile.load_function(mesh, name="Pressure")
        u_, p_ = m.subfunctions
        u_.assign(u_dump)
        p_.assign(p_dump)
        displacement = afile.load_function(mesh, name="Displacement")
        deviatoric_stress = afile.load_function(mesh, name="Deviatoric stress")
else:
    u_, p_ = m.subfunctions
    displacement = Function(V, name="displacement").assign(0)
    deviatoric_stress = Function(TP1, name='deviatoric_stress')

u, p = split(m)  # Returns symbolic UFL expression for u and p

u_old = Function(V, name="u old")
u_old.assign(u_)

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

rho_ice = 931
g = 9.8125  # there is also a list but Aspect doesnt use...

viscosity = Function(W, name="viscosity")
for i in range(0, len(viscosity_values)-1):
    viscosity.interpolate(conditional(z >= radius_values[i+1] - radius_values[0],
                          conditional(z <= radius_values[i] - radius_values[0],
                          viscosity_values[i], viscosity), viscosity))

shear_modulus = Function(W, name="shear modulus")
for i in range(0, len(shear_modulus_values)-1):
    shear_modulus.interpolate(conditional(z >= radius_values[i+1] - radius_values[0],
                              conditional(z <= radius_values[i] - radius_values[0],
                              shear_modulus_values[i], shear_modulus), shear_modulus))

density = Function(W, name="density")
for i in range(0, len(density_values)-1):
    density.interpolate(conditional(z >= radius_values[i+1] - radius_values[0],
                        conditional(z <= radius_values[i] - radius_values[0],
                        density_values[i], density), density))

# Timestepping parameters
year_in_seconds = Constant(3600 * 24 * 365.25)

if LOAD_CHECKPOINT:
    time = Constant(110e3 * year_in_seconds)  # this needs to be changed!
else:
    time = Constant(0.0)

short_simulation = False

if short_simulation:
    dt = Constant(2.5 * year_in_seconds)  # Initial time-step
else:
    dt = Constant(50 * year_in_seconds)

if short_simulation:
    Tend = Constant(200 * year_in_seconds)
else:
    Tend = Constant(110e3 * year_in_seconds)

max_timesteps = round(Tend/dt)
log("max timesteps", max_timesteps)

if short_simulation:
    dt_out = Constant(10 * year_in_seconds)
else:
    dt_out = Constant(10e3 * year_in_seconds)

dump_period = round(dt_out / dt)
log("dump_period", dump_period)
log("dt", dt.values()[0])

ice_load = Function(W)

if short_simulation:
    T1_load = 100 * year_in_seconds
else:
    T1_load = 90e3 * year_in_seconds

T2_load = 100e3 * year_in_seconds

ramp = Constant(0)
if short_simulation:
    Hice = 100
else:
    Hice = 1000

disc_radius = 100e3
k_disc = 2*pi/(8*dx)  # wavenumber for disk 2pi / lambda
disc = 0.5*(1-tanh(k_disc * (x - disc_radius)))

ice_load.interpolate(ramp * rho_ice * g * Hice * disc)

approximation = SmallDisplacementViscoelasticApproximation(density, displacement, g=g)

# Write output files in VTK format:
# Next rename for output:
u_.rename("Incremental Displacement")
p_.rename("Pressure")
# Create output file and select output_frequency:
filename = "2d_weerdesteijn/"
if OUTPUT:
    output_file = File(filename+"out.pvd")

# Setup boundary conditions
stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'normal_stress': ice_load, 'free_surface': {'exterior_density': conditional(time < T2_load, rho_ice*disc, 0)}},
    1: {'ux': 0},
    2: {'ux': 0},
}


stokes_solver = ViscoelasticStokesSolver(m, viscosity, shear_modulus, density, deviatoric_stress, displacement, approximation, dt, bcs=stokes_bcs,
                                         cartesian=True)


prefactor_prestress = Function(W, name='prefactor prestress').interpolate(stokes_solver.prefactor_prestress)
effective_viscosity = Function(W, name='effective viscosity').interpolate(stokes_solver.effective_viscosity)

if OUTPUT:
    output_file.write(u_, u_old, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity)

# Now perform the time loop:
displacement_min_array = []
for timestep in range(1, max_timesteps+1):
    if short_simulation:
        ramp.assign(conditional(time < T1_load, time / T1_load, 1))
    else:
        ramp.assign(conditional(time < T1_load, time / T1_load,
                                conditional(time < T2_load, 1 - (time - T1_load) / (T2_load - T1_load),
                                            0)
                                )
                    )

    ice_load.interpolate(ramp * rho_ice * g * Hice * disc)

    stokes_solver.solve()

    time.assign(time+dt)

    # Write output:
    if timestep % dump_period == 0:
        log("timestep", timestep)
        log("time", time.values()[0])
        if OUTPUT:
            output_file.write(u_, u_old, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, Function(W, name='prefactor prestress').interpolate(stokes_solver.prefactor_prestress), Function(W, name='effective viscosity').interpolate(stokes_solver.effective_viscosity))

        with CheckpointFile(filename+"chk.h5", "w") as checkpoint:
            checkpoint.save_function(u_, name="Incremental Displacement")
            checkpoint.save_function(p_, name="Pressure")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(deviatoric_stress, name="Deviatoric stress")

    # Compute diagnostics:
    bc_displacement = DirichletBC(displacement.function_space(), 0, top_id)
    displacement_z_min = displacement.dat.data_ro_with_halos[bc_displacement.nodes, 1].min(initial=0)
    displacement_min = displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    log("Greatest (-ve) displacement", displacement_min)

    displacement_min_array.append([float(time/year_in_seconds), displacement_min])

if MPI.COMM_WORLD.rank == 0:
    np.savetxt("displacement-weerdesteijn-2d.dat", displacement_min_array)
