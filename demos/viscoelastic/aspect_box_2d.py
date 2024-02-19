# box model based on weerdestejin et al 2023

from gadopt import *
from mpi4py import MPI
import os
import numpy as np
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("date", help="date format: dd.mm.yy")
args = parser.parse_args()

    
OUTPUT=True
output_directory="/g/data/xd2/ws9229/viscoelastic/2d_aspect_box/"
# Set up geometry:
dx = 5e3  # horizontal grid resolution
L = 1500e3  # length of the domain in m

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]


thickness_values = [70e3, 350e3, 250e3, 2221e3, 3480e3]

density_values = [3037, 3438, 3871, 4978, 10750]

shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11,2.28340e11, 0]

viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

nx = round(L/dx)

D = radius_values[0]-radius_values[-1]
nz = 80 
dz = D / nz  # because of extrusion need to define dz after
nz = round(D/dz)

LOAD_CHECKPOINT = False

checkpoint_file ="/g/data/xd2/ws9229/viscoelastic/aspect_box/27.10.23_832cores_viscoelastic_weerdesteijn_aspectbox_dx5km_nz80scaled_a4_dt50years_dtout10000years_Tend110000years_extruded_zhongprefactor_oldFD_TDG2interp_strong1e10_drhorho1/chk.h5"


if LOAD_CHECKPOINT: 
    with CheckpointFile(checkpoint_file, 'r') as afile:
        mesh = afile.load_mesh("surface_mesh_extruded")
else:
 #   surface_mesh = SquareMesh(10, 10, L)
#    surface_mesh = Mesh("./aspect_box_refined_surface.msh", name="surface_mesh")
    surface_mesh = IntervalMesh(150, L, name="surface_mesh")
    mesh = ExtrudedMesh(surface_mesh, nz, layer_height=dz)
#    mesh = BoxMesh(10,10,10,L,L,D)
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
#bottom_id, top_id = 5, 6  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q3 = FunctionSpace(mesh, "CG", 3)  # Temperature function space (scalar)
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

#eta_surf = Function(W, name="eta")
#eta_eq = FreeSurfaceEquation(W, W, surface_id=top_id)

T = Function(Q, name="Temperature").assign(0)
# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# timestepping

rho_ice = 931
g= 9.8125  # there is also a list but Aspect doesnt use...

viscosity = Function(W, name="viscosity")
for i in range(0,len(viscosity_values)-1):
    viscosity.interpolate(
            conditional(z >= radius_values[i+1] - radius_values[0],
                conditional( z <= radius_values[i] - radius_values[0],
                    viscosity_values[i], viscosity), viscosity))

shear_modulus = Function(W, name="shear modulus")
for i in range(0,len(shear_modulus_values)-1):
    shear_modulus.interpolate(
            conditional(z >= radius_values[i+1] - radius_values[0],
                conditional( z <= radius_values[i] - radius_values[0],
                    shear_modulus_values[i], shear_modulus), shear_modulus))

density = Function(W, name="density")
for i in range(0,len(density_values)-1):
    density.interpolate(
            conditional(z >= radius_values[i+1] - radius_values[0],
                conditional( z <= radius_values[i] - radius_values[0],
                    density_values[i], density), density))



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

dt_elastic = Constant(dt)
#    dt_elastic = conditional(dt_elastic<2*maxwell_time, 2*0.0125*tau0, dt_elastic)
#    max_timesteps = round(20*tau0/dt_elastic)

if short_simulation:
    Tend = Constant(200* year_in_seconds) # do a test of checkpointing
else:
    Tend = Constant(110e3 * year_in_seconds)

max_timesteps = round(Tend/dt)
log("max timesteps", max_timesteps)

if short_simulation:
    dt_out = Constant(10 * year_in_seconds)
else:
    dt_out = Constant(50e3 * year_in_seconds)

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
r =  x#pow(pow(x, 2)+ pow(y, 2), 0.5)
k_disc = 2*pi/(8*dx)  # wavenumber for disk 2pi / lambda 
disc = 0.5*(1-tanh(k_disc *(r - disc_radius)))

ice_load.interpolate(ramp * rho_ice * g *Hice* disc/scale_mu)


previous_stress = Function(TP1, name='previous_stress').interpolate(prefactor_prestress * deviatoric_stress/scale_mu)
averaged_deviatoric_stress = Function(TP1, name='averaged deviatoric_stress')
# previous_stress = prefactor_prestress *  2 * effective_viscosity * sym(grad(u_old))

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(0)  # Rayleigh number
approximation = SmallDisplacementViscoelasticApproximation(density, displacement, g=g)

steady_state_tolerance = 1e-9

# Nullspaces and near-nullspaces:

# Write output files in VTK format:
# Next rename for output:
u_.rename("Incremental Displacement")
p_.rename("Pressure")
# Create output file and select output_frequency:
filename=os.path.join(output_directory, str(args.date))
filename += "_2d_new_viscoelastic_weerdesteijn_aspectbox_dx5km_nz"+str(nz)+"scaled_a4_dt"+str(round(dt/year_in_seconds))+"years_dtout"+str(round(dt_out.values()[0]/year_in_seconds))+"years_Tend"+str(round(Tend.values()[0]/year_in_seconds))+"years/"
if OUTPUT:
    output_file = File(filename+"out.pvd")
stokes_bcs = {
    bottom_id: {'uy': 0},
#        top_id: {'stress': -rho0 * g * (eta + dot(displacement, n)) * n},
#        top_id: {'stress': -rho0 * g * eta * n},
    top_id: {'stress': -ice_load*n },
    #    top_id: {'old_stress': prefactor_prestress*(-rho0 * g * (eta + dot(displacement,n)) * n)},
    1: {'ux': 0},
    2: {'ux': 0},
}


eta_fields = {'velocity': u_/dt,
                'surface_id': top_id}

eta_bcs = {} 

eta_strong_bcs = [InteriorBC(W, 0., top_id)]
stokes_solver = ViscoelasticStokesSolver(m, viscosity, shear_modulus, density, deviatoric_stress, approximation, dt, bcs=stokes_bcs, equations=ViscoElasticEquations,
                             cartesian=True)

#stokes_solver.solver_parameters['ksp_view']=None
#stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason']=None
#stokes_solver.solver_parameters['fieldsplit_0']['ksp_monitor_true_residual']=None
#stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason']=None
#stokes_solver.solver_parameters['fieldsplit_1']['ksp_monitor_true_residual']=None


#eta_timestepper = BackwardEuler(eta_eq, eta_surf, eta_fields, dt, bnd_conditions=eta_bcs, strong_bcs=eta_strong_bcs,solver_parameters=mumps_solver_parameters) 
#    stokes_solver.fields['source'] = div(previous_stress)
# analytical function
#eta_analytical = Function(Q3, name="eta analytical").interpolate(F0-eta)
if OUTPUT:
    output_file.write(u_, u_old, displacement, p_, previous_stress, shear_modulus, viscosity)

eta_midpoint =[]
#eta_midpoint.append(displacement.at(L/2+100, -0.001)[1])

displacement_vom_matplotlib_df = pd.DataFrame()
surface_nodes = []
surface_dx = 1000
surface_nx = round(L /surface_dx)

distance_from_centre =[]

for i in range(surface_nx):
    surface_nodes.append([i*surface_dx, -0.1])

surface_VOM = VertexOnlyMesh(mesh,surface_nodes, missing_points_behaviour='warn')
DG0_vom = FunctionSpace(surface_VOM, "DG", 0)
displacement_vom = Function(DG0_vom)

#DG0_vom_input_ordering = FunctionSpace(surface_VOM.input_ordering, "DG", 0)
#displacement_vom_input = Function(DG0_vom_input_ordering)

def displacement_vom_out(t):
    displacement_vom.interpolate(displacement[1])
    #displacement_vom_input.interpolate(displacement_vom) # new vom
    displacement_vom_array = displacement_vom.dat.data  # old vom
    displacement_vom_array = mesh.comm.gather(displacement_vom_array, root=0) # old vom
    if mesh.comm.rank == 0:
        # concatenate arrays
        displacement_vom_array_f = np.concatenate(displacement_vom_array)  # old vom
        displacement_vom_matplotlib_df['displacement_vom_array_{:.0f}years'.format(t/year_in_seconds.values()[0])] = displacement_vom_array_f # old vom need displacement_vom_input for new...
        t_inyears = t / year_in_seconds.values()[0]
    #    displacement_vom_matplotlib_df['displacement_vom_array_{:.0f}years'.format(t_inyears)] = displacement_vom_input.dat.data  # new
        displacement_vom_matplotlib_df.to_csv(filename+"displacement_oldvom_arrays.csv")

displacement_vom_out(0)
error = 0
# Now perform the time loop:

for timestep in range(1, max_timesteps+1):#int(max_timesteps/2)+1):
    if short_simulation:
        ramp.assign(conditional(time < T1_load, time / T1_load, 1))
    else:
        ramp.assign(conditional(time < T1_load, time / T1_load, 
                                conditional(time < T2_load, 1 - (time - T1_load) / (T2_load - T1_load),
                                            0)
                                )
                    )

#    print(ramp.values()[0]) 
    ice_load.interpolate(ramp * rho_ice * g *Hice* disc/scale_mu)

    stokes_solver.solve()
    
    time.assign(time+dt)
    

    
#            output_file.write(u_, u_old, displacement, p_, previous_stress, eta_analytical)
    # Write output:
    if timestep % dump_period == 0:
        log("timestep", timestep)
        log("time", time.values()[0])
        if OUTPUT:
            output_file.write(u_, u_old, displacement, p_, previous_stress, shear_modulus, viscosity)
#            displacement_vom.interpolate(displacement[2])

        with CheckpointFile(filename+"chk.h5", "w") as checkpoint:
            checkpoint.save_function(u_, name="Incremental Displacement")
            checkpoint.save_function(p_, name="Pressure")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(deviatoric_stress, name="Deviatoric stress")

    displacement_vom_out(time.values()[0])
    
#with open(filename+"_D3e6_visc1e21_shearmod1e11_nx"+str(nx)+"_dt"+str(dt_factor)+"tau_a6_refinemesh_nosurfadv_expfreesurface.txt", 'w') as file:
#    for line in eta_midpoint:
#        file.write(f"{line}\n")
#final_error = pow(error,0.5)/L





