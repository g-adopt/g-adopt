# Idealised 2-D viscoelastic loading problem in a square box
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in a square box. Here we will focus purely on viscoelastic deformation by a surface load (i.e. a synthetic ice sheet!). In the next tutorials we will build up the complexity of our physical approximation of Glacial Isostatic Adjustment (GIA), linked to feedbacks from gravity, rotation, transient rheology and ultimately sea level.
#
# You may already have seen how G-ADOPT can be applied to mantle convection problems in our other tutorials. Generally the setup of the G-ADOPT model should be familiar but the equations we are solving and the necessary input fields will be slightly different.
#
# Governing equations
# -------------------
# Let's start by reviewing some equations! Similar to mantle convection, the governing equations for viscoelastic loading are derived from the
# conservation laws of mass and momentum.
#
# The conservation of momentum is
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma} - \rho \nabla \Phi = 0,
# \end{equation}
# where $\boldsymbol{\sigma}$ is the full stress tensor, $\rho$ is the density and $\Phi$ is the gravitational potential field. Similar to mantle convection, we have neglected inertial terms.
#
# For incompressible materials conservation of mass is
# \begin{equation}
#     \frac{\partial \rho}{\partial t} + \textbf{v} \cdot \nabla \rho = 0,
# \end{equation}
# where $\textbf{v}$ is the velocity. For the moment we are focusing on incompressible materials where
# \begin{equation}
#     \nabla \cdot \textbf{v} = 0.
# \end{equation}
#
#

# Linearisation
# -------------
# The conservation of momentum is usually linearised due to the small displacements relative to the depth of the mantle, i.e.
# \begin{equation}
#     \rho = \rho_0 + \rho_1,
# \end{equation}
# \begin{equation}
#     \boldsymbol{\sigma} = \boldsymbol{\sigma}_0 + \boldsymbol{\sigma}_1,
# \end{equation}
# \begin{equation}
#     \nabla \Phi = \nabla \Phi_0 + \nabla \Phi_1 = \textbf{g} + \nabla \Phi_1.
# \end{equation}
# Subbing this into the momentum equation and neglecting higher order terms gives
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_0 + \nabla \cdot \boldsymbol{\sigma}_1  - \rho_0 \textbf{g}  - \rho_1 \textbf{g} - \rho_0 \nabla \Phi_1 = 0.
# \end{equation}
# The background state is assumed to be in hydrostatic equilibrium so
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_0 = \rho_0 \textbf{g}.
# \end{equation}
# Therefore we can simplify the momentum balance to
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1   - \rho_1 \textbf{g} - \rho_0 \nabla \Phi_1 = 0.
# \end{equation}
# For this tutorial we are going to ignore changes in the gravitational field, so the last term drops out, giving
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1   - \rho_1 \textbf{g}  = 0.
# \end{equation}
#
# Linearising the conservation of mass gives
# \begin{equation}
#     \frac{\partial (\rho_0+\rho_1)}{\partial t} + \textbf{v} \cdot \nabla (\rho_0 + \rho_1) = 0.
# \end{equation}
# After neglecting higher order terms and assuming the background density field, $\rho_0$, does not vary in time, the conservation of mass becomes
# \begin{equation}
#     \frac{\partial \rho_1}{\partial t} + \textbf{v} \cdot \nabla \rho_0  = 0.
# \end{equation}
# Integrating this equation with respect to time gives
# \begin{equation}
#     \rho_1 = - \textbf{u} \cdot \nabla \rho_0 ,
# \end{equation}
# assuming $\rho_1 = 0$ at initial times. Note that $\textbf{u}$ is now the displacement vector. The density perturbation $\rho_1$ is referred to as the *Eulerian density pertubation* by the GIA community.
#

# Incremental Lagrangian Stress Tensor
# -------------------------------------
# The GIA community usually reformulate the linearised momentum equation in terms of the *Incremental lagrangian stress tensor*. This can be traced back to the early roots of the GIA modellers adopting Laplace transform methods. The idea behind this is to convert the time dependent viscoelastic problem to a time independent `elastic' problem by the correspondence principle.
#
# From an elastic wave theory point of view, \citet{dahlen_theoretical_1998} make the point that it is the Lagrangian perturbation in stress not the Eulerian perturbation that is related to the displacement gradient by the elastic parameters. Transforming between the Lagrangian perturbation in stress and the Eulerian description is given by
# \begin{equation}
#      \boldsymbol{\sigma}_{L1} =  \boldsymbol{\sigma}_1 + \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ,
# \end{equation}
# where $\boldsymbol{\sigma}_{L1}$ is the Incremental lagrangian stress tensor.
#
# This is effectively accounting for an advection of a background quantity when translating between the Eulerian and Lagrangian frames of reference through a first order Taylor series expansion.
#
# This advection of prestress can be important for very long wavelength loads. \citet{cathles_viscosity_2016} estimates that the term becomes leading order when the wavelength is greater than 30000 km for typical Earth parameters, i.e. only when the wavelength is the same order of magnitude as the circumference of the Earth.
#
# For the viscoelastic problem, however, this term is crucial because it acts as a restoring force to isostatic equilibrium. If the Laplace transform methods do not include this term a load placed on the surface of the Earth will keep sinking \cite{wu_viscous_1982}!
#
# Subbing into the stress balance gives
#
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1 = \nabla \cdot (\boldsymbol{\sigma}_{L1}  - \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ) = \nabla \cdot \boldsymbol{\sigma}_{L1} - \nabla (\rho_0 g u_r)
# \end{equation}
# where $u_r$ is the radial displacement vector.
#

# Maxwell Rheology
# ----------------
#
# The GIA community generally model the mantle as an incompressible Maxwell solid. The conceptual picture is a spring and a dashpot connected together in series \citep{ranalli_rheology_1995}. For this viscoelastic model the elastic and viscous stresses are the same but the total displacements combine.
#
# The viscous constitutive relationship is
# \begin{equation}
#     \overset{\cdot}{\boldsymbol{\epsilon}}^v = \dfrac{1}{2 \eta} (\boldsymbol{\sigma}_{L1} + p \textbf{I}).
# \end{equation}
# The corresponding elastic constitutive equation is
# \begin{equation}
#     \boldsymbol{\epsilon}^e  = \dfrac{1}{2 \mu} (\boldsymbol{\sigma}_{L1} + p \textbf{I})
# \end{equation}
# where $\boldsymbol{\epsilon}^v$ and $\boldsymbol{\epsilon}^e$ are the strain tensors for viscous and elastic deformation, $\eta$  is the viscosity, $\mu$ is the shear modulus, $\textbf{I}$ is the Identity matrix, $p$ is the pressure, and $\boldsymbol{\sigma}_{L1} $ is the incremental lagrangian stress tensor. An overhead dot notes the time derivative i.e the viscous strain rate is proportional to stress, while the elastic strain  is proportional to stress. The total strain is
# \begin{equation}
#     \boldsymbol{\epsilon} = \boldsymbol{\epsilon}^v + \boldsymbol{\epsilon}^e = \dfrac{1}{2} ( \nabla \textbf{u} + (\nabla \textbf{u})^T),
# \end{equation}
# where $\textbf{u}$ is the displacement vector.
# Taking the time derivative of the total strain and substituting this into the consitutive equations gives
# \begin{equation}
#     \boldsymbol{\sigma}_{L1}+ \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}}_{L1} = - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}}.
# \end{equation}
#

# Summary
# --------
# The linearised governing equations for an incompressible Maxwell body used by the GIA community are
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_{L1} - \nabla (\rho_0 g u_r)   - \rho_1 \textbf{g}  = 0,
# \end{equation}
# \begin{equation}
#     \nabla \cdot \textbf{v} = 0,
# \end{equation}
# \begin{equation}
#     \rho_1 = - \textbf{u} \cdot \nabla \rho_0,
# \end{equation}
# \begin{equation}
#     \boldsymbol{\sigma}_{L1}+ \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}}_{L1} = - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}} .
# \end{equation}
# Note as stated above, this still neglects perturbations in the gravitational field and we will leave solving the associated Poisson equation to later demos.
#

# This example
# -------------
# We are going to simulate a viscoelastic loading and unloading problem based on a 2D version of the test case presented in Weerdesteijn et al 2023.
#
# Let's get started! The first step is to import the `gadopt` module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from mpi4py import MPI
import numpy as np
from gadopt.utility import vertical_component as vc
import pandas as pd
# from gadopt.utility import step_func

# Next we need to create a mesh of the mantle region we want to simulate. The Weerdesteijn test case is a 3D box 1500 km wide horizontally and 2891 km deep. As mentioned to speed up things for this first demo let's just consider a 2D domain, i.e. taking a vertical cross section through the 3D box.
#
# For starters let's use one of the default meshes provided by Firedrake, `RectangleMesh`. We have chosen 80 triangular elements in the $x$ direction and 150 triangular elements in the $y$ direction. For real simulations we can use fully unstructured meshes to accurately resolve important features in the model, for instance near coastlines or sharp discontinuities in mantle properties.  We can print out the grid resolution using `log`, a utility provided by gadopt. (N.b. `log` is equivalent to python's `print` function, except simplifies outputs when running simulations in parallel.)
#
# Boundaries are automatically tagged by the built-in meshes supported by Firedrake. For the `RectangleMesh` being used here, tag 1 corresponds to the plane $x=0$; 2 to the $x=L$ plane; 3 to the $y=0$ plane; and 4 to the $y=D$ plane. For convenience, we can rename these to `left_id`, `right_id`, `bottom_id` and `top_id`.
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points
# in the negative z-direction. This attribute is used by G-ADOPT specifically, not
# Firedrake. By contrast, a non-Cartesian geometry is assumed to have gravity
# pointing in the radially inward direction.

# +
# Set up geometry:
L = 1500e3  # length of the domain in m
D = 2891e3  # Depth of domain in m
nx = 150  # Number of horizontal cells
nz = 80  # number of vertical cells

# Let's print out the grid resolution in km
log(f"Horizontal resolution {L/nx/1000} km")
log(f"Vertical resolution {D/nz/1000} km")

# mesh = RectangleMesh(nx, nz, L, D, name="mesh")
# mesh.cartesian = True
# +
# Set up geometry:
dx = 10e3  # horizontal grid resolution in m
L = 1500e3  # length of the domain in m
nz = 80  # number of vertical cells


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


# left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs
# -

# We also need function spaces, which is achieved by associating the
# mesh with the relevant finite element: `V` , `W`, `TP1` and `R` are symbolic
# variables representing function spaces. They also contain the
# function space’s computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis.

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
TP1 = TensorFunctionSpace(mesh, "DG", 2)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the velocity and pressure spaces to form
# a function space for the mixed Stokes problem, `Z`.

Z = MixedFunctionSpace([V, W])  # Mixed function space.

# We also specify functions to hold our solutions: `z` in the mixed
# function space, noting that a symbolic representation of the two
# parts – incremental displacement and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of z Incremental Displacement and Pressure.
#
# We also need to initialise two functions `displacement` and `deviatoric stress` that are used when timestepping the constitutive equation.

# +
z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
u_, p_ = z.subfunctions
u_.rename("Incremental Displacement")
p_.rename("Pressure")

displacement = Function(V, name="displacement").assign(0)
deviatoric_stress = Function(TP1, name='deviatoric_stress')
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

# Let's start initialising some parameters. First of all Firedrake has a helpful function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties. In this case the density, shear modulus and viscosity only vary in the vertical direction. We will approximate the series of layers using a smooth $tanh$ function with a width of 10 km.


# +

# layer properties from spada et al 2011
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
density_values = [3037, 3438, 3871, 4978, 10750]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0]
viscosity_values = [1e40, 1e21, 1e21, 2e21, 0]

'''
def initialise_background_field(field, background_values, vertical_tanh_width=10e3):
    profile = background_values[0]
    sharpness = 1 / vertical_tanh_width
    depth = X[1]
    for i in range(1, len(background_values)-1):
        centre = radius_values[i] - radius_values[0]
        mag = background_values[i] - background_values[i-1]
        profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

    field.interpolate(profile)
'''

density = Function(W, name="density")
initialise_background_field(density, density_values)

shear_modulus = Function(W, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)

viscosity = Function(W, name="viscosity")
initialise_background_field(viscosity, viscosity_values)
# -

# Next let's create a function to store our ice load. Following the long test from Weeredesteijn et al 2023 we'll grow a 1 km thick ice sheet over 90000 years and then remove the ice much more quickly by 100000 years. The width of the ice sheet is 100 km and we have again smoothed out the transition using a $tanh$ function.

# +
rho_ice = 931
g = 9.8125

# Initialise ice loading
ice_load = Function(W)
Hice = 1000
year_in_seconds = Constant(3600 * 24 * 365.25)
T1_load = 90e3 * year_in_seconds
T2_load = 100e3 * year_in_seconds

# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3
disc_dx = 5e3
k_disc = 2*pi/(8*disc_dx)  # wavenumber for disk 2pi / lambda
r = X[0]
disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
ramp = Constant(0)


# -

# Next let's define the length of our time step. If we want to accurately resolve the elastic response we should choose a timestep lower than the Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation to `catch up' the initial, instantaneous elastic deformation.
#
# Let's print out the maxwell time for each layer

# +

for layer_visc, layer_mu in zip(viscosity_values[:-1], shear_modulus_values[:-1]):
    log("Maxwell time:", float(layer_visc/layer_mu/year_in_seconds), "years")
# -

# As we can see the shortest Maxwell time is given by the lower mantle and is about 280 years, i.e. it will take about 280 years for the viscous deformation in that layer to catch up any instantaneous elastic deformation. Conversely the top layer, our lithosphere, has a maxwell time of $10^{21}$ years. This layer has effectively infinite viscosity ($\eta = 10^{40}$ Pa s) so the viscous deformation over the course of the simulation will always be negligible compared with the elastic deformation, given that our simulations only run for 110000 years. For now let's choose a conservative timestep of 50 years and an output time step of 10000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 50
dt = Constant(dt_years * year_in_seconds)
Tend_years = 110e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 10e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart*year_in_seconds)/dt)
log("max timesteps: ", max_timesteps)

dump_period = round(dt_out / dt)
log("dump_period:", dump_period)
log(f"dt: {float(dt / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")

do_write = True
# -

# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and side boundaries to be free slip, i.e. no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy` `gadopt` knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as indicating this is a free surface. As the weight of the ice load builds up, the mantle will deform elastically and viscously, pushing up mantle material away from the ice load. This forebulge will eventually grow enough that it balances the weight of the ice, i.e the mantle is in isostatic isostatic equilbrium and the deformation due to the ice load stops. When the ice is removed the topographic highs associated with forebulges are now out of equilibrium so the flow of material in the mantle reverses back towards the previously glaciated region.
#
# The `delta_rho_fs` option accounts for the density contrast across the free surface whether there is ice or air (or in later examples ocean!) above a particular region of the mantle.

# Setup boundary conditions
exterior_density = conditional(time < T2_load, rho_ice*disc, 0)
stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - exterior_density}},
    1: {'ux': 0},
    2: {'ux': 0},
}


# Firedrake can write out vtk files which can be read by programs like Paraview or pyvista

# We also need to specify a G-ADIOT approximation which sets up the various parameters and fields needed for the viscoelastic loading problem.


approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, viscosity, g=g)

# We finally come to solving the variational problem, with solver
# objects for the Stokes systems created. We pass in the solution fields z and various fields needed for the solve along with the approximation, timestep and boundary conditions.
#

stokes_solver = ViscoelasticStokesSolver(z, deviatoric_stress, displacement, approximation,
                                         dt, bcs=stokes_bcs)

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
prefactor_prestress = Function(W, name='prefactor prestress').interpolate(stokes_solver.prefactor_prestress)
effective_viscosity = Function(W, name='effective viscosity').interpolate(approximation.effective_viscosity(dt))

if do_write:
    # Create output file
    output_file = VTKFile(f"viscoelastic_loading/out_dtout{dt_out_years}a.pvd")
    output_file.write(u_, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity)

displacement_min_array = []
# -

# Let's setup some more helper functions for logging the displacement at the surface.

# +
P2 = FunctionSpace(mesh, "CG", 2)
vertical_displacement = Function(P2)
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


# -

# Now let's run the simulation! We are going to control the ice thickness using the ramp parameter. At each step we call `solve` to calculate the incremental displacement and pressure fields. This will update the displacement at the surface and stress values accounting for the time dependent Maxwell consitutive equation.

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
            output_file.write(u_, displacement, p_, stokes_solver.previous_stress, shear_modulus, viscosity, density, prefactor_prestress, effective_viscosity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u_, name="Incremental Displacement")
            checkpoint.save_function(p_, name="Pressure")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(deviatoric_stress, name="Deviatoric stress")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displacement_filename, displacement_min_array)
