# Idealised 2-D viscoelastic loading problem in a square box
# ==========================================================

# In this tutorial, we examine an idealised 2-D loading problem in a square box. Here we
# will focus purely on viscoelastic deformation by a surface load (i.e. a synthetic ice
# sheet!). In the next tutorials we will build up the complexity of our physical
# approximation of Glacial Isostatic Adjustment (GIA), linked to feedbacks from gravity,
# rotation, transient rheology and ultimately sea level.

# You may already have seen how G-ADOPT can be applied to mantle convection problems in
# our other tutorials. Generally the setup of the G-ADOPT model should be familiar but
# the equations we are solving and the necessary input fields will be slightly
# different.

# Governing equations
# -------------------
# Let's start by reviewing some equations! Similar to mantle convection, the governing
# equations for viscoelastic loading are derived from the conservation laws of mass and
# momentum.

# The conservation of momentum is

# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma} - \rho \nabla \Phi = 0,
# \end{equation}

# where $\boldsymbol{\sigma}$ is the full stress tensor, $\rho$ is the density, and
# $\Phi$ is the gravitational potential field. Similar to mantle convection, we have
# neglected inertial terms.

# For incompressible materials conservation of mass is

# \begin{equation}
#     \frac{\partial \rho}{\partial t} + \textbf{v} \cdot \nabla \rho = 0,
# \end{equation}

# where $\textbf{v}$ is the velocity. For the moment we are focusing on incompressible
# materials where

# \begin{equation}
#     \nabla \cdot \textbf{v} = 0.
# \end{equation}

# Linearisation
# -------------
# The conservation of momentum is usually linearised due to the small displacements
# relative to the depth of the mantle, i.e.

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

# For this tutorial we are going to ignore changes in the gravitational field, so the
# last term drops out, giving

# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1   - \rho_1 \textbf{g}  = 0.
# \end{equation}

# Linearising the conservation of mass gives

# \begin{equation}
#     \frac{\partial (\rho_0+\rho_1)}{\partial t} + \textbf{v} \cdot \nabla (\rho_0 + \rho_1) = 0.
# \end{equation}

# After neglecting higher order terms and assuming the background density field,
# $\rho_0$, does not vary in time, the conservation of mass becomes

# \begin{equation}
#     \frac{\partial \rho_1}{\partial t} + \textbf{v} \cdot \nabla \rho_0  = 0.
# \end{equation}

# Integrating this equation with respect to time gives

# \begin{equation}
#     \rho_1 = - \textbf{u} \cdot \nabla \rho_0 ,
# \end{equation}

# assuming $\rho_1 = 0$ at initial times. Note that $\textbf{u}$ is now the displacement
# vector. The density perturbation $\rho_1$ is referred to as the *Eulerian density
# pertubation* by the GIA community.

# Incremental Lagrangian Stress Tensor
# -------------------------------------
# The GIA community usually reformulate the linearised momentum equation in terms of the
# *Incremental lagrangian stress tensor*. This can be traced back to the early roots of
# the GIA modellers adopting Laplace transform methods. The idea behind this is to
# convert the time dependent viscoelastic problem to a time independent `elastic'
# problem by the correspondence principle.

# From an elastic wave theory point of view, \citet{dahlen_theoretical_1998} make the
# point that it is the Lagrangian perturbation in stress not the Eulerian perturbation
# that is related to the displacement gradient by the elastic parameters. Transforming
# between the Lagrangian perturbation in stress and the Eulerian description is given by

# \begin{equation}
#     \boldsymbol{\sigma}_{L1} =  \boldsymbol{\sigma}_1 + \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ,
# \end{equation}

# where $\boldsymbol{\sigma}_{L1}$ is the incremental Lagrangian stress tensor.

# This is effectively accounting for an advection of a background quantity when
# translating between the Eulerian and Lagrangian frames of reference through a first
# order Taylor series expansion.

# This advection of prestress can be important for very long wavelength loads.
# \citet{cathles_viscosity_2016} estimates that the term becomes leading order when the
# wavelength is greater than 30000 km for typical Earth parameters, i.e. only when the
# wavelength is the same order of magnitude as the circumference of the Earth.

# For the viscoelastic problem, however, this term is crucial because it acts as a
# restoring force to isostatic equilibrium. If the Laplace transform methods do not
# include this term a load placed on the surface of the Earth will keep sinking
# \cite{wu_viscous_1982}!

# Subbing into the stress balance gives

# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1 = \nabla \cdot (\boldsymbol{\sigma}_{L1}  - \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ) = \nabla \cdot \boldsymbol{\sigma}_{L1} - \nabla (\rho_0 g u_r)
# \end{equation}

# where $u_r$ is the radial displacement vector.

# Maxwell Rheology
# ----------------
# The GIA community generally model the mantle as an incompressible Maxwell solid. The
# conceptual picture is a spring and a dashpot connected together in series
# \citep{ranalli_rheology_1995}. For this viscoelastic model the elastic and viscous
# stresses are the same but the total displacements combine.

# The viscous constitutive relationship is

# \begin{equation}
#     \overset{\cdot}{\boldsymbol{\epsilon}}^v = \dfrac{1}{2 \eta} (\boldsymbol{\sigma}_{L1} + p \textbf{I}).
# \end{equation}

# The corresponding elastic constitutive equation is

# \begin{equation}
#     \boldsymbol{\epsilon}^e  = \dfrac{1}{2 \mu} (\boldsymbol{\sigma}_{L1} + p \textbf{I})
# \end{equation}

# where $\boldsymbol{\epsilon}^v$ and $\boldsymbol{\epsilon}^e$ are the strain tensors
# for viscous and elastic deformation, $\eta$  is the viscosity, $\mu$ is the shear
# modulus, $\textbf{I}$ is the Identity matrix, $p$ is the pressure, and
# $\boldsymbol{\sigma}_{L1} $ is the incremental lagrangian stress tensor. An overhead
# dot notes the time derivative i.e the viscous strain rate is proportional to stress,
# while the elastic strain  is proportional to stress. The total strain is

# \begin{equation}
#     \boldsymbol{\epsilon} = \boldsymbol{\epsilon}^v + \boldsymbol{\epsilon}^e = \dfrac{1}{2} ( \nabla \textbf{u} + (\nabla \textbf{u})^T),
# \end{equation}

# where $\textbf{u}$ is the displacement vector.

# Taking the time derivative of the total strain and substituting this into the
# consitutive equations gives

# \begin{equation}
#     \boldsymbol{\sigma}_{L1}+ \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}}_{L1} = - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}}.
# \end{equation}

# Summary
# --------
# The linearised governing equations for an incompressible Maxwell body used by the GIA
# community are

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

# Note as stated above, this still neglects perturbations in the gravitational field and
# we will leave solving the associated Poisson equation to later demos.


# This example
# -------------
# We are going to simulate a viscoelastic loading and unloading problem based on a 2D
# version of the test case presented in Weerdesteijn et al. (2023).

# Let's get started! The first step is to import the `gadopt` module, which provides
# access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import vertical_component

# Next we need to create a mesh of the mantle region we want to simulate. The
# Weerdesteijn test case is a 3D box 1500 km wide horizontally and 2891 km deep. As
# mentioned to speed up things for this first demo let's just consider a 2D domain, i.e.
# taking a vertical cross section through the 3D box.

# For starters let's use one of the default meshes provided by Firedrake,
# `RectangleMesh`. We have chosen 80 triangular elements in the $x$ direction and 150
# triangular elements in the $y$ direction. For real simulations we can use fully
# unstructured meshes to accurately resolve important features in the model, for
# instance near coastlines or sharp discontinuities in mantle properties.  We can print
# out the grid resolution using `log`, a utility provided by gadopt. (N.b. `log` is
# equivalent to python's `print` function, except simplifies outputs when running
# simulations in parallel.)

# Boundaries are automatically tagged by the built-in meshes supported by Firedrake. For
# the `RectangleMesh` being used here, tag 1 corresponds to the plane $x=0$; 2 to the
# $x=L$ plane; 3 to the $y=0$ plane; and 4 to the $y=D$ plane. For convenience, we can
# rename these to `left_id`, `right_id`, `bottom_id` and `top_id`.

# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points in the
# negative z-direction. This attribute is used by G-ADOPT specifically, not Firedrake.
# By contrast, a non-Cartesian geometry is assumed to have gravity pointing in the
# radially inward direction.

# +
# Set up geometry:
L = 1500e3  # length of the domain in m
D = 2891e3  # Depth of domain in m
nx = 150  # Number of horizontal cells
nz = 80  # number of vertical cells
dz = D / nz  # because of extrusion need to define dz after

# Let's print out the grid resolution in km
log(f"Horizontal resolution {L / nx / 1e3} km")
log(f"Vertical resolution {D / nz / 1e3} km")

surface_mesh = IntervalMesh(nx, L, name="Surface mesh")
mesh = ExtrudedMesh(surface_mesh, nz, layer_height=dz)
mesh.cartesian = True
mesh.coordinates.dat.data[:, -1] -= D

X = SpatialCoordinate(mesh)
X_vert = vertical_component(X)

# rescale vertical resolution
a, b = 4.0, 0.0
z_scaled = X_vert / D
Cs = (1.0 - b) * sinh(a * z_scaled) / sinh(a) + b * (
    tanh(a * (z_scaled + 0.5)) / (2 * tanh(0.5 * a)) - 0.5
)

depth_c = 500.0
scaled_z_coordinates = depth_c * z_scaled + (D - depth_c) * Cs
mesh.coordinates.interpolate(
    as_vector([*[X[i] for i in range(len(X) - 1)], scaled_z_coordinates])
)
# -

# We also need function spaces, which is achieved by associating the mesh with the
# relevant finite element: `V` , `W`, `TP1` and `R` are symbolic variables representing
# function spaces. They also contain the function space’s computational implementation,
# recording the association of degrees of freedom with the mesh and pointing to the
# finite element basis. # Function spaces can be combined in the natural way to create
# mixed function spaces, combining the velocity and pressure spaces to form a function
# space for the mixed Stokes problem, `Z`.

# +
# Set up function spaces - currently using the bilinear Q2Q1 element pair:
# (Incremental) Displacement function space (vector)
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
# (Discontinuous) Stress tensor function space (tensor)
TP1 = TensorFunctionSpace(mesh, "DQ", 2)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Output function space information:
log(f"Number of Velocity DOF: {V.dim()}")
log(f"Number of Pressure DOF: {W.dim()}")
log(f"Number of Velocity and Pressure DOF: {V.dim() + W.dim()}")

z = Function(Z)  # a field over the mixed function space Z
z.subfunctions[0].rename("Incremental displacement")
z.subfunctions[1].rename("Pressure")
u, p = split(z)  # Returns indexed UFL expressions for u and p
# -


# +
def layered_initial_condition(values, radii):
    radius = radii.pop()
    value = values.pop()

    if radii:
        return conditional(
            X_vert <= radius - radii[0], value, layered_initial_condition(values, radii)
        )
    else:
        return value


# layer properties from Spada et al. (2011)
radius_values = [6371e3, 6301e3, 5951e3, 5701e3]
rho_values = [3037, 3438, 3871, 4978]
G_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
mu_values = [1e40, 1e21, 1e21, 2e21]

rho = Function(W, name="Density")
rho.interpolate(layered_initial_condition(rho_values, radius_values.copy()))
G = Function(W, name="Shear modulus")
G.interpolate(layered_initial_condition(G_values, radius_values.copy()))
mu = Function(W, name="Viscosity")
mu.interpolate(layered_initial_condition(mu_values, radius_values))
# -

# Next let's define the length of our time step. If we want to accurately resolve the
# elastic response we should choose a timestep lower than the Maxwell time,
# $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation
# to `catch up' the initial, instantaneous elastic deformation.

# Let's print out the Maxwell time for each layer
year_in_seconds = 8.64e4 * 365.25
for mu_layer, G_layer in zip(mu_values, G_values):
    log(f"Maxwell time: {float(mu_layer / G_layer / year_in_seconds)} years")

# As we can see the shortest Maxwell time is given by the lower mantle and is about 280
# years, i.e. it will take about 280 years for the viscous deformation in that layer to
# catch up any instantaneous elastic deformation. Conversely the top layer, our
# lithosphere, has a maxwell time of $10^{21}$ years. This layer has effectively
# infinite viscosity ($\eta = 10^{40}$ Pa s) so the viscous deformation over the course
# of the simulation will always be negligible compared with the elastic deformation,
# given that our simulations only run for 110000 years. For now let's choose a
# conservative timestep of 50 years and an output time step of 10000 years.

# +
# Timestepping parameters
time_start_years = 0
time_start = time_start_years * year_in_seconds
time_end_years = 110e3
time_end = time_end_years * year_in_seconds
time = Function(R).assign(time_start)

dt_years = 50
dt_out_years = 10e3
dt = dt_years * year_in_seconds
dt_out = dt_out_years * year_in_seconds

max_timesteps = round((time_end - time_start) / dt)
dump_period = round(dt_out / dt)
log(f"max timesteps: {max_timesteps}")
log(f"dump_period: {dump_period}")
log(f"dt: {dt / year_in_seconds} years")
log(f"Simulation start time: {time_start_years} years")

# Next let's create a function to store our ice load. Following the long test from
# Weeredesteijn et al. (2023) we'll grow a 1 km thick ice sheet over 90000 years and
# then remove the ice much more quickly by 100000 years. The width of the ice sheet is
# 100 km and we have again smoothed out the transition using a $tanh$ function.

# +
rho_ice = 931
g = 9.8125

# Initialise ice loading
t1_load = 90e3 * year_in_seconds
t2_load = 100e3 * year_in_seconds
ramp_after_t1 = conditional(
    time < t2_load, 1 - (time - t1_load) / (t2_load - t1_load), 0
)
ramp = conditional(time < t1_load, time / t1_load, ramp_after_t1)

# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3
disc_delta_x = 5e3
k_disc = 2 * pi / 8 / disc_delta_x  # wavenumber for disk 2pi / lambda
disc = 0.5 * (1 - tanh(k_disc * (X[0] - disc_radius)))
Hice = 1000

# ice_load = Function(W)
ice_load = ramp * rho_ice * g * Hice * disc
# -

# We can now define the boundary conditions to be used in this simulation. Let's set the
# bottom and side boundaries to be free slip, i.e. no normal flow
# $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy` `gadopt` knows
# to specify these as Strong Dirichlet boundary conditions.

# For the top surface we need to specify a normal stress, i.e. the weight of the ice
# load, as well as indicating this is a free surface. As the weight of the ice load
# builds up, the mantle will deform elastically and viscously, pushing up mantle
# material away from the ice load. This forebulge will eventually grow enough that it
# balances the weight of the ice, i.e the mantle is in isostatic isostatic equilbrium
# and the deformation due to the ice load stops. When the ice is removed the topographic
# highs associated with forebulges are now out of equilibrium so the flow of material in
# the mantle reverses back towards the previously glaciated region.

# The `delta_rho_fs` option accounts for the density contrast across the free surface
# whether there is ice or air (or in later examples ocean!) above a particular region of
# the mantle.

# Setup boundary conditions
rho_ext = conditional(time < t2_load, rho_ice * disc, 0)
stokes_bcs = {
    "bottom": {"uy": 0},
    "top": {
        "normal_stress": ice_load,
        "free_surface": {"rho_diff": rho - rho_ext},
    },
    1: {"ux": 0},
    2: {"ux": 0},
}

# We also need to specify a G-ADOPT approximation, which sets up the various parameters
# and fields needed for the viscoelastic loading problem.

# +
approximation = Approximation(
    "VE", dimensional=True, parameters={"G": G, "g": g, "mu": mu, "rho": rho}
)
# -

# We finally come to solving the variational problem, with solver objects for the Stokes
# systems created. We pass in the solution fields z and various fields needed for the
# solve along with the approximation, timestep and boundary conditions.

# +
displ = Function(V, name="Displacement")
tau_old = Function(TP1, name="Deviatoric stress (old)")

viscoelastic_solver = ViscoelasticSolver(
    approximation, z, displ, tau_old, dt, bcs=stokes_bcs
)
# -

# We next set up our output, in VTK format. This format can be read by programs like
# PyVista and ParaView.

# Create output file
output_file = VTKFile(f"viscoelastic_loading/out_dtout{dt_out_years}a.pvd")
output_file.write(*z.subfunctions, displ, tau_old, mu, rho, G)

# Now let's run the simulation! We are going to control the ice thickness using the
# ramp parameter. At each step we call `solve` to calculate the incremental displacement
# and pressure fields. This will update the displacement at the surface and stress
# values accounting for the time dependent Maxwell consitutive equation.

# +
import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

displ_vert = Function(FunctionSpace(mesh, "CG", 2))
displ_min_array = []

checkpoint_filename = "viscoelastic_loading-chk.h5"
displ_filename = "displacement-weerdesteijn-2d.dat"
bc_fs = Function(W, name="Normal stress")
for timestep in range(max_timesteps):
    # ice_load.interpolate(ramp * rho_ice * g * Hice * disc)

    viscoelastic_solver.solve()

    time.assign(time + dt)

    # Compute diagnostics:
    displ_vert.interpolate(vertical_component(displ))
    displ_bc = DirichletBC(displ_vert.function_space(), 0, "top")
    displ_z_min = displ_vert.dat.data_ro_with_halos[displ_bc.nodes].min(initial=0)
    # Minimum displacement at surface (should be top left corner with greatest (-ve)
    # deflection due to ice loading
    displ_min = displ_vert.comm.allreduce(displ_z_min, MPI.MIN)
    log(f"Greatest (-ve) displacement: {displ_min}")
    displ_min_array.append([float(time / year_in_seconds), displ_min])

    # Write output:
    if (timestep + 1) % dump_period == 0:
        log(f"timestep: {timestep}")

        output_file.write(*z.subfunctions, displ, tau_old, mu, rho, G)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Viscoelastic")
            checkpoint.save_function(displ, name="Displacement")
            checkpoint.save_function(tau_old, name="Deviatoric stress (old)")

        if MPI.COMM_WORLD.rank == 0:
            np.savetxt(displ_filename, displ_min_array)
