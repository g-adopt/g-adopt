# Idealised 2-D viscoelastic loading problem in a square box
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in a square box.
# Here we will focus purely on viscoelastic deformation by a surface load, i.e.
# a synthetic ice sheet! The setup is similar to a 2-D version of the test case
# presented in [Weerdesteijn et al. (2023)](https://doi.org/10.1029/2022GC010813),
# but we include compressibility.
#
# You may already have seen how G-ADOPT can be applied to mantle convection
# problems in our other tutorials. Generally the setup of the G-ADOPT model
# should be familiar but the equations we are solving and the necessary input
# fields will be slightly different.
#
# Governing equations
# -------------------
# Let's start by reviewing some equations! Similar to mantle convection, the
# governing equations for viscoelastic loading are derived from the
# conservation laws of mass and momentum. The full GIA equations include
# rotational and gravitational effects arising from changes in the surface
# load as water is redistributed between ice and ocean, which is governed
# by the sea-level equation. For now let's neglect those terms, and
# concentrate on viscoelastic deformation.
#
# We start by assuming that the mantle is in a state of isostatic equilibrium
# before we apply any load at the Earth's surface. This allows us to split
# fields into a time-independent background component and a small
# time-dependent perturbation. For example, density can be written as
# $\rho = \rho_0 + \rho_1$, where $\rho_0$ is the background state and
# $\rho_1$ is the perturbation. Similarly, stress can be written as
# $\boldsymbol{\sigma} = \boldsymbol{\sigma}_0 + \boldsymbol{\sigma}_1^E$,
# where the superscript $E$ indicates an Eulerian perturbation.
#
# The background state of hydrostatic balance can then be written in
# nondimensional form as
#
# \begin{equation}
# \nabla \cdot \boldsymbol{\sigma}_0 = B_{\mu}\, \rho_0\, g\,\boldsymbol{\hat{e}}_k.
# \end{equation}
#
# where $ B_{\mu} = \frac{\bar{\rho} \bar{g} L}{\bar{\mu}}$ is a non-dimensional
# number describing the ratio of buoyancy to elastic shear strength,
# $\bar{\rho}$ is a characteristic density scale, $g$ is
# the non-dimensional gravity relative to a characteristic gravity scale, $\bar{g}$,
# $L$ is a characteristic length scale and $\mu$ is a characteristic shear
# modulus. Note that $\boldsymbol{\hat{e}}_k$ is aligned with either
# the $z$-axis or radial direction in Cartesian or spherical coordinates,
# respectively.
#
#
# Given the size and timescale of loads involved during GIA, the load induced
# deformation, $\textbf{u}$, is small typically relative to the depth of the
# mantle. As such, we can simplify our problem by linearising the equations of
# motion and neglecting higher order terms that involve products of small
# perturbations. We can also ignore inertial terms which will be very small.
# Making these assumptions yields the following non-dimensional linearised
# form of the momentum equation
#
# \begin{equation}
#    \textbf{0} = \nabla \cdot \boldsymbol{\sigma}_1^L -  B_{\mu} \nabla \left( \rho_0 g u_k\right) -  B_{\mu} \rho_1 g \, \boldsymbol{\hat{e}}_k.
# \end{equation}
#
# where $\boldsymbol{\sigma}_1^L$ is the incremental Lagrangian stress (see
# Eq. 3.16 of Dahlen and Tromp, 1998) and $u_k$ is the component of
# non-dimensional displacement in the $\boldsymbol{\hat{e}}_k$ direction.
#
# We write the momentum equation using $\boldsymbol{\sigma}_1^L$ because
# viscoelastic constitutive relations that link stress to strain (and their
# time derivatives) are fundamentally defined in a Lagrangian framework.
# Substituting $\boldsymbol{\sigma}_1^L$ for $\boldsymbol{\sigma}_1^E$ gives
# rise to the second term in the momentum equation above, referred to as the
# *advection of hydrostatic prestress*. This term accounts for the change in
# stress due to displacement of material through the background stress gradient.
# The advection of hydrostatic prestress can be important for very long
# wavelength loads. Cathles (1975) estimates that the term becomes leading
# order when the wavelength is greater than 30,000 km for typical Earth
# parameters, i.e. only when the wavelength is the same order of magnitude
# as the circumference of the Earth. For the viscoelastic problem, however,
# this term is crucial because it acts as a restoring force to isostatic
# equilibrium. If the Laplace transform methods do not include this term
# a load placed on the surface of the Earth will keep sinking
# ([Wu and Peltier, 1982](https://doi.org/10.1111/j.1365-246X.1982.tb04976.x))!
#
# We obtain the non-dimensional linearised density perturbation, $\rho_1$,
# by linearising the conservation of mass equation
#
# \begin{equation}
#     \partial_t \rho_1 + \nabla \cdot (\rho_0\, \boldsymbol{v}) = 0,
# \end{equation}
#
# where $\textbf{v}$ is the velocity and we have neglected the
# $\nabla \cdot (\rho_1\, \boldsymbol{v})$ term, by assuming that
# perturbations in density are small compared with background
# values.
# In non-dimensionalising the velocity, we have used the
# scale $\frac{L}{\bar{\alpha}}$, where $L$ is a characteristic length scale
# and $\bar{\alpha}$ is a characteristic Maxwell time that describes the
# transition between dominantly elastic and viscous behaviour
#
# \begin{equation}
#    \bar{\alpha} = \frac{\bar{\eta}}{\bar{\mu}},
# \end{equation}
#
# where $\bar{\eta}$ and $\bar{\mu}$ are characteristic dynamic viscosity and
# shear modulus values respectively.
#
#
# Integrating the conservation of mass equation through time
# gives
# \begin{equation}
#     \rho_1 = - \nabla \cdot \left( \rho_0\, \boldsymbol{u} \right)
# \end{equation}
# assuming a vanishing initial displacement, $\boldsymbol{u}(t=0) = \textbf{0}$.
# The density perturbation $\rho_1$ is often referred to as the
# *Eulerian density perturbation* by the GIA community.
#

# Boundary Conditions
# --------------------
# To complete the problem specification, we must also define the boundary
# conditions. At the top of the domain (i.e.,  Earth's surface), normal stress
# is balanced by the applied surface load such that
#
# \begin{equation}
#     \hat{\boldsymbol{n}} \cdot \boldsymbol{\sigma}_1^L = - B_{\mu} \, \rho_\mathrm{load} \, g \, h_\mathrm{load} \, \boldsymbol{\hat{e}}_k,
# \end{equation}
#
# where $\rho_\mathrm{load}$ and $h_\mathrm{load}$ are the non-dimensional
# density and vertical thickness of the surface load, and
# $\hat{\boldsymbol{n}}$ is the outward unit normal vector to the boundary.
# For this tutorial we impose a no-normal-displacement condition at side walls
# and the base of the domain (i.e., core–mantle boundary)
#
# \begin{equation}
#     \boldsymbol{u} \cdot \hat{\boldsymbol{n}} = 0.
# \end{equation}
#
# Finally, we assume continuity of both displacement and traction across all
# internal boundaries, such that
# \begin{align}
#     [\boldsymbol{u}]^+_- &= \boldsymbol{0}, \\
#     [\hat{\boldsymbol{n}} \cdot \boldsymbol{\sigma}_{L1}]^+_- &= \boldsymbol{0},
# \end{align}
#
# where $[\cdot]^+_-$ denotes the jump in the associated property across an
# interface.

# Viscoelastic Rheology
# ----------------
#
# The GIA community generally model the mantle as a Maxwell solid. The
# conceptual picture is a spring and a dashpot connected together in series
# ([Ranalli, 1995](https://link.springer.com/book/9780412546709)).
# For this viscoelastic model the elastic and viscous stresses are the same
# but the total displacements combine.
#
#
# We follow the internal variable formulation adopted by Al-Attar and
# Tromp (2014) and Crawford et al. (2017, 2018), in which viscoelastic
# constitutive equations are expressed in integral form and reformulated using
# so-called *internal variables*. Conceptually, this approach consists of a set
# of elements with different shear relaxation timescales, arranged in parallel.
# This formulation provides a compact, flexible and convenient means to
# incorporate transient rheology into viscoelastic deformation models: using a
# single internal variable is equivalent to a simple Maxwell material; two
# correspond to a Burgers model with two characteristic relaxation frequencies;
# and using a series of internal variables permits approximation of a
# continuous range of relaxation timescales for more complicated rheologies.
#
# For a linear, compressible viscoelastic material, the constitutive equation
# takes the form
# \begin{equation}
#     \boldsymbol{\sigma}^L_1 = \kappa \nabla \cdot \boldsymbol{u}(t)\, \boldsymbol{I} + 2 \mu_0 \boldsymbol{d}(t) - 2 \sum_i \mu_i \boldsymbol{m}_i(t),
# \end{equation}
#
#
# where $\kappa$ is the non-dimensional bulk modulus and $\mu_0$ is the
# non-dimensional effective shear modulus given by
# \begin{equation}
#     \mu_0 = \sum_i \mu_i
# \end{equation}
# where $\mu_i$ are the non-dimensional shear moduli associated with each
# internal variable, $\boldsymbol{m}_i$. The deviatoric strain tensor is given
# as
#
# \begin{equation}
#     \boldsymbol{d} = \boldsymbol{e} -\frac{1}{3} \textrm{Tr}(\boldsymbol{e}) \boldsymbol{I},
# \end{equation}
#
#  where $\textrm{Tr}(\cdot)$ is the trace operator and the strain tensor, $\boldsymbol{e}$, is
# \begin{equation}
#    \boldsymbol{e}  = \frac{1}{2} \left( \nabla \boldsymbol{u}  + \left( \nabla \boldsymbol{u}\right)^T \right).
# \end{equation}
#
# We note that all non-dimensional moduli are obtained by division of terms by
# the characteristic shear modulus $\bar{\mu}$.
#
# Each internal variable, $\boldsymbol{m}_i$, is defined by
#
# \begin{equation}
#     \boldsymbol{m}_i = \frac{1}{\alpha_i} \int^t_{t_0} \textrm{e}^{-\frac{(t-t')}{\alpha_i}} \boldsymbol{d}(t') \, dt',
# \end{equation}
# where $\alpha_i$ is the Maxwell time for each element.
#
# Equivalently, each internal variable evolves according to
# \begin{equation}
#     \partial_t \boldsymbol{m}_i + \frac{1}{\alpha_i} \left( \boldsymbol{m}_i - \boldsymbol{d} \right) = \boldsymbol{0}, \quad \boldsymbol{m}_i(t_0) = \boldsymbol{0}.
# \end{equation}
#
#

# Time discretisation
# -------------------
#
# One of the key differences with the mantle convection demos is that the
# constitutive equation now depends on time. We discretise the internal
# variable evolution equation in time using the implicit Backward Euler (BE)
# scheme. This choice allows us to take timesteps larger than the characteristic
#  Maxwell time without compromising numerical stability. Such flexibility is
# particularly advantageous when the timescale of glacial loading is
# substantially slower than the Maxwell time -- as is often the case in
# low-viscosity regions -- thereby avoiding having to take prohibitively small
# timesteps in realistic simulations of glacial cycles. Applying the BE scheme,
# the evolution of each internal variable becomes
#
# \begin{equation}
#     \boldsymbol{m}^{n+1}_i = \dfrac{1}{1 + \dfrac{\Delta t}{\alpha_i}} \left(\boldsymbol{m}^{n}_i + \dfrac{\Delta t}{\alpha_i}\, \boldsymbol{d}(\boldsymbol{u}^{n+1}) \right),
# \end{equation}
#
# where the superscript $n$ refers to the previous timestep, $n+1$ is the next
# timestep, and $\Delta t$ is the timestep duration.
#

# Weak form and spatial discretisation
# ----------
# To derive the finite-element discretisation of these governing equations, we
# first translate them into their weak form. By selecting appropriate function
# spaces that contain both solution fields and test functions, the weak form
# can be obtained by multiplying the equations by their test functions and
# integrating over the domain, $\Omega$. For conservation of momentum, we use
# the (vector) test function, $\boldsymbol{\phi}$, to give
# \begin{equation}
#    0 =  \int_\Omega \boldsymbol{\phi} \cdot \left(\nabla \cdot \boldsymbol{\sigma}_1^L -  B_{\mu} \nabla \left( \rho_0 g u_k\right) -  B_{\mu} \rho_1 g \, \boldsymbol{\hat{e}}_k \right) dx .
# \end{equation}
#
# A this stage, it becomes necessary to define the finite element spaces that
# will be used for spatial discretisation. Generally, for each component of
# displacement, we use $Q2$ finite elements on hexahedral meshes (i.e., the
# piecewise continuous tri-quadratic tensor product of quadratic continuous
# polynomials in each direction). For the deviatoric strain tensor (and hence
# the internal variable), since it is proportional to the gradient of
# displacement, we choose the discontinuous $DG1$ space (i.e., linear
# variations within each finite element cell and discontinuous jumps between
# cells) for each component. For purely radial variations in density, viscosity
# and shear modulus, we choose the $DG0$ space (i.e., constant within a finite
# element cell but discontinuous between cells), while for laterally varying
# viscosity fields, we again select $DG1$ finite element functions.
# Once these spaces have been chosen, we can integrate various terms by parts
# within each element to introduce weak boundary conditions and to move
# derivatives from the trial function to the test function.
#
# The final system of equations is
#
# \begin{align}
#     \int_{\Omega}  \nabla \boldsymbol{\phi} : \left( \kappa \nabla \cdot \boldsymbol{u}^{n+1}\, \boldsymbol{I}  +  2 \sum_i \dfrac{\eta_i}{\alpha_i + \Delta t} \boldsymbol{d}(\boldsymbol{u}^{n+1})  \right) \, dx  \\
#     - \int_{\Omega} \nabla \cdot \boldsymbol{\phi} B_{\mu} \rho_0 g u_k^{n+1} \, dx  + \int_{\Gamma} \boldsymbol{\phi} \cdot B_{\mu} u_k^{n+1} [[\rho_0 g \hat{\boldsymbol{n}}]]  \, ds  + \int_{\partial \Omega_\textrm{top}} \boldsymbol{\phi} \cdot \hat{\boldsymbol{n}} B_{\mu} \rho_0 g u_k^{n+1} \, ds \\
#     - \int_\Omega \boldsymbol{\phi} \cdot  B_{\mu}  g \left(\boldsymbol{u}^{n+1} \cdot \nabla \rho_0 +  \rho_0\, \nabla  \cdot \boldsymbol{u}^{n+1} \right) \, \boldsymbol{\hat{e}}_k  \, dx \\
#     = - \int_{\partial \Omega_\textrm{top}}  \boldsymbol{\phi} \cdot \left( B_{\mu} \, \rho_\mathrm{load} \, g\, h_\mathrm{load}  \hat{\boldsymbol{n}} \right) \, ds + \int_{\Omega}  \nabla \boldsymbol{\phi} : \left( 2 \sum_i \dfrac{\eta_i}{\alpha_i + \Delta t}  \boldsymbol{m}_i^n  \right) \, dx .
# \end{align}
#
# Briefly, the first line corresponds to the stress term after integration
# by parts. The second corresponds to the advection of hydrostatic prestress
# term. Since our $DG0$ discretisation allows density and gravity
# discontinuities, we can no longer assume that integrating along both
# sides of interior boundaries in the mesh will cancel out. As such, we
# have to write an additional term accounting for the 'jumps' in material
# properties along interior boundaries, $\Gamma$, denoted by the square
# brackets. The third line corresponds to the buoyancy term involving
# $\rho_1$ and the final line includes the surface load boundary
# condition at the top of the domain, $\partial \Omega_\textrm{top}$
# and the contributibution of the internal variable from the previous
# timestep.
#
# Implicit terms involving the unknown displacement at the next time step,
#  $\boldsymbol{u}^{n+1}$, have been collected on the left-hand side and
# explicit terms known from the current time step are on the right-hand side.
# Finding the new values of $\boldsymbol{u}^{n+1}$ requires solving a linear
# system at each timestep, which is solved in Firedrake through PETSc's
# comprehensive linear algebra library. For more details on the numerical
# discretisation please refer to Scott et al. (2025).
#

# This example
# -------------
# We will simulate a viscoelastic loading and unloading problem based on a 2D
# version of the test case presented in Weerdesteijn et al. (2023).
#
# Let's get started! The first step is to import the `gadopt` module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import extruded_layer_heights
from gadopt.utility import initialise_background_field

# Next we need to create a mesh of the mantle region we want to simulate. The
# Weerdesteijn test case is a 3D box 1500 km wide horizontally and 2891 km deep.
# To speed up things for this first demo, we consider a 2D domain, i.e. taking
# a vertical cross section through the 3D box.
#
# We have chosen a mesh with 60 elements in the $x$ direction and 5 elements
# per rheological layer in the $y$ direction. It is worth emphasising that the
# setup has coarse grid resolution so that the demo is quick to run! For real
# simulations we can use fully unstructured meshes to accurately resolve
# important features in the model, for instance near coastlines or sharp
# discontinuities in mantle properties.
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity
# points in the negative z-direction. This attribute is used by G-ADOPT
# specifically, not Firedrake. By contrast, a non-Cartesian geometry is assumed
# to have gravity pointing in the radially inward direction.
#
# Boundaries are automatically tagged by the built-in meshes supported by
# Firedrake. We can use G-ADOPT's `get_boundary_ids` function to inspect
# the mesh to detect this and allow the boundaries to be referred to more intuitively
# (e.g. `boundary.left`, `boundary.top`, etc.).

# +
L = 1500e3  # Length of the domain in m
radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
D = radius_values[0]-radius_values[-1]
L_tilde = L / D
radius_values_tilde = np.array(radius_values)/D
nx = 60

surface_mesh = IntervalMesh(nx, L_tilde)

layer_heights = extruded_layer_heights(5, radius_values_tilde)

mesh = ExtrudedMesh(
    surface_mesh,
    layers=len(layer_heights),
    layer_height=layer_heights,
)

mesh.cartesian = True

boundary = get_boundary_ids(mesh)
# -

# We now need to choose finite element function spaces. `V` , `S`, `DG0` and `R`
# are symbolic variables representing function spaces. They also contain the
# function space's computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis. We will choose Q2 for the displacement similar to the
# velocity field in our mantle convection demos. We also initialise a
# discontinuous tensor function space that will store our previous values of the
# internal variable as the gradient of the continous displacement field will be
# discontinuous. Given that we have purely radial variations in density,
# viscosity and shear modulus, we choose the DG0 space (i.e., constant within a
# finite element cell but discontinuous between cells).

# Set up function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Displacement function space
S = TensorFunctionSpace(mesh, "DQ", 1)  # Stress tensor function space
DG0 = FunctionSpace(mesh, "DQ", 0)  # Density/viscosity/shear modulus function space
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Let's use the python package *PyVista* to visualise the resulting mesh.

# + tags=["active-ipynb"]
# import pyvista as pv
# import matplotlib.pyplot as plt
#
# VTKFile("mesh.pvd").write(Function(V))
# mesh_data = pv.read("mesh/mesh_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(edges, color="black")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# We now specify a function to store our displacement solution. We also need to
# initialise a function to store the old value of the internal variable. Since
# we are assuming Maxwell rheology we only need one internal variable.

# +
u = Function(V, name="displacement")  # field to hold our displacement solution

m = Function(S, name="Internal variable")  # Lagged internal variable at previous timestep
m_list = [m]
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF) using `log`, a utility provided by
# G-ADOPT. (N.b. `log` is equivalent to python's `print` function, except that
# it simplifies outputs for parallel simulations.)

# Output function space information:
log("Number of Displacement DOF:", V.dim())
log("Number of Internal variable DOF:", S.dim())

# Let's start initialising some parameters. First of all Firedrake has a helpful
# function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties.
# In this case the density, shear modulus and viscosity only vary in the vertical
# direction. The layer properties specified are from
# [Spada et al. (2011)](https://doi.org/10.1111/j.1365-246X.2011.04952.x).

# +
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e40, 1e21, 1e21, 2e21]

# Characteristic scales used for non-dimensionalisation
density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale
gravity_scale = 9.81
B_mu = Constant(density_scale * D * gravity_scale / shear_modulus_scale)

density_values_tilde = np.array(density_values)/density_scale
shear_modulus_values_tilde = np.array(shear_modulus_values)/shear_modulus_scale
viscosity_values_tilde = np.array(viscosity_values)/viscosity_scale
bulk_modulus_values_tilde = 2 * shear_modulus_values_tilde

density = Function(DG0, name="density")
initialise_background_field(
    density, density_values_tilde, X, radius_values_tilde,
    shift=radius_values_tilde[-1])

shear_modulus = Function(DG0, name="shear modulus")
initialise_background_field(
    shear_modulus, shear_modulus_values_tilde, X, radius_values_tilde,
    shift=radius_values_tilde[-1])

bulk_modulus = Function(DG0, name="bulk modulus")
initialise_background_field(
    bulk_modulus, bulk_modulus_values_tilde, X, radius_values_tilde,
    shift=radius_values_tilde[-1])

viscosity = Function(DG0, name="viscosity")
initialise_background_field(
    viscosity, viscosity_values_tilde, X, radius_values_tilde,
    shift=radius_values_tilde[-1])
# -

# We can also plot the viscosity field using *PyVista*.

# + tags=["active-ipynb"]
# VTKFile("viscosity.pvd").write(viscosity)
# visc_data = pv.read("viscosity/viscosity_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(visc_data,
#                  log_scale=True,
#                  scalar_bar_args={
#                      "title": 'log10(Viscosity)',
#                      "position_x": 0.8,
#                      "position_y": 0.2,
#                      "vertical": True,
#                      "title_font_size": 20,
#                      "label_font_size": 16,
#                      "fmt": "%.0f",
#                      "font_family": "arial",
#                  }
#                  )
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# Next let's define the length of our time step. If we want to accurately
# resolve the elastic response we should choose a timestep lower than the
# Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for
# the viscous deformation to 'catch up' with the initial, instantaneous elastic
# deformation.
#
# Let's print out the Maxwell time for each layer

year_in_seconds = 8.64e4 * 365.25
for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(layer_visc/layer_mu/year_in_seconds):.0f} years")


# As we can see the shortest Maxwell time is given by the lower mantle and is
# about 280 years, i.e. it will take about 280 years for the viscous
# deformation in that layer to catch up any instantaneous elastic deformation.
# Conversely the top layer, our lithosphere, has a Maxwell time of 6 x10$^{21}$
# years. Given that our simulations only run for 110,000 years the viscous
# deformation over the course of the simulation will always be negligible
# compared with the elastic deformation. For now let's choose a timestep of
# 1000 years and an output frequency of 2000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds / characteristic_maxwell_time)

dt_years = 1000
dt = Constant(dt_years * year_in_seconds / characteristic_maxwell_time)
Tend_years = 110e3
Tend = Constant(Tend_years * year_in_seconds / characteristic_maxwell_time)
dt_out_years = 2e3
dt_out = Constant(dt_out_years * year_in_seconds / characteristic_maxwell_time)

max_timesteps = round((Tend - Tstart * year_in_seconds/characteristic_maxwell_time) / dt)
log("max timesteps: ", max_timesteps)

output_frequency = round(dt_out / dt)
log("output_frequency:", output_frequency)
log(f"dt: {float(dt * characteristic_maxwell_time / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")
# -

# Next let's setup our ice load. Following the long test from Weeredesteijn et
# al. (2023), during the first 90 thousand years of the simulation the ice sheet
# will grow to a thickness of 1 km. The ice thickness will rapidly shrink to ice
#  free conditions in the next 10 thousand years. Finally, the simulation will
# run for a further 10 thousand years to allow the system to relax towards
# isostatic equilibrium. This is approximately the length of an
# interglacial-glacial cycle. The width of the ice sheet is 100 km and we have
# used a tanh function again to smooth out the transition from ice to
# ice-free regions.
#
# As the loading and unloading cycle only varies linearly in time, let's write
# the ice load as a symbolic expression.

# Initialise ice loading
rho_ice = 931 / density_scale
g = 9.81 / gravity_scale
Hice = 1000 / D
t1_load = 90e3 * year_in_seconds / characteristic_maxwell_time
t2_load = 100e3 * year_in_seconds / characteristic_maxwell_time
ramp_after_t1 = conditional(
    time < t2_load, 1 - (time - t1_load) / (t2_load - t1_load), 0
)
ramp = conditional(time < t1_load, time / t1_load, ramp_after_t1)
# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3 / D
disc_dx = 5e3 / D
k_disc = 2*pi/(8*disc_dx)  # wavenumber for disk 2pi / lambda
r = X[0]
disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
ice_load = ramp * B_mu * rho_ice * g * Hice * disc

# We can now define the boundary conditions to be used in this simulation.
# Let's set the bottom and side boundaries to be free slip with no normal
# flow $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy`,
# G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of
# the ice load, as well as indicating this is a free surface.

# Setup boundary conditions
stokes_bcs = {
    boundary.bottom: {'uy': 0},
    boundary.top: {'free_surface': {'normal_stress': ice_load}},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
}

# We also need to specify a G-ADOPT approximation which sets up the various
# parameters and fields needed for the viscoelastic loading problem.

approximation = MaxwellApproximation(
    bulk_modulus=bulk_modulus,
    density=density,
    shear_modulus=shear_modulus,
    viscosity=viscosity,
    B_mu=B_mu)

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields `u` and
# various fields needed for the solve along with the approximation, timestep,
# list of internal variables and boundary conditions.

stokes_solver = InternalVariableSolver(u, approximation, dt=dt, m_list=m_list, bcs=stokes_bcs)

# We next set up our output, in VTK format. This format can be read by programs
# like pyvista and Paraview. We also create a function to store the velocity
# field at each timestep.

# +
# initialise velocity and old displacement functions for plotting
velocity = Function(u, name="velocity")
disp_old = Function(u, name="old_disp").assign(u)

# Create output file
output_file = VTKFile("output.pvd")
output_file.write(u, *m_list, velocity)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max"
)
gd = GeodynamicalDiagnostics(u, density, boundary.bottom, boundary.top)

checkpoint_filename = "viscoelastic_loading-chk.h5"
# -

# Now let's run the simulation! We are going to control the ice thickness using
# the `ramp` parameter. At each step we call `solve` to calculate the
# incremental displacement and pressure fields. This will update the
# displacement at the surface and stress values accounting for the time
# dependent Maxwell constitutive equation.

# +
for timestep in range(max_timesteps):
    time.assign(time+dt)
    stokes_solver.solve()

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(dt)} {gd.u_rms()} "
                 f"{gd.u_rms_top()} {gd.ux_max(boundary.top)}")

    if timestep % output_frequency == 0:
        log("timestep", timestep)
        velocity.interpolate((u - disp_old)/dt)
        disp_old.assign(u)
        output_file.write(u, *m_list, velocity)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(u, name="displacement")
            for i, m_i in enumerate(m_list):
                checkpoint.save_function(m_i, name=f"internal_variable_{i}")


plog.close()
# -

# We can use *PyVista* to create an animation of the displacement through time.
# We will use the calculated displacement to artifically scale the mesh.
# We have exaggerated the stretching by a factor of 500, **BUT...** it is
# important to remember this is just for ease of visualisation - the mesh
# is not moving in reality!

# + tags=["active-ipynb"]
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
#     # Artificially warp the output data in the vertical direction by the free
#     # surface height. Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=500)
#     arrows = data.glyph(orient="velocity", scale="velocity", factor=75000, tolerance=0.05)
#     plotter.add_mesh(arrows, color="white", lighting=False)
#
#     data['displacement'] *= D  # Make displacement dimensional
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="displacement",
#         component=None,
#         lighting=False,
#         show_edges=False,
#         clim=[0, 140],
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
#     plotter.camera_position = [(L_tilde/2, -0.5, 2.3),
#                         (L_tilde/2, -0.5, 0.0),
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

# Looking at the animation, we can see that as the weight of the ice load builds
# up the mantle deforms, pushing up material away from the ice load. If we kept
# the ice load fixed this forebulge will eventually grow enough that it balances
# the weight of the ice, i.e the mantle is in isostatic equilibrium and the
# deformation due to the ice load stops. At 100 thousand years when the ice is
# removed the topographic highs associated with forebulges are now out of
# equilibrium so the flow of material in the mantle reverses back towards the
# previously glaciated region.

# ![SegmentLocal](displacement_warp.gif "segment")

# + [markdown] tags=["exercise"]
# Exercises
# --------------
# Now some exercises for you to try!
# 1. Try making the width of the ice load larger - how does this change the
# resulting displacement field?
# 2. By default G-ADOPT includes fully compressible effects but we can approximate
# an incompressible simulation by setting the parameter `bulk_shear_ratio` in
# the `approximation` to a large number e.g. 1000. How much does this effect the
# # solution?
# 3. Try varying the viscosity values and thickness of the rheological model. What
# happens if we remove the lithosphere by decreasing the effective viscosity in the
# top layer? Can you find the tradeoff between viscosity jumps, layer thickness and
# load wavelength based on the 'Cathles parameter' by $Ct = \eta*(D/\lambda)^3$,
# where $\eta*$ is the viscosity contrast, $D$ is the thickness of the Low Viscosity
# Zone, and $\lambda$ is the flow wavelength. For more details see Richards and
# Lenardic, 2018)
# -

# References
# ----------
# Cathles L.M. (1975). *Viscosity of the Earth's Mantle*, Princeton University Press.
#
# Dahlen F. A. and Tromp J. (1998). *Theoretical Global Seismology*, Princeton
# University Press.
#
# Ranalli, G. (1995). Rheology of the Earth. Springer Science & Business Media.
#
# Richards, M. A., & Lenardic, A. (2018). *The Cathles parameter (Ct): A geodynamic
# definition of the asthenosphere and implications for the nature of plate tectonics.*
# Geochemistry, Geophysics, Geosystems.
#
# Weerdesteijn, M. F., Naliboff, J. B., Conrad, C. P., Reusen, J. M., Steffen, R.,
# Heister, T., & Zhang, J. (2023). *Modeling viscoelastic solid earth deformation
# due to ice age and contemporary glacial mass changes in ASPECT*. Geochemistry,
# Geophysics, Geosystems.
#
# Wu P., Peltier W. R. (1982). *Viscous gravitational relaxation*, Geophysical Journal
# International.
