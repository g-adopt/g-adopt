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

# Linearisation
# -------------

# Incremental Lagrangian Stress Tensor
# -------------------------------------

# Maxwell Rheology
# ----------------

# Summary
# --------

# This example
# -------------
# We are going to simulate a viscoelastic loading and unloading problem based on a 2D
# version of the test case presented in Weerdesteijn et al. (2023).

# Let's get started! The first step is to import the `gadopt` module, which provides
# access to Firedrake and associated functionality.

import numpy as np
from mpi4py import MPI

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
log(f"Horizontal resolution {L/nx/1000} km")
log(f"Vertical resolution {D/nz/1000} km")

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
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())

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

mu = Function(W, name="Viscosity")
mu.interpolate(layered_initial_condition(mu_values, radius_values.copy()))
G = Function(W, name="Shear modulus")
G.interpolate(layered_initial_condition(G_values, radius_values.copy()))
rho = Function(W, name="Density")
rho.interpolate(layered_initial_condition(rho_values, radius_values))
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
    approximation, z, displ, tau_old, dt, bcs=stokes_bcs, solver_parameters="direct"
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
displ_vert = Function(FunctionSpace(mesh, "CG", 2))
displ_min_array = []

checkpoint_filename = "viscoelastic_loading-chk.h5"
displ_filename = "displacement-weerdesteijn-2d.dat"
bc_fs = Function(W, name="Normal stress")
for timestep in range(max_timesteps):
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
