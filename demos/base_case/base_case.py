# Idealised 2-D mantle convection problem in a square box
# =======================================================
#
# In this tutorial, we examine the slow creeping motion of Earth’s
# mantle over geological timescales. We start with an idealised 2-D
# problem.
#
# Governing equations
# -------------------
#
# The equations governing mantle convection are derived from the
# conservation laws of mass, momentum and energy.  The simplest
# mathematical formulation assumes incompressibility and the
# Boussinesq approximation (e.g. McKenzie et al., 1973), under which
# the non–dimensional momentum and continuity equations are given
# by:
#
# $$\nabla \cdot \mathbb{\sigma} + Ra_0 \ T \ \hat{k} = 0,$$
# $$\nabla \cdot \vec{u} = 0$$
#
# where $\sigma$ is the stress tensor, $\vec{u}$ is the velocity and T
# temperature. $\hat{k}$ is the unit vector in the direction opposite
# to gravity and $Ra_0$ denotes the Rayleigh number, a dimensionless
# number that quantifies the vigor of convection:
#
# $$Ra0 = \frac{\rho_0 \alpha \Delta T g d^3}{\mu_0 \kappa}$$
#
# Here, $\rho_0$ denotes reference density, $\alpha$ the thermal
# expansion coefficient, $\Delta T$ the characteristic temperature
# change across the domain, $g$ the gravitational acceleration, $d$
# the characteristic length, $\mu_0$ the reference dynamic viscosity
# and $\kappa$ the thermal diffusivity. The mantle's Rayleigh number
# is estimated to be between $10^7$ and $10^9$, but we will focus on
# cases at a lower convective vigor in this notebook.
#
# When simulating incompressible flow, the full stress tensor,
# $\sigma$, is decomposed into deviatoric and volumetric components:
# $$ \sigma = \tau - p I,$$
# where $\tau$ is the deviatoric stress tensor, $p$ is dynamic
# pressure and $I$ is the identity matrix. Substituting this into the
# first equation presented above and utilizing the constitutive
# relation,
#
# $$\tau = 2\mu \dot\epsilon =
#   2\mu \operatorname{sym}(\nabla \vec{u}) =
#   \mu\left[ \nabla \vec{u} + \left(\nabla \vec{u}\right)^T\right] $$
#
# which relates the deviatoric stress tensor, $\tau$, to the
# strain-rate tensor, $\dot\epsilon=\operatorname{sym}(\nabla
# \vec{u})$, yields:
#
# $$ \nabla \cdot \mu \left[{\nabla\vec{u}} + \left(\nabla\vec{u}\right)^T\right]
#   - \nabla p + Ra_{0} T\hat{\vec{k}} = 0. $$
#
# The viscous flow problem can thus be posed in terms of pressure,
# $p$, velocity, $\vec{u}$, and temperature, $T$.
#
# The evolution of the thermal field is controlled by an
# advection-diffusion equation, where, for simplicity, we ignore
# internal heating:
# $$ \frac{\partial T}{\partial t} + \vec{u}\cdot \nabla T - \nabla \cdot \left(\kappa \nabla T\right) = 0 $$
# These governing equations are sufficient to solve for the three
# unknowns, together with adequate boundary and initial conditions.
#
# Weak formulation
# ----------------
#
# For the finite element discretisation of these equations, we start
# by writing them in their weak form.  We select appropriate function
# spaces V, W, and Q that contain, respectively, the solution fields
# for velocity u, pressure p, and temperature T , and also contain the
# test functions v, w and q. The weak form is then obtained by
# multiplying these equations with the test functions and integrating
# over the domain $\Omega$,
#
# $$\int_\Omega (\nabla \vec{v})\colon \mu \left[ \nabla \vec{u} + \left( \nabla \vec{u} \right)^T\right] \ dx
#  - \int_{\Omega} \left( \nabla \cdot \vec{v}\right)\ p \ dx
#  - \int_{\Omega} Ra_0\ T\ \vec{v}\cdot\hat{k} \ dx = 0 \ \text{ for all } v\in V,$$
#
# $$ \int_\Omega w \nabla \cdot \vec{u} \ dx\ \text{ for all } v\in V,$$
#
# $$  \int_\Omega q\frac{\partial T}{\partial t} \ dx
#   + \int_\Omega q \vec{u}\cdot \nabla T \ dx
#   + \int_\Omega \left(\nabla q\right) \cdot \left(\kappa \nabla T\right) \ dx = 0   \text{ for all } q\in Q.$$
#
# Note that we have integrated by parts the viscosity and pressure
# gradient terms in the Stokes equations, and the diffusion term in
# the energy equation, but have omitted the corresponding boundary
# terms.
#
# Solution procedure
# ------------------
#
# For temporal integration, we apply a simple $\theta$ scheme to the
# energy equation:
#
# $$
#   F_{\text{energy}}(q; T^{n+1}) :=
#     \int_\Omega q \frac{T^{n+1} - T^n}{\Delta t} dx
#   + \int_\Omega q\vec{u}\cdot\nabla T^{n+\theta} dx
#   + \int_\Omega \left(\nabla q\right)\cdot \left(\kappa \nabla
#   T^{n+\theta}\right) dx = 0
#   \text{ for all } q\in Q,
# $$
#
# where
# $$
#   T^{n+\theta} = \theta T^{n+1} + (1-\theta) T^n
# $$
#
# is interpolated between the temperature solutions $T^n$ and
# $T^{n+1}$ at the beginning and end of the $n+1$-th time step using a
# parameter $0\leq\theta\leq 1$.  In this example we use a
# Crank-Nicolson scheme, where $\theta = 0.5$.
#
# To simplify we will solve for velocity and pressure, $\vec{u}$ and
# $p$, in a separate step before solving for the new temperature
# $T^{n+1}$. Since these weak equations need to hold for all test
# functions $\vec{v}\in V$ and $w\in W$ we can equivalently write,
# using a single residual functional $F_{\text{Stokes}}$:
#
# $$
#   F_{\text{Stokes}}(\vec{v},w; \vec{u}, p) =
#     \int_\Omega \left(\nabla \vec{v}\right) \colon \mu \left[{\nabla\vec{u}}
#     + \left(\nabla\vec{u}\right)^T\right] dx
#   - \int_\Omega \left(\nabla\cdot \vec{v}\right) p dx \\
#   - \int_\Omega Ra_{0} T\vec{v}\cdot\hat{\vec{k}} dx
#   - \int_\Omega w \nabla \cdot \vec{u} dx = 0
#   \text{ for all } \vec{v}\in V,
#   w\in W,
# $$
#
# where we have multiplied the continuity equation with $-1$ to ensure
# symmetry between the $\nabla p$ and $\nabla\cdot u$ terms. This
# combined weak form that we simultaneously solve for a velocity $u\in
# V$ and pressure $p\in W$ is referred to as a mixed problem, and the
# combined solution $(u, p)$ is said to be found in the mixed function
# space $V\oplus W$.
#
# This example
# ------------
#
# Firedrake provides a complete framework for solving finite element
# problems, highlighted by previous notebooks and herein with the most
# basic mantle dynamics problem - isoviscous, incompressible
# convection, heated from below (T=1), cooled from the top (T=0) in an
# enclosed 2-D Cartesian box (i.e. free-slip mechanical boundary
# conditions on all boudaries), from Blankenbach et al. (1989). Let's
# get started!

from gadopt import *
from mpi4py import MPI

# We have set up the problem using a bilinear quadrilateral element
# pair (Q2-Q1) for velocity and pressure, with Q2 elements for
# temperature. Firedrake user code is written in Python, so the first
# step, illustrated above, is to import the Firedrake module.
#
# We next need a mesh: for simple domains such as the unit square,
# Firedrake provides built-in meshing functions. As such, the
# following code defines the mesh, with 20 quadrilateral elements in x
# and y directions. We also tag boundary IDs.  Boundaries are
# automatically tagged by the built-in meshes supported by
# Firedrake. For the `UnitSquareMesh` being used here, tag 1
# corresponds to the plane $x=0$; 2 to $x=1$; 3 to $y=0$; and 4 to
# $y=1$. We name these `left`, `right`, `bottom` and `top`,
# respectively.

nx, ny = 40, 40
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# We also need function spaces, which is achieved by associating the
# mesh with the relevant finite element: V , W and Q are symbolic
# variables representing function spaces. They also contain the
# function space’s computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis.

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the velocity and pressure spaces to form
# a function space for the mixed Stokes problem, Z.

Z = MixedFunctionSpace([V, W])  # Mixed function space.

# We also specify functions to hold our solutions: z in the mixed
# function space, noting that a symbolic representation of the two
# parts – velocity and pressure – is obtained with `split`.

z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Mantle convection is an initial and boundary-value problem. We
# assume the initial temperature distribution to be prescribed by
#
# $T(x,y) = (1-y) + 0.05\ cos(\pi x)\ sin(\pi y)$
#
# In the following code, we first obtain symbolic expressions for
# coordinates in the physical mesh and subsequently use these to
# initialize the old temperature field.  This is where Firedrake
# transforms a symbolic operation into a numerical computation for the
# first time: the `interpolate` method generates C code that evaluates
# this expression at the nodes of $T$.

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

# We can visualise the initial temperature field using Firedrake's
# built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# The Rayleigh number for this problem is defined in addition to the
# initial timestep $\Delta t$. The viscosity and thermal diffusivity
# are left at their default values (both = 1). We note that viscosity
# could also be a Function, if we wanted spatial variation, and will
# return to this in Part 2 of this notebook below.  These constants
# are used to create an *Approximation* representing the physical
# setup of the problem (options include Boussinesq, Extended
# Boussinesq, Truncated Anelastic Liquid and Anelastic Liquid), and a
# *Timestep Adaptor*, for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time.

# +
delta_t = Constant(1e-6)  # Initial time-step

Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0
steady_state_tolerance = 1e-9
max_timesteps = 20000
kappa = Constant(1.0)  # Thermal diffusivity
# -

# With closed boundaries, and no constraint on pressure anywhere in
# the domain, this problem has a constant pressure nullspace and we
# must ensure that our solver removes this space. To do so, we build a
# nullspace object, which will subsequently be passed to the solver,
# and PETSc will seek a solution in the space orthogonal to the
# provided nullspace.  When building the nullspace object, the
# 'closed' keyword handles the constant pressure nullspace, whilst the
# 'rotational' keyword deals with rotational modes, which, for
# example, manifest in an a annulus domain with free slip top and
# bottom boundaries (as we will see in the next tutorial).

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = VTKFile("output.pvd")
dump_period = 50
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4

# Open file for logging diagnostic output:
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")

gd = GeodynamicalDiagnostics(u, p, T, bottom_id, top_id)
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

# We specify strong Dirichlet boundary conditions for velocity and
# temperature. The user must provide the part of the mesh at which
# each boundary condition applies.  Note how boundary conditions have
# the granularity to be applied to the $x$ and $y$ components of the
# velocity field only, if desired.

# +
temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}
# -

# We finally come to solving the variational problem, with solver
# objects for the energy and Stokes systems created. For the energy
# system we pass in the solution field T, velocity u, the physical
# approximation, time step, temporal discretisation approach
# (i.e. implicit middle point, being equivalent to the Crank Nicolson
# scheme outlined above) and boundary conditions. For the Stokes
# system, we pass in the solution fields z, Temperature, the physical
# approximation, boundary conditions and the nullspace
# object. Solution of the two variational problems is undertaken by
# PETSc.

energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             cartesian=True, constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)

# We can now initiate the time-loop, with the Stokes and energy
# systems solved seperately. These `solve` calls once again convert
# symbolic mathematics into computation. In the time loop, set here to
# run for 500 time-steps, we compute the RMS velocity and surface
# Nusselt number for diagnostic purposes, and print these results
# every 50 timesteps.

# +
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, T)

    dt = t_adapt.update_timestep()
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    bcu = DirichletBC(u.function_space(), 0, top_id)
    ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
    ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
    nusselt_number_top = gd.Nu_top()
    nusselt_number_base = gd.Nu_bottom()
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} {gd.u_rms_top()} {ux_max} "
                 f"{nusselt_number_top} {nusselt_number_base} "
                 f"{energy_conservation} {gd.T_avg()} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        checkpoint_file.save_function(T, name="Temperature", idx=timestep)
        checkpoint_file.save_function(z, name="Stokes", idx=timestep)

plog.close()
checkpoint_file.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
