# Tutorial: Computing Normal Stresses and Dynamic Topography in G-ADOPT
# ====================================================================
#
# One of the most commonly studied geodynamic observables is **dynamic topography**
# surface or lithospheric deflection caused by vertical stresses from mantle flow, with regions being pushed
# up or pulled down due to underlying mantle convection.
# This "dynamic" topography is transient, and differs to isostatic topography (like
# mountain-building from crustal thickening).
#
# This tutorial demonstrates how to compute **normal stresses acting on a boundary**
# and subsequent dynamic topohraphy using **G-ADOPT**.
#
# Specifically, we will compute the radial stress $\sigma_{rr}$ on the boundaries of a
# 2-D annular domain, and use them to calculate dynamic topography. We examine a time-independent
# simulation with free-slip boundary conditions, where the internal structure is loaded
# from a checkpoint file from a previous 2-D annulus case. Note that given the lack of time-dependence,
# we do not solve an energy equation, and deal with the Stokes system only.
#
# Theory Refresher
# ----------------
# Dynamic topography arises from the internal stress field deforming a surface.
# Consider the top boundary of an annular domain located at radius $r_{max}$.
# Under equilibrium, the vertical (radial) stress acting on this boundary is $\sigma_{rr}$.
# Due to internal forces, the surface deforms by an amount $\delta h$, which can be calculated as follows:
#
# $$\delta h = \sigma_{rr} / (\delta \rho\,g)$$
#
# Where:
#   - $\sigma_{rr}$ is the normal stress at the boundary,
#   - $\delta \rho$ is the density difference between mantle and the overlying medium (air or water),
#   - $g \approx 9.8 m/s^2$ is gravitational acceleration.
#

#
# Implementation in G-ADOPT
# -------------------------
#
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.
# We also import pyvista and matplotlib, which are used for plotting purposes.

# +
from gadopt import *
# -

# We also import pyvista for doing file visualisations for this notebook.

# + tags=["active-ipynb"]
# import pyvista as pv
# -

# We next load the mesh from a checkpoint file and initialise the temperature field
# noting that in this tutorial the temperature field is used only through the fixed buoyancy term on the
# RHS of the Stokes equation. Cartesian flags and boundary IDs are collected in a way that is
# consistent with our previous tutorials.

# +
with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

mesh.cartesian = False
boundaries = get_boundary_ids(mesh)
# -

# We next set up function spaces, and specify functions to hold our solutions,
# as with our previous tutorials.

# +
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, W])

z = Function(Z)
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
# -

# We can now visualise the mesh.

# + tags=["active-ipynb"]
# VTKFile("temperature.pvd").write(T)
# temp_data = pv.read("temperature/temperature_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data)
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# We next specify the constants and physical approximation for this problem.
# Note that this case is time independent and hence, when compared to most of our previous
# tutorials, no timestepping options are specified.

Ra = Constant(1e7)
approximation = BoussinesqApproximation(Ra)

# As noted previously, with a free-slip boundary condition on both boundaries, one can add an arbitrary rotation
# of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity solution (i.e. this case incorporates a velocity nullspace,
# as well as a pressure nullspace). These lead to null-modes (eigenvectors) for the linear system, resulting in a singular matrix.
# In preconditioned Krylov methods these null-modes must be subtracted from the approximate solution at every iteration. We do that below,
# setting up a nullspace object as we did in the previous tutorial.

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

# Given the increased computational expense (typically requiring more degrees of freedom) in a 2-D annulus domain, G-ADOPT defaults to iterative
# solver parameters. G-ADOPT's iterative solver setup is configured to use the GAMG preconditioner
# for the velocity block of the Stokes system, to which we must provide near-nullspace information,
# which, in 2-D, consists of two rotational and two translational modes.

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Boundary conditions are next specified. For velocity, we specify free‐slip conditions on both boundaries. As noted in our
# 2-D cylindrical tutorial, we incorporate these <b>weakly</b> through the <i>Nitsche</i> approximation. Given we do not solve
# an energy equation for this time independent case, no boundary conditions are required for temperature.

stokes_bcs = {
    "bottom": {"un": 0},
    "top": {"un": 0},
}

# We can now setup and solve the variational problem for the Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.

# +
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters="direct",
)

stokes_solver.solve()
# -

# At this point, we have a global solution for velocity and pressure. Let's visualise them:

# + tags=["active-ipynb"]
# VTKFile("velocity.pvd").write(z.subfunctions[0])
# u_data = pv.read("velocity/velocity_0.vtu").glyph(factor=1e-5, orient="Velocity", tolerance=1e-2)
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data)
# plotter.add_mesh(u_data, color="black")
# plotter.camera_position = "xy"
# plotter.camera.zoom(1.5)
# plotter.show(jupyter_backend="static", interactive=False)
# -

# Next, the global solutions is used to compute normal stresses at both
# top and bottom boundaries, via the `force_on_boundary` method.

# +
ns_top = stokes_solver.force_on_boundary(boundaries.top)
ns_bottom = stokes_solver.force_on_boundary(boundaries.bottom)
# -

# With these normal stresses, we can now calculate the dynamic topography at both the surface and the core-mantle boundary (CMB) using:
#
# $$\delta_{h} = \frac{\sigma_{rr}}{(\Delta \rho * g)}$$
#
# where:
#   - $\sigma_{rr}$ is the normal stress at the boundary,
#   - $\Delta \rho$ is the density difference across the boundary,
#   - $g$ is gravitational acceleration.
#
# Since the Stokes system in G-ADOPT is non-dimensionalized, we use non-dimensional values for density and gravity at the top boundary (both set to Constant(1.0)).
# For the CMB, we assume the outer core is 2.5 times denser than the mantle (\Delta \rho_CMB = Constant(2.5)), and set gravity to Constant(1.0).
# To further normalize, you can multiply the non-dimensional normal stress by the mantle depth,  $ L = 2.89 \times 10^6 $  m.

# +
# Create Function objects for dynamic topography
dimensionalisation_factor = Constant(3e-5 * 4e3 * 3e6) / Ra
dynamic_topography_top = Function(W, name="Dynamic_Topography_Top")
dynamic_topography_bottom = Function(W, name="Dynamic_Topography_Bottom")

# Compute dynamic topography values
delta_rho_top = Constant(2800 / 4e3)
g_top = Constant(1.0)
dynamic_topography_top.interpolate(ns_top / (delta_rho_top * g_top) * dimensionalisation_factor)

delta_rho_cmb = Constant(-2.5)
g_cmb = Constant(1.0)
dynamic_topography_bottom.assign(ns_bottom / (delta_rho_cmb * g_cmb) * dimensionalisation_factor)
# -

# Now it's time to visualise our cool stuff:

# + tags=["active-ipynb"]
# VTKFile("dt.pvd").write(dynamic_topography_top, dynamic_topography_bottom)
# # Loading dynamic topography calculations
# dt_data = pv.read("./dt/dt_0.vtu")
# # Scaling of the dynamic topography fields so we can see temperature and velocity fields
# # Scale top boundary layer by  1.3 and bottom boundary layer by 0.7
# import numpy as np
# transform_top = np.array([[1.3, 0, 0], [0, 1.3, 0], [0, 0, 1]])
# transform_bottom = np.array([[0.7, 0, 0], [0, 0.7, 0], [0, 0, 1]])
#
# # Do complicated stuff to extract nice topography surfaces for the mesh
# dt_data.compute_implicit_distance(pv.Circle(2.21), inplace=True)
# outter = dt_data.threshold(0.0, scalars='implicit_distance', invert=True).transform(transform_top, inplace=True)
# warped_outter = outter.warp_by_scalar("Dynamic_Topography_Top", factor=1e-3)
# dt_data.compute_implicit_distance(pv.Circle(1.23), inplace=True)
# inner = dt_data.threshold(0.0, scalars='implicit_distance', invert=False).transform(transform_bottom, inplace=True)
# warped_inner = inner.warp_by_scalar("Dynamic_Topography_Bottom", factor=2e-3)
#
# # Plot those nice surfaces
# plotter = pv.Plotter(notebook=True)
# # Assuming a 4000 temperature difference across the mantle for dimensionalisation
# t_plot = plotter.add_mesh(temp_data, scalars="Temperature", cmap="coolwarm", clim=[0, 1], show_scalar_bar=False)
# plotter.add_mesh(u_data, color="black")
# dt_plot = plotter.add_mesh(warped_outter, scalars="Dynamic_Topography_Top", cmap="bwr", show_scalar_bar=False)
# plotter.add_mesh(warped_inner, scalars="Dynamic_Topography_Bottom", clim=[-1e4, 1e4], cmap="bwr", show_scalar_bar=False)
# plotter.camera_position = "xy"
# plotter.camera.zoom(6)
# _ = plotter.add_scalar_bar('Temperature []', mapper=t_plot.mapper, position_x=0.02, position_y=0.5, vertical=True)
# _ = plotter.add_scalar_bar( 'Topography [m]', mapper=dt_plot.mapper, position_x=0.85, position_y=0.5, vertical=True)
# plotter.show(jupyter_backend="static", interactive=False)
# -

# This tutorial has demonstrated how to:
# - Load fields and meshes from checkpoint files.
# - Solve the Stokes system using G-ADOPT.
# - Compute normal stresses on top and bottom boundaries.
# - Estimate dynamic topography from those normal stresses.
# - Save and visualize the results.
