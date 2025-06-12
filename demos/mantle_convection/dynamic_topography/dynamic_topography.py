# Tutorial: Computing Normal Stresses and Dynamic Topography in G-ADOPT
# ====================================================================
#
# One of the most commonly studied geodynamic observables is **dynamic topography** —
# surface or lithospheric deflection caused by vertical stresses from mantle flow, with
# regions being pushed up or pulled down due to underlying mantle convection.
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

# Theory Refresher
# ----------------
# Dynamic topography arises from the internal stress field deforming a surface.
# Consider the top boundary of an annular domain located at radius $r_{max}$.
# Under equilibrium, the vertical (radial) stress acting on this boundary is $\sigma_{rr}$.
# Due to internal forces, the surface deforms by an amount $\delta h$, which can be calculated as follows:
#
# $$\delta h = \sigma_{rr} / (\delta \rho g)$$
#
# Where:
# - $\sigma_{rr}$ is the normal stress at the boundary,
# - $\delta \rho$ is the density difference between mantle and the overlying medium (air or water),
# - $g \approx 9.8 m/s^2$ is gravitational acceleration.
#

#
# Implementation in G-ADOPT
# -------------------------
#
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.
# We also import pyvista and matplotlib, which are used for plotting purposes.

from gadopt import *
# + tags=["active-ipynb"]
# import pyvista as pv
# import matplotlib.pyplot as plt
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
# as with our previous tutorials (using a Q2-Q1 element pair for velocity and pressure).

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
# VTKFile("mesh.pvd").write(Function(V))
# mesh_data = pv.read("mesh/mesh_0.vtu")
# edges = mesh_data.extract_all_edges()
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(edges, color="black")
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# We can also plot our initial temperature field:

# + tags=["active-ipynb"]
# VTKFile("temp.pvd").write(T)
# temp_data = pv.read("temp/temp_0.vtu")
# plotter = pv.Plotter(notebook=True)
# plotter.add_mesh(temp_data)
# plotter.camera_position = "xy"
# plotter.show(jupyter_backend="static", interactive=False)
# -

# We next specify the important constants for this problem, and set up the approximation.
# Note that this case is time independent and hence, when compared to most of our previous
# tutorials, no timestepping options are specified.

Ra = Constant(1e7)
approximation = BoussinesqApproximation(Ra)

# As noted previously, with a free-slip boundary condition on both boundaries, one can add an arbitrary rotation
# of the form $(-y, x)=r\hat{\mathbf{\theta}}$ to the velocity solution (i.e. this case incorporates a velocity nullspace,
# as well as a pressure nullspace). These lead to null-modes (eigenvectors) for the linear system, resulting in a singular matrix.
# In preconditioned Krylov methods these null-modes must be subtracted from the approximate solution at every iteration. We do that below,
# setting up a nullspace object as we did in the previous tutorial, albeit speciying the `rotational` keyword argument to be True.
# This removes the requirement for a user to configure these options, further simplifying the task of setting up a (valid) geodynamical simulation.

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
)

stokes_solver.solve()
# -

# At this point, we have a global solution for velocity and pressure. This is next used to
# compute normal stresses at both top and bottom boundaries, via the <i>force_on_boundary</i> function.

ns_top = stokes_solver.force_on_boundary(boundaries.top)
ns_bottom = stokes_solver.force_on_boundary(boundaries.bottom)

# With these normal stresses, we can now compute dynamic topography at the surface (assuming air loading),
# and at the CMB (assuming a non-dimensional core density of XXX). Outward normals point in opposite directions at these boundaries.

# SIA ADD CALCULATIONS HERE.

# We next setup our output for visualisation of results:

output_file = VTKFile("output.pvd")
output_file.write(u, p, T, ns_top, ns_bottom)

# And plot up those results.

# + tags=["active-ipynb"]
# tripcolor(ns_top)
# plt.title("Normal Stress at Top Boundary")
# plt.colorbar(label="Stress (Pa)")
# plt.show()
# -

# This tutorial has demonstrated how to:
# - Load fields and meshes from checkpoint files.
# - Solve the Stokes system using G-ADOPT.
# - Compute normal stresses on top and bottom boundaries.
# - Estimate dynamic topography from those normal stresses.
# - Save and visualize the results.
