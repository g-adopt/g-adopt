# Tutorial: Computing Normal Stresses and Dynamic Topography in G-ADOPT
# ====================================================================
# Overview
# --------
# One of the most commonly studied geodynamic observables is **dynamic topography** —
# the surface deformation caused by internal stresses within Earth's mantle.
# This tutorial demonstrates how to compute **normal stresses acting on a boundary**
# using **G-ADOPT**.
# Specifically, we will compute the radial stress $\sigma_{rr}$ on a boundary and
# use it to estimate dynamic topography.

# Theory Refresher
# ----------------
# Dynamic topography arises from the internal stress field deforming a surface.
# Consider the top boundary of an annular domain located at radius $r_{max}$.
# Under equilibrium, the vertical (radial) stress acting on this boundary is $\sigma_{rr}(r_max)$.
# Due to internal forces, the surface deforms by an amount $\delta h$, leading to:
#
# $$\sigma_{rr}(r_max) = \sigma_{rr}(r_max + \delta h) - \rho g\, \delta h$$
#
# Solving for $\sigma h$:
#
# $$\delta h = \sigma_{rr} / (\rho g)$$
#
# Where:
# - $\sigma_{rr}$ is the normal stress at the boundary,
# - $\rho$ is the density difference between the mantle and the overlying medium (air or water),
# - $g \approx 9.8 m/s^2$ is gravitational acceleration.

# Implementation in G-ADOPT
# -------------------------

# Step 1: Load Mesh and Temperature Field
from gadopt import *

with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")
    T = f.load_function(mesh, "Temperature")

mesh.cartesian = False
boundaries = get_boundary_ids(mesh)

# Step 2: Set Up Function Spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity
W = FunctionSpace(mesh, "CG", 1)        # Pressure
Z = MixedFunctionSpace([V, W])         # Combined space

z = Function(Z)
u, p = split(z)

Ra = Constant(1e7)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Step 3: Define Boundary Conditions and Solve Stokes Flow
stokes_bcs = {
    "bottom": {"un": 0},
    "top": {"un": 0},
}

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
)

u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")

stokes_solver.solve()

# Step 4: Compute Normal Stresses at Boundaries
ns_top = stokes_solver.force_on_boundary(boundaries.top)
ns_bottom = stokes_solver.force_on_boundary(boundaries.bottom)

# Saving and Visualizing the Results
# ----------------------------------

# Save to File for Visualization
output_file = VTKFile("output.pvd")
output_file.write(u, p, T, ns_top, ns_bottom)

# Plotting Example (in Python, optional)
# Plotting using matplotlib (optional, only if you have matplotlib)
# import matplotlib.pyplot as plt
# tripcolor(ns_top)
# plt.title("Normal Stress at Top Boundary")
# plt.colorbar(label="Stress (Pa)")
# plt.show()

# Exercise 1: Compute Dynamic Topography
# -------------------------------------
# Assume:
# - $\delta \rho = 3300 kg/m^3$
# - $g = 9.8 m/s^2$
# Compute $\delta h = \sigma_{rr} / (\rho g)$ at the top boundary

rho = 3300
g = 9.8

# Compute $delta h$
delta_h = Function(W, name="Dynamic Topography")
delta_h.interpolate(ns_top / (rho * g))

# Save for visualization
output_file.write(delta_h)

# Exercise 2: Modify Boundary Conditions
# -------------------------------------
# Change the free-slip condition to no-slip on the bottom boundary:
# stokes_bcs = {
#     "bottom": {"u": as_vector([0, 0])},
#     "top": {"un": 0},
# }
# Re-run the solver and observe how σ_rr changes.

# Exercise 3: Vary Rayleigh Number
# --------------------------------
# Try different values of Rayleigh number:
# Ra = 1e5, 1e6, 1e8
# Observe how the stress field and dynamic topography vary with Ra.

# Summary
# -------
# This tutorial showed how to:
# - Load fields and mesh from checkpoint files,
# - Solve the Stokes system using G-ADOPT,
# - Compute normal stresses on boundaries,
# - Estimate dynamic topography from stress,
# - Save and visualize the results.

# Dynamic topography is a critical observable in geodynamics
# and computing it accurately requires careful stress analysis —
# G-ADOPT provides all the tools necessary to perform such analyses efficiently.
