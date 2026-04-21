# Three-dimensional infiltration into heterogeneous soil
# =====================================================
#
# This demo reproduces the 3D benchmark from Cockett, Heagy, and Haber
# (Computers & Geosciences, 2018, doi:10.1016/j.cageo.2018.04.006).
# The problem simulates infiltration of water into a block of soil
# composed of two materials — sand and loamy sand — arranged in a
# spatially complex heterogeneous pattern.
#
# The domain is a rectangular prism of 2 x 2 x 2.6 m, built as an
# extruded quadrilateral mesh. The top boundary is held at h = -0.1 m,
# the bottom at h = -0.3 m, and all four sides are no-flux. The initial
# condition is an exponential profile that satisfies both boundary values:
#
# $$h(x, y, z, 0) = 0.2 \exp\bigl(5(z - 2.6)\bigr) - 0.3$$
#
# Water retention and hydraulic conductivity follow the van Genuchten
# model. The two soil types are blended through a smooth indicator field
# constructed from a sum of sinusoidal functions with a sharp tanh
# transition. Sand has higher saturated conductivity ($K_s = 5.82 \times
# 10^{-5}$ m/s) than loamy sand ($K_s = 1.69 \times 10^{-5}$ m/s),
# producing visibly faster infiltration in sand regions.
#
# The simulation runs for 24 hours with a time step of 600 seconds using
# the DIRK22 scheme, which is second-order accurate and stiffly accurate,
# ensuring mass conservation when used with the default stage value
# discretisation in RichardsSolver.

from gadopt import *

# Set up the extruded mesh: a 20x20 quadrilateral base mesh extruded
# vertically into 26 layers.

Lx, Ly, Lz = 2.0, 2.0, 2.6
nx, ny, nz = 20, 20, 26

mesh2d = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(mesh2d, nz, layer_height=Lz / nz)
X = SpatialCoordinate(mesh)

# We use a discontinuous Galerkin space of polynomial degree 0 (piecewise
# constant) on hexahedral elements, giving 10,400 degrees of freedom.

V = FunctionSpace(mesh, "DQ", 0)
log(f"Number of degrees of freedom: {V.dim()}")

# The heterogeneous soil field is constructed by summing nine sinusoidal
# functions with fixed offsets, then applying a sharp tanh transition.
# The result is an indicator field I that is close to 1 in sand regions
# and close to 0 in loamy sand regions.

r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837,
     0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
epsilon = 1 / 500

indicator = (sin(3 * (X[0] - r[0])) + sin(3 * (X[1] - r[1]))
             + sin(3 * (X[2] - r[2])) + sin(3 * (X[0] - r[3]))
             + sin(3 * (X[1] - r[4])) + sin(3 * (X[2] - r[5]))
             + sin(3 * (X[0] - r[6])) + sin(3 * (X[1] - r[7]))
             + sin(3 * (X[2] - r[8])))
indicator = 0.5 * (1 + tanh(indicator / epsilon))

# The van Genuchten soil parameters are interpolated between the two
# materials using the indicator field. Parameters are from standard
# tabulated values for sand and loamy sand.

soil_curve = VanGenuchtenCurve(
    theta_r=0.02 * indicator + 0.035 * (1 - indicator),
    theta_s=0.417 * indicator + 0.401 * (1 - indicator),
    Ks=5.82e-05 * indicator + 1.69e-05 * (1 - indicator),
    alpha=13.8 * indicator + 11.5 * (1 - indicator),
    n=1.592 * indicator + 1.474 * (1 - indicator),
    Ss=0,
)

# Boundary conditions: fixed pressure head on top and bottom, no-flux
# on the four sides.

boundary_ids = get_boundary_ids(mesh)
top_bc, bottom_bc = -0.1, -0.3

richards_bcs = {
    boundary_ids.left: {"flux": 0},
    boundary_ids.right: {"flux": 0},
    boundary_ids.back: {"flux": 0},
    boundary_ids.front: {"flux": 0},
    boundary_ids.bottom: {"h": bottom_bc},
    boundary_ids.top: {"h": top_bc},
}

# The initial condition smoothly transitions from the bottom boundary
# value to the top boundary value via an exponential profile.

h = Function(V, name="PressureHead")
h.interpolate(0.2 * exp(5 * (X[2] - Lz)) - 0.3)

theta = Function(V, name="MoistureContent")
theta.interpolate(soil_curve.moisture_content(h))

# We use DIRK22 (a second-order, stiffly accurate DIRK scheme) for time
# integration. With the default stage_type="value" in RichardsSolver,
# this gives mass conservation to solver tolerance. We use a direct
# solver since the problem size (162,500 DOFs) is manageable.

dt = Constant(600.0)
t_final = 24 * 3600  # 24 hours in seconds

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=DIRK22,
    bcs=richards_bcs,
    solver_parameters="direct",
)

# Set up output: VTK files for visualisation and a parameter log for
# diagnostics, following the same pattern as the Vauclin 2D demo.

output = VTKFile("3d_cockett.pvd")
output.write(h, theta, time=0)

plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt min_h max_h")

# Time loop: 144 timesteps over 24 hours.

time = 0.0
timestep_count = 0

while time < t_final:
    richards_solver.solve()
    time += float(dt)
    timestep_count += 1

    min_h = h.dat.data.min()
    max_h = h.dat.data.max()

    plog.log_str(f"{timestep_count} {time} {float(dt)} {min_h} {max_h}")

    if timestep_count % 50 == 0:
        theta.interpolate(soil_curve.moisture_content(h))
        output.write(h, theta, time=time)

# Write the final state and close the log.

theta.interpolate(soil_curve.moisture_content(h))
output.write(h, theta, time=time)
plog.close()

PETSc.Sys.Print(f"Simulation complete: {timestep_count} timesteps")
PETSc.Sys.Print(f"Final min_h={min_h:.4f}, max_h={max_h:.4f}")
