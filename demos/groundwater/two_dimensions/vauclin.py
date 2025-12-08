from gadopt import *
import ufl

"""
Recharge of a two-dimensional water table
=========================================
Here we reproduce the test case presented in:
    Vauclin, Khanju, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089
The simulation is performed in a domain of 3 x 2 metres, and the initial
condition is chosen such that the region z <= 0.65 m is fully satured ($\theta =
\theta_s$), $h(t=0) = z - 0.65$. For the boundary conditions, the bottom and left boundary are no flux ($q \cdot n = 0$), the right boundary fixed the height of the water table ($h = z - 0.65$ m). For the top boundary, water is injected at a rate of 14.8 cm/hour  in the region where x <= 0.5 m and 0 otherwise. The simulation is concluded after 8 hours
"""

Lx, Ly = 300, 200  # Domain length [cm]
nodes_x, nodes_y = 181, 121

# Create rectangular mesh
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=False)
X = SpatialCoordinate(mesh)

# Steady state water table initial condition
with CheckpointFile("vauclin_ic.h5", 'r') as afile:
    mesh_ic = afile.load_mesh("mesh_ic")
    h_ic = afile.load_function(mesh_ic, "InitialCondition")

V = FunctionSpace(mesh, "DG", 1)
W = VectorFunctionSpace(mesh, 'DG', 2)

soil_curve = HaverkampCurve(
    theta_r=0.00,   # Residual water content [-]
    theta_s=0.37,   # Saturated water content [-]
    Ks=9.722e-03,   # Saturated hydraulic conductivity [cm/s]
    alpha=40000,    # Fitting parameter [cm]
    beta=2.90,      # Fitting parameter [-]
    A=2.99e06,      # Fitting parameter [cm]
    gamma=5.0,      # Fitting parameter [-]
    Ss=0.00,       # Specific storage coefficient [1/cm]
)

moisture_content = soil_curve.moisture_content
relative_permeability = soil_curve.relative_permeability

h_ic = Function(V, name="InitialCondition").interpolate(h_ic)
h = Function(V, name="PressureHead").interpolate(h_ic)
theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))
q = Function(W, name='VolumetricFlux')
K = Function(V, name='RelativeConductivity').interpolate(relative_permeability(h))

# Set up boundary conditions
time_var = Constant(0.0)
top_flux = tanh(0.000125 * time_var) * 4.11e-03 * (
    0.5 * (1 + tanh(25 * (X[0] + 50)))
    - 0.5 * (1 + tanh(25 * (X[0] - 50)))
)

# Boundary conditions
boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    boundary_ids.left: {'flux': 0.0},
    boundary_ids.right: {'h': h_ic},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': -top_flux},
}

time = 0
dt = Constant(200)
t_final = 32400

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=BackwardEuler,
    bcs=richards_bcs,
    solver_parameters="direct",
    quad_degree=5,
)

output = VTKFile("vauclin.pvd")
output.write(h, theta, q)

external_flux = 0
initial_mass = assemble(theta*dx)

while time < t_final:

    time_var.assign(time)
    richards_solver.solve()
    time += float(dt)

    theta.interpolate(moisture_content(h))
    K.interpolate(relative_permeability(h))
    q.interpolate(-K*grad(h + X[1]))

    n = FacetNormal(mesh)
    external_flux += assemble(float(dt)*dot(q, -n)*ds)

    output.write(h, theta, q)

final_mass = assemble(theta*dx)
print(f"External flux: {external_flux}")
print(f"Initiall mass: {initial_mass}")
print(f"Final mass: {final_mass}")
print(f"Mass balance: {(final_mass-initial_mass)/external_flux}")
