from gadopt import *

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

Lx, Ly = 3.00, 2.00  # Domain length [m]
nodes_x, nodes_y = 91, 61
dt = Constant(10)
t_final = 28800

# Create rectangular mesh
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
X = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DQ", 2)
W = VectorFunctionSpace(mesh, 'DQ', 2)

soil_curve = HaverkampCurve(
    theta_r=0.01,   # Residual water content [-]
    theta_s=0.37,   # Saturated water content [-]
    Ks=9.722e-05,   # Saturated hydraulic conductivity [m/s]
    alpha=0.44,     # Fitting parameter [m]
    beta=1.2924,    # Fitting parameter [-]
    A=0.0104,       # Fitting parameter [m]
    gamma=1.5722,   # Fitting parameter [-]
    Ss=0.00,        # Specific storage coefficient [1/m]
)

moisture_content = soil_curve.moisture_content
relative_permeability = soil_curve.relative_permeability

h_ic = Function(V, name="InitialCondition").interpolate(0.65 - 1.001*X[1])
h = Function(V, name="PressureHead").interpolate(h_ic)
h_old = Function(V, name="PreviousSolution").interpolate(h_ic)
theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))
q = Function(W, name='VolumetricFlux')
K = Function(V, name='RelativeConductivity').interpolate(relative_permeability(h))

# Set up boundary conditions
time_var = Constant(0.0)
top_flux = tanh(0.000125 * time_var) * 4.11e-05 * (
    0.5 * (1 + tanh(10 * (X[0] + .50)))
    - 0.5 * (1 + tanh(10 * (X[0] - .50)))
)

# Boundary conditions
boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    boundary_ids.left: {'flux': 0.0},
    boundary_ids.right: {'h': h_ic},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': top_flux},
}

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=ImplicitMidpoint,
    bcs=richards_bcs,
    solver_parameters="direct",
    quad_degree=5,
)

output = VTKFile("vauclin.pvd")
output.write(h, theta, q)

time = 0
external_flux = 0
initial_mass = assemble(theta*dx)

ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": 5})
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 5})

plot_iteration = 0

while time < t_final:

    h_old.assign(h)
    time_var.assign(time)
    richards_solver.solve()
    time += float(dt)

    theta.interpolate(moisture_content(h))
    K.interpolate(relative_permeability((h+h_old)/2))
    q.interpolate(-K*grad((h+h_old)/2 + X[1]))

    external_flux += assemble(float(dt)*dot(q, -FacetNormal(mesh))*ds)

    plot_iteration += 1
    if plot_iteration % 25 == 0:
        output.write(h, theta, q)

final_mass = assemble(theta*dx)

print(f"External flux: {external_flux}")
print(f"Initiall mass: {initial_mass}")
print(f"Final mass: {final_mass}")
print(f"Mass balance: {(final_mass-initial_mass)/external_flux}")
