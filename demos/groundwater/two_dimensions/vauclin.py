from gadopt import *

"""
Recharge of a two-dimensional water table
=========================================
Here we reproduce the test case presented in:
    Vauclin, Khanju, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089
The simulation is performed in a domain of 3 x 2 metres, and the initial
condition is chosen such that the region z <= 0.65 m is fully satured ($theta =
theta_s$), $h(t=0) = z - 0.65$. For the boundary conditions, the bottom and left boundary are no flux ($q cdot n = 0$), the right boundary fixed the height of the water table ($h = z - 0.65$ m). For the top boundary, water is injected at a rate of 14.8 cm/hour  in the region where x <= 0.5 m and 0 otherwise. The simulation is concluded after 8 hours
"""

Lx, Ly = 3.00, 2.00  # Domain length [m]
nodes_x, nodes_y = 46, 31
dt = Constant(50)
t_final = 28800

# Create rectangular mesh
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
X = SpatialCoordinate(mesh)

polynomial_degree = 2
V = FunctionSpace(mesh, "DQ", polynomial_degree)
W = VectorFunctionSpace(mesh, 'DQ', polynomial_degree)

soil_curves = HaverkampCurve(
    theta_r=0.01,   # Residual water content [-]
    theta_s=0.37,   # Saturated water content [-]
    Ks=9.722e-05,   # Saturated hydraulic conductivity [m/s]
    alpha=0.44,     # Fitting parameter [m]
    beta=1.2924,    # Fitting parameter [-]
    A=0.0104,       # Fitting parameter [m]
    gamma=1.5722,   # Fitting parameter [-]
    Ss=0.00,        # Specific storage coefficient [1/m]
)

moisture_content      = soil_curves.moisture_content
relative_conductivity = soil_curves.relative_conductivity

h     = Function(V, name="PressureHead").interpolate(0.65 - X[1])
h_old = Function(V, name="PreviousSolution").interpolate(0.65 - X[1])
theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))
q     = Function(W, name='VolumetricFlux')
K     = Function(V, name='RelativeConductivity').interpolate(relative_conductivity(h))

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
    boundary_ids.right: {'h': 0.65 - X[1]},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': top_flux},
}

eq = RichardsEquation(V=V,
                    soil_curves=soil_curves,
                    bcs=richards_bcs,
                    )
richards_solver = RichardsSolver(h, h_old, time_var, dt, eq)

output = VTKFile("vauclin.pvd")
output.write(h, theta, q, time=0)

time = 0

plot_iteration = 0

while time < t_final:

    h_old.assign(h)
    time_var.assign(time)
    richards_solver.solve()
    time += float(dt)

    plot_iteration += 1
    if plot_iteration % 25 == 0:
        theta.interpolate(moisture_content(h))
        K.interpolate(relative_conductivity(h))
        q.interpolate(-K*grad(h + X[1]))

        output.write(h, theta, q, time=time)

