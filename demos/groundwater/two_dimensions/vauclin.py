from gadopt import *


r"""
Recharge of a two-dimensional water table
=========================================
Here we reproduce the test case presented in:
    Vauclin, Khanju, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089

The simulation is performed in a domain of 300 x 200 cm, and the initial
condition is chosen such that the region z <= 65 cm is fully saturated ($\theta =
\theta_s$), $h(t=0) = z - 65$. For the boundary conditions, the bottom and left 
boundary are no flux ($q \cdot n = 0$), the right boundary fixed the height of the 
water table ($h = z - 65$ cm). For the top boundary, water is injected at a rate 
of 14.8 cm/hour in the region where x <= 0.5 m and 0 otherwise. The simulation 
is concluded after 8 hours.
"""

Lx, Ly = 300, 200  # Domain length [cm]
nodes_x, nodes_y = 46, 31
dt = Constant(50)
t_final = 28800  # In seconds

# Create rectangular mesh
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
X = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DQ", 2)
W = VectorFunctionSpace(mesh, 'DQ', 2)

soil_curve = HaverkampCurve(
    theta_r=0.00,   # Residual water content [-]
    theta_s=0.30,   # Saturated water content [-]
    Ks=9.722e-03,   # Saturated hydraulic conductivity [cm/s]
    alpha=40000,    # Fitting parameter [cm]
    beta=2.90,      # Fitting parameter [-]
    A=2.99e6,       # Fitting parameter [cm]
    gamma=5.00,     # Fitting parameter [-]
    Ss=0.00,        # Specific storage coefficient [1/cm]
)

moisture_content = soil_curve.moisture_content
relative_conductivity = soil_curve.relative_conductivity

h_ic  = Function(V, name="InitialCondition").interpolate(65 - X[1])
h     = Function(V, name="PressureHead").interpolate(h_ic)
h_old = Function(V, name="PreviousSolution").interpolate(h_ic)
theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))
q     = Function(W, name='VolumetricFlux')
K     = Function(V, name='RelativeConductivity').interpolate(relative_conductivity(h))

# Set up boundary conditions
time_var = Constant(0.0)
top_flux = tanh(0.000125 * time_var) * 4.11e-03 * (
    0.5 * (1 + tanh(0.1 * (X[0] + 50)))
    - 0.5 * (1 + tanh(0.1 * (X[0] - 50)))
)

# Boundary conditions
boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    boundary_ids.left:   {'flux': 0.0},
    boundary_ids.right:  {'h': h_ic},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top:    {'flux': top_flux},
}

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=BackwardEuler,
    bcs=richards_bcs,
)

time = 0

ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": 5})
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 5})

plot_iteration = 0
output = VTKFile("vauclin.pvd")
output.write(h, theta, q, time=time)

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
