from gadopt import *

"""
Three-dimensional infultration of water into a heterogeneous soil
=================================================================
Here we present an example of the infultration into a heterogeneous column of soil composed of a mixture of sand and loamy sand, as described by
    Cockett, Heagy, and Haber, Computers and Geosciences, 2018, Efficient 3D inversions using the Richards equation
    https://doi.org/10.1016/j.cageo.2018.04.006
Simulations are formed in a rectangular prism of side length 2.0 x 2.0 x 2.6 m. No flux is imposed on all the boundaries except the top where $h = -0.1$ m.
"""

# Set up mesh
nodesX = 20
nodesY, nodesZ = nodesX, round(1.3*nodesX)
Lx, Ly, Lz = 2, 2, 2.6

mesh2D = RectangleMesh(nodesX, nodesY, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(mesh2D, nodesZ, layer_height=Lz/nodesZ)
X = SpatialCoordinate(mesh)

dt = Constant(300.0)  # Time step size
t_final = float(dt)*250

V = FunctionSpace(mesh, "DQ", 2)

# Construct the heterogeneuous soil
epsilon = 1/500
r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
I = sin(3*(X[0]-r[0])) + sin(3*(X[1]-r[1])) + sin(3*(X[2]-r[2])) + sin(3*(X[0]-r[3])) + sin(3*(X[1]-r[4])) + sin(3*(X[2]-r[5]))+sin(3*(X[0]-r[6])) + sin(3*(X[1]-r[7])) + sin(3*(X[2]-r[8]))
I = 0.5*(1 + tanh(I/epsilon))

# Specify the hydrological parameters
soil_curve = VanGenuchtenCurve(
    theta_r=0.02*I + 0.035*(1-I),    # Residual water content [-]
    theta_s=0.417*I + 0.401*(1-I),   # Saturated water content [-]
    Ks=5.82e-05*I + 1.69e-05*(1-I),  # Saturated hydraulic conductivity [m/s]
    alpha=13.8*I + 11.5*(1-I),       # Related to inverse of air entry [1/m]
    n=1.592*I + 1.474*(1-I),         # Measure of pore distribution [-]
    Ss=0,                            # Specific storage coefficient [1/m]
)

# Set up boundary conditions
boundary_ids = get_boundary_ids(mesh)
top_bc, bottom_bc = -0.1, -0.3
richards_bcs = {
    boundary_ids.left: {'flux': 0},
    boundary_ids.right: {'flux': 0},
    boundary_ids.back: {'flux': 0},
    boundary_ids.front: {'flux': 0},
    boundary_ids.bottom: {'h': bottom_bc},
    boundary_ids.top: {'h': top_bc},
}

# Initial condition
h = Function(V, name="PressureHead").interpolate(bottom_bc - (bottom_bc-top_bc)*exp(5*(X[2]-Lz)))

richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=ImplicitMidpoint,
    bcs=richards_bcs,
    solver_parameters="iterative",
    quad_degree=5,
)

time = 0
while time < t_final:

    richards_solver.solve()
    time += float(dt)