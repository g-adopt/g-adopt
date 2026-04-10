from gadopt import *

"""
Three-dimensional infultration of water into a heterogeneous soil
=================================================================
Here we present an example of the infultration into a heterogeneous column of soil composed of a mixture of sand and loamy sand, as described by
    Cockett, Heagy, and Haber, Computers and Geosciences, 2018, Efficient 3D inversions using the Richards equation
    https://doi.org/10.1016/j.cageo.2018.04.006
Simulations are formed in a rectangular prism of side length 2.0 x 2.0 x 2.6 m. No flux is imposed on all the boundaries except the top and bottom where $h = -0.1$ and -0.3 m, respectively.
"""


Lx, Ly, Lz = 2, 2, 2.6              # Domain dimensions [m]
refinement_levels = 0               # Number of refinement levels of mesh hierarchy for GMG
factor = 1.26**0                    # Weak-scaling factor (kept at 1.0 here)
nodes = 128/(2**refinement_levels)  # nodes in x/y direction for coarse mesh
nodes_x, nodes_y, nodes_z = round(factor*nodes), round(factor*nodes), round(factor*1.3*nodes)

# --- Mesh construction ---
base_mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, quadrilateral=True)  # Create a 2D quadrilateral base mesh
mh2d = MeshHierarchy(base_mesh, refinement_levels=refinement_levels)     # Build a hierarchy of refined 2D meshes
layer_list = [nodes_z * (2**i) for i in range(refinement_levels + 1)]    # Number of vertical layers at each refinement level
mh3d = ExtrudedMeshHierarchy(mh2d, height=Lz, layers=layer_list)         # Extrude the 2D hierarchy into 3D
mesh = mh3d[-1]

X = SpatialCoordinate(mesh)

dt = Constant(300.0)  # Time step size
t_final = 259200      # 72 hours in seconds

V = FunctionSpace(mesh, "DQ", 0)
PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

# --- Construction of heterogeneous soil properties ---
epsilon = 1/500  # Controls sharpness of material transitions
r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407]  # Random phase shifts for the sinusoidal modes (fixed for reproducibility
I = sin(3*(X[0]-r[0])) + sin(3*(X[1]-r[1])) + sin(3*(X[2]-r[2])) + sin(3*(X[0]-r[3])) + sin(3*(X[1]-r[4])) + sin(3*(X[2]-r[5]))+sin(3*(X[0]-r[6])) + sin(3*(X[1]-r[7])) + sin(3*(X[2]-r[8]))
I = 0.5*(1 + tanh(I/epsilon))  # Smoothly map to [0, 1]

# Specify the hydrological parameters
soil_curve = VanGenuchtenCurve(
    theta_r=0.02*I + 0.035*(1-I),    # Residual water content [-]
    theta_s=0.417*I + 0.401*(1-I),   # Saturated water content [-]
    Ks=5.82e-05*I + 1.69e-05*(1-I),  # Saturated hydraulic conductivity [m/s]
    alpha=13.8*I + 11.5*(1-I),       # Related to inverse of air entry [1/m]
    n=1.592*I + 1.474*(1-I),         # Measure of pore distribution [-]
    Ss=0,                            # Specific storage coefficient [1/m]
)

# -- Set up boundary conditions --
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
moisture_content = soil_curve.moisture_content
h     = Function(V, name="PressureHead").interpolate(bottom_bc - (bottom_bc-top_bc)*exp(5*(X[2]-Lz)))
theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

time_var = Constant(0)
richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=BackwardEuler,
    bcs=richards_bcs,
)

output = VTKFile("cockett.pvd")
output.write(h, theta, time=0)

time = 0
plot_iteration = 0

while time < t_final:

    time_var.assign(time)
    richards_solver.solve()
    time += float(dt)

    plot_iteration += 1
    if plot_iteration % 10 == 0:
        theta.interpolate(moisture_content(h))
        output.write(h, theta, time=time)

    snes = richards_solver.ts.stepper.solver.snes
    nl_it, l_it = snes.getIterationNumber(), 0
    for i in range(snes.getIterationNumber()):
        l_it += snes.getKSP().getIterationNumber()

    PETSc.Sys.Print(f"Timestep number: {plot_iteration} | " 
        f"t = {time/(3600):.2f} h | "
        f"NL iters = {nl_it} | "
        f"L iters = {l_it} | "
        )
