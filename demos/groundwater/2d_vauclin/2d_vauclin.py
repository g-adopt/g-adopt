from gadopt import *
import gwassess


r"""
Recharge of a two-dimensional water table
==========================================
Here we reproduce the benchmark presented in:
    Vauclin, Khanji, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional
    unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089

The simulation is performed in a domain of 3 x 2 metres. The initial condition
is chosen such that the region z <= 0.65 m is fully saturated ($\theta =
\theta_s$), with pressure head $h(t=0) = z - 0.65$. For the boundary
conditions, the bottom and left boundaries are no flux ($q \cdot n = 0$), the
right boundary fixes the height of the water table ($h = z - 0.65$ m). For the
top boundary, water is injected at a rate of 14.8 cm/hour in the region where
$x \leq 0.5$ m and 0 otherwise. The simulation runs for 8 hours.

The soil hydraulic properties follow the Haverkamp model with parameters from the
original paper (Eq. 1 and 9): $\theta_s = 0.30$, $\alpha = 40{,}000$,
$\beta = 2.90$, $A = 2.99 \times 10^6$, $\gamma = 5.0$, $K_s = 35$ cm/hr
(all converted to SI units via gwassess).
"""

# Initialise reference solution from gwassess for domain and soil parameters
vauclin_solution = gwassess.VauclinRichardsSolution2D()

Lx = vauclin_solution.Lx
Ly = vauclin_solution.Ly

nodes_x, nodes_y = 46, 31
dt = Constant(25)
t_final = vauclin_solution.SIMULATION_DURATION  # 8 hours

# Create rectangular mesh with quadrilateral elements
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
mesh.cartesian = True
X = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DQ", 2)
W = VectorFunctionSpace(mesh, "DQ", 2)

# Get soil parameters from gwassess (SI units, converted from the paper's CGS)
soil_params = vauclin_solution.get_soil_parameters()

soil_curve = HaverkampCurve(
    theta_r=soil_params['theta_r'],
    theta_s=soil_params['theta_s'],
    Ks=soil_params['Ks'],
    Ss=soil_params['Ss'],
    alpha=soil_params['alpha'],
    beta=soil_params['beta'],
    A=soil_params['A'],
    gamma=soil_params['gamma'],
)

moisture_content = soil_curve.moisture_content
relative_permeability = soil_curve.relative_permeability

# Initial condition: water table at z = 0.65 m from bottom
water_table_height = vauclin_solution.WATER_TABLE_HEIGHT
h_ic = Function(V, name="InitialCondition").interpolate(
    water_table_height - 1.001 * X[1]
)
h = Function(V, name="PressureHead").interpolate(h_ic)
h_old = Function(V, name="PreviousSolution").interpolate(h_ic)

theta = Function(V, name="MoistureContent").interpolate(moisture_content(h))
q = Function(W, name="VolumetricFlux")

# Time-dependent top boundary flux with tanh smoothing
time_var = Constant(0.0)
infiltration_rate = vauclin_solution.INFILTRATION_RATE
infiltration_width = vauclin_solution.INFILTRATION_WIDTH

top_flux = tanh(0.000125 * time_var) * infiltration_rate * (
    0.5 * (1 + tanh(10 * (X[0] + infiltration_width)))
    - 0.5 * (1 + tanh(10 * (X[0] - infiltration_width)))
)

# Boundary conditions from the paper:
# Left: no flux, Right: fixed water table, Bottom: no flux, Top: infiltration
boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    boundary_ids.left: {'flux': 0.0},
    boundary_ids.right: {'h': h_ic},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': top_flux},
}

# DIRK22 is 2nd-order and stiffly accurate, giving mass conservation
# with stage_type="value" (the default for RichardsSolver).
richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=DIRK22,
    bcs=richards_bcs,
    solver_parameters="direct",
    quad_degree=5,
)

# Quadrature measures for mass balance tracking
ds_mesh = Measure("ds", domain=mesh, metadata={"quadrature_degree": 5})
dx_mesh = Measure("dx", domain=mesh, metadata={"quadrature_degree": 5})

initial_mass = assemble(moisture_content(h) * dx_mesh)
external_flux = 0

# Logging
plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt min_h max_h mass_balance")

output = VTKFile("vauclin.pvd")
output.write(h, theta, q, time=0)

time = 0
timestep_count = 0

while time < t_final:
    h_old.assign(h)
    time_var.assign(time)
    richards_solver.solve()
    time += float(dt)
    timestep_count += 1

    # Compute flux diagnostics for mass balance
    K = relative_permeability((h + h_old) / 2)
    q.interpolate(-K * grad((h + h_old) / 2 + X[1]))
    external_flux += assemble(float(dt) * dot(q, -FacetNormal(mesh)) * ds_mesh)

    # Mass balance ratio: (mass gained) / (flux through boundaries)
    current_mass = assemble(moisture_content(h) * dx_mesh)
    mass_balance = (current_mass - initial_mass) / external_flux if external_flux != 0 else 0

    min_h = h.dat.data.min()
    max_h = h.dat.data.max()

    plog.log_str(
        f"{timestep_count} {time} {float(dt)} {min_h} {max_h} {mass_balance}"
    )

    if timestep_count % 50 == 0:
        theta.interpolate(moisture_content(h))
        output.write(h, theta, q, time=time)

plog.close()

PETSc.Sys.Print(f"Simulation complete: {timestep_count} timesteps")
PETSc.Sys.Print(f"Final min_h={min_h:.4f}, max_h={max_h:.4f}, mass_balance={mass_balance:.6f}")
