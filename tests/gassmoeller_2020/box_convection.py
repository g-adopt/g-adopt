import matplotlib.pyplot as plt
from firedrake import *

from gadopt import ImplicitMidpoint, TimestepAdaptor, get_boundary_ids

from approximations_solvers import Approximation, EnergySolver, StokesSolver


def calculate_diagnostics() -> None:
    grad_T_n = dot(grad(temperature), FacetNormal(mesh))
    heat_flux_top.append(assemble(-apx.ref_profiles["k"] * grad_T_n * ds(boundary.top)))
    diag_time.append(simu_time / year_to_seconds)


def write_output() -> None:
    temperature.project(apx.ref_profiles["T"] + T)
    match apx.name:
        case "BA" | "EBA" | "TALA" | "ALA":
            density.project(apx.density(p, T, 0.0) - apx.ref_profiles["rho"])
        case "ICA" | "HCA" | "PDA":
            density.project(apx.density(p, T, 0.0) - apx.density(0.0, 0.0, 0.0))

    pvd.write(
        *stokes.subfunctions, temperature, density, time=simu_time / year_to_seconds
    )


domain_dims = (6e6, 3e6)
mesh_resolution = 2.34375e4
mesh_cells = [int(dim / mesh_resolution) for dim in domain_dims]
mesh = RectangleMesh(*mesh_cells, *domain_dims, quadrilateral=True)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)
x, y = SpatialCoordinate(mesh)
depth = domain_dims[1] - y

V = VectorFunctionSpace(mesh, "Q", 2)
W = FunctionSpace(mesh, "Q", 1)
Z = MixedFunctionSpace([V, W])
Q = FunctionSpace(mesh, "Q", 2)
R = FunctionSpace(mesh, "R", 0)

stokes = Function(Z, name="Stokes")
stokes.subfunctions[0].rename("Velocity")
stokes.subfunctions[1].rename("Pressure")
u, p = split(stokes)
T = Function(Q, name="Temperature (deviation)")
temperature = Function(W, name="Temperature")
density = Function(W, name="Density (deviation)")
time_step = Function(R, name="Time step").assign(1e12)

T.interpolate(
    5.0 * exp(-((x - 2e5) ** 2) / 2 / 4e5**2 - depth**2 / 2 / 2e5**2)
    + 3.0 * exp(-((x - 3.5e6) ** 2) / 2 / 4e5**2 - depth**2 / 2 / 2e5**2)
)

ref_state = {
    "alpha": Function(R, name="Coefficient of thermal expansion (state)").assign(4e-5),
    "cp": Function(R, name="Isobaric specific heat capacity (state)").assign(1250.0),
    "eta": Function(R, name="Dynamic viscosity (state)").assign(1e22),
    "g": Function(R, name="Gravitational acceleration (state)").assign(10.0),
    "H": Function(R, name="Rate of radiogenic heat production (state)").assign(0.0),
    "K": Function(R, name="Bulk modulus (state)").assign(2.5e11),
    "k": Function(R, name="Thermal conductivity (state)").assign(4.7),
    "p": Function(R, name="Pressure (state)").assign(0.0),
    "rho": Function(R, name="Density (state)").assign(3300.0),
    "T": Function(R, name="Temperature (state)").assign(1600.0),
}
ref_profiles = {}
ref_profiles["rho"] = ref_state["rho"] * exp(
    ref_state["rho"] * ref_state["g"] / ref_state["K"] * depth
)
ref_profiles["p"] = ref_state["p"] + ref_state["K"] * (
    ref_profiles["rho"] / ref_state["rho"] - 1.0
)
ref_profiles["alpha"] = ref_state["alpha"] * exp(-1.117979e-11 * ref_profiles["p"])
ref_profiles["T"] = ref_state["T"] * exp(
    ref_profiles["alpha"] * ref_state["g"] / ref_state["cp"] * depth
)
apx = Approximation("HCA", ref_state, ref_profiles)

energy_bcs = [
    DirichletBC(Q, 3700.0 - apx.ref_profiles["T"], boundary.bottom),
    DirichletBC(Q, 273.0 - apx.ref_profiles["T"], boundary.top),
]
for bc in energy_bcs:
    bc.apply(T)
stokes_bcs = [
    DirichletBC(Z.subspaces[0].sub(0), 0.0, boundary.left),
    DirichletBC(Z.subspaces[0].sub(0), 0.0, boundary.right),
    DirichletBC(Z.subspaces[0].sub(1), 0.0, boundary.bottom),
    DirichletBC(Z.subspaces[0].sub(1), 0.0, boundary.top),
]

energy_solver = EnergySolver(
    T, apx, u, time_step, ImplicitMidpoint, strong_bcs=energy_bcs
)
stokes_solver = StokesSolver(
    stokes,
    apx,
    T=T,
    T_old=energy_solver.timestepper.solution_old if apx.name == "PDA" else 0.0,
    time_step=time_step if apx.name == "PDA" else None,
    strong_bcs=stokes_bcs,
    nullspace={
        "closed": True,
        "rotational": False,
        "translations": None,
        "boundary_id": boundary.top if apx.name == "ALA" else None,
    },
    transpose_nullspace={"closed": True, "rotational": False, "translations": None},
)
stokes_solver.solve()

year_to_seconds = 365.25 * 8.64e4
ts_adaptor = TimestepAdaptor(
    time_step, u, V, target_cfl=1.0, maximum_timestep=1e7 * year_to_seconds
)
ts_adaptor.update_timestep()

simu_time = 0.0
pvd = VTKFile(f"output_box_convection_{apx.name}.pvd")
write_output()

diag_time = []
heat_flux_top = []
calculate_diagnostics()

while simu_time < 4e8 * year_to_seconds:
    energy_solver.solve()
    stokes_solver.solve()

    simu_time += ts_adaptor.update_timestep()

    write_output()
    calculate_diagnostics()

fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlim(1.5e8, 4e8)
ax.set_ylim(2e5, 1.2e6)
ax.plot(diag_time, heat_flux_top)
ax.set_xlabel("Time")
ax.set_ylabel("Top heat flux")
ax.grid(which="both")
plt.savefig(f"heat_flux_top_{apx.name}.pdf", bbox_inches="tight", dpi=300)
