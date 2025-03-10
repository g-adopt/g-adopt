from gadopt import *
from gadopt.inverse import *


def helmholtz(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u)) * dx + 100.0 * v * u * dx - v * source * dx

    solve(F == 0, u)
    return u


def run(optimiser, rf, rank, filename):
    if rank == 0:
        with open(filename, "w") as f:
            rf.eval_cb_post = lambda val, *args: f.write(f"{val}\n")
            optimiser.run()
            rf.eval_cb_pots = lambda *args: None
    else:
        optimiser.run()


mesh = UnitIntervalMesh(10)
num_processes = mesh.comm.size
mesh_checkpoint = f"mesh_helmholtz_np{num_processes}.h5"
# create a checkpointable mesh by writing to disk and restoring
with CheckpointFile(mesh_checkpoint, "w") as f:
    f.save_mesh(mesh)
with CheckpointFile(mesh_checkpoint, "r") as f:
    mesh = f.load_mesh("firedrake_default")

V = FunctionSpace(mesh, "CG", 1)
source_ref = Function(V)
x = SpatialCoordinate(mesh)
source_ref.interpolate(cos(pi * x**2))

with stop_annotating():
    # compute reference solution
    u_ref = helmholtz(V, source_ref)

source = Function(V)
c = Control(source)
# tape the forward solution
u = helmholtz(V, source)

J = assemble(1e6 * (u - u_ref) ** 2 * dx)
rf = ReducedFunctional(J, c)

T_lb = Function(V, name="Lower bound")
T_ub = Function(V, name="Upper bound")
T_lb.assign(-1.0)
T_ub.assign(1.0)

minimisation_problem = MinimizationProblem(rf, bounds=(T_lb, T_ub))
minimisation_parameters["Status Test"]["Iteration Limit"] = 10

# run full optimisation, checkpointing every iteration
checkpoint_dir = f"optimisation_checkpoint_np{num_processes}"
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=checkpoint_dir,
)
run(optimiser, rf, mesh.comm.rank, f"full_optimisation_np{num_processes}.dat")

# re-initialise optimiser, and restore from checkpoint 5
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=checkpoint_dir,
    auto_checkpoint=False,
)
optimiser.restore(5)
run(
    optimiser,
    rf,
    mesh.comm.rank,
    f"restored_optimisation_from_it_5_np{num_processes}.dat",
)

# re-initialise optimiser, and restore from last stored checkpoint

minimisation_parameters["Status Test"]["Iteration Limit"] = 15
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=checkpoint_dir,
    auto_checkpoint=True,
)
optimiser.restore()
run(
    optimiser,
    rf,
    mesh.comm.rank,
    f"restored_optimisation_from_last_it_np{num_processes}.dat",
)
