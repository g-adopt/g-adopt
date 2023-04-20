from firedrake import *
from firedrake.petsc import PETSc
from pyop2.utils import cached_property
from mpi4py import MPI
import numpy as np

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
nx, ny = 40, 40
mesh = UnitSquareMesh(nx, ny, quadrilateral=False)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation
domain_volume = assemble(1.*dx(domain=mesh))  # Required for diagnostics (e.g. RMS velocity)


# Define logging convenience functions:
def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


def log_params(f, str):
    """Log diagnostic paramters on root processor only"""
    if mesh.comm.rank == 0:
        f.write(str + "\n")
        f.flush()


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = FunctionSpace(mesh, "BDM", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "DG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Output function space information:
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
Told, Tnew = Function(Q, name="OldTemp"), Function(Q, name="NewTemp")
pi = 3.141592653589793238
Told.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))
Tnew.assign(Told)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5 * Tnew + (1-0.5) * Told

# Define time stepping parameters:
steady_state_tolerance = 1e-9
max_timesteps = 20000
target_cfl_no = 1.0
maximum_timestep = 0.1
increase_tolerance = 1.5
time = 0.0

# Timestepping - CFL related stuff:
ref_vel = Function(V, name="Reference_Velocity")


def compute_timestep(u, current_delta_t):
    """Return the timestep, based upon the CFL criterion"""

    ref_vel.interpolate(dot(JacobianInverse(mesh), u))
    ts_min = 1. / mesh.comm.allreduce(np.abs(ref_vel.dat.data).max(), MPI.MAX)
    # Grab (smallest) maximum permitted on all cores:
    ts_max = min(float(current_delta_t)*increase_tolerance, maximum_timestep)
    # Compute timestep:
    tstep = min(ts_min*target_cfl_no, ts_max)
    return tstep


# Solver dictionary:
solver_parameters_lu = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

solver_parameters_allu = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    # "pc_fieldsplit_schur_factorization_type": "upper",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "fgmres",
    "fieldsplit_1_ksp_max_it": 1,
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": __name__ + ".MassForm",
    "fieldsplit_1_aux_pc_type": "lu",
    "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
}

class MassForm(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        a = (gamma + mu) * inner(test, trial)*dx
        bcs = FixAtPointBC(trial.function_space(), 0, as_vector([0, 0]))
        #bcs = None
        return (a, bcs)

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(1e4)  # Rayleigh number
mu = Constant(1.0)  # Viscosity
k = Constant((0, 1))  # Unit vector (in direction opposite to gravity).
gamma = Constant(100) # Augmented Lagrangian term

# Temperature equation related constants:
delta_t = Constant(1e-6)  # Initial time-step
kappa = Constant(1.0)  # Thermal diffusivity

# Stokes equations in UFL form:
sigma = Constant(10) * (V.ufl_element().degree() + 1) * V.ufl_element().degree()  # SIPG penalty parameter
h = CellDiameter(mesh)
n = FacetNormal(mesh)
F_stokes = (
             inner(2 * mu * sym(grad(u)), grad(v)) * dx
           - inner(avg(2 * mu * sym(grad(u))), 2 * avg(outer(v, n))) * dS
           - inner(avg(2 * mu * sym(grad(v))), 2 * avg(outer(u, n))) * dS
           + sigma/avg(h) * inner(2 * mu * avg(outer(u,n)), 2 * avg(outer(v,n))) * dS
           - div(v) * p * dx
           - (dot(v, k) * Ra * Ttheta) * dx
           + gamma * inner(div(v), div(u))*dx
           - w * div(u) * dx
           # Would need more terms for weak enforcement of Dirichlet BCs on tangential components,
           # but this problem doesn't have any.
           # If you do, see line 685 of branch fw/hdiv of alfi/solver.py in alfi---the code would be
           # - inner(outer(v,n),2*mu*sym(grad(u)))*ds(bid)
           # - inner(outer(u-g,n),2*mu*sym(grad(v)))*ds(bid)
           # + mu*(sigma/h)*inner(v,u-g)*ds(bid)
           # where bid = the boundary id where you want to impose the condition, and
           #       g = Dirichlet data
           )


# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx + dot(grad(q), kappa * grad(Ttheta)) * dx

# Set up boundary conditions for Stokes: We first extract the velocity
# field from the mixed function space Z. This is done using
# Z.sub(0). We subsequently extract the x and y components of
# velocity. This is done using an additional .sub(0) and .sub(1),
# respectively. Note that the final arguments here are the physical
# boundary ID's.
if V.ufl_element().family() == "Lagrange":
    bcvx, bcvy = DirichletBC(Z.sub(0).sub(0), 0, (left_id, right_id)), DirichletBC(Z.sub(0).sub(1), 0, (bottom_id, top_id))
elif V.ufl_element().family() == "Brezzi-Douglas-Marini":
    bcvx = DirichletBC(Z.sub(0), 0, (left_id, right_id))  # only normal component will be enforced
    bcvy = DirichletBC(Z.sub(0), 0, (bottom_id, top_id))  # same


class FixAtPointBC(DirichletBC):
    def __init__(self, V, g, bc_point):
        super().__init__(V, g, "on_boundary")
        self.bc_point = bc_point

    @cached_property
    def nodes(self):
        V = self.function_space()

        x_sc = SpatialCoordinate(V.mesh())
        point_dist = interpolate(sqrt(dot(x_sc - self.bc_point, x_sc - self.bc_point)), V)

        with point_dist.dat.vec as v:
            min_index, min_value = v.min()
            assert min_value < 1e-14

            v.zeroEntries()
            v.setValue(min_index, 1.0)

        nodes, = np.nonzero(point_dist.dat.data_ro_with_halos)
        assert len(nodes) <= 1
        nodes = np.asarray([nodes[0]] if (len(nodes) > 0) else [], dtype=int)

        nodes_lgmap = V.dof_dset.lgmap.applyInverse([min_index])
        nodes_lgmap = nodes_lgmap[nodes_lgmap >= 0]
        assert len(nodes_lgmap) <= 1

        assert set(nodes) == set(nodes_lgmap)
        return nodes
bcp = FixAtPointBC(Z.sub(1), 0, as_vector([0, 0]))

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0, bottom_id), DirichletBC(Q, 0.0, top_id)

# Pressure nullspace
p_nullspace = None

# Write output files in VTK format:
u, p = z.split()  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")
# Create output file and select output_frequency:
output_file = File("output.pvd")
dump_period = 50
# Frequency of checkpoint files:
checkpoint_period = dump_period * 4
# Open file for logging diagnostic output:
f = open("params.log", "w")
log_params(f, "timestep time dt maxchange u_rms u_rms_surf ux_max nu_base nu_top energy avg_t")

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z, bcs=[bcvx, bcvy, bcp])
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=solver_parameters_lu, nullspace=p_nullspace, transpose_nullspace=p_nullspace)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=solver_parameters_lu)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    # Write output:
    if timestep % dump_period == 0:
        output_file.write(u, p, Tnew)

    current_delta_t = delta_t
    if timestep != 0:
        delta_t.assign(compute_timestep(u, current_delta_t))  # Compute adaptive time-step
    time += float(delta_t)

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    u_rms = sqrt(assemble(dot(u, u) * dx)) * sqrt(1./domain_volume)  # RMS velocity
    u_rms_surf = sqrt(assemble(u[0] ** 2 * ds(top_id)))  # RMS velocity at surface
    bcu = DirichletBC(u.function_space(), 0, top_id)
    #ux_max = u.dat.data_ro_with_halos[bcu.nodes, 0].max(initial=0)
    ux_max = 0
    ux_max = u.comm.allreduce(ux_max, MPI.MAX)  # Maximum Vx at surface
    nusselt_number_top = -1 * assemble(dot(grad(Tnew), n) * ds(top_id))
    nusselt_number_base = assemble(dot(grad(Tnew), n) * ds(bottom_id))
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))
    average_temperature = assemble(Tnew * dx) / domain_volume

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((Tnew - Told)**2 * dx))

    # Log diagnostics:
    log_params(f, f"{timestep} {time} {float(delta_t)} {maxchange} {u_rms} {u_rms_surf} {ux_max} "
               f"{nusselt_number_base} {nusselt_number_top} "
               f"{energy_conservation} {average_temperature} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)

    # Checkpointing:
    if timestep % checkpoint_period == 0:
        # Checkpointing during simulation:
        checkpoint_data = DumbCheckpoint(f"Temperature_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(Tnew, name="Temperature")
        checkpoint_data.close()

        checkpoint_data = DumbCheckpoint(f"Stokes_State_{timestep}", mode=FILE_CREATE)
        checkpoint_data.store(z, name="Stokes")
        checkpoint_data.close()

f.close()

# Write final state:
final_checkpoint_data = DumbCheckpoint("Final_Temperature_State", mode=FILE_CREATE)
final_checkpoint_data.store(Tnew, name="Temperature")
final_checkpoint_data.close()

final_checkpoint_data = DumbCheckpoint("Final_Stokes_State", mode=FILE_CREATE)
final_checkpoint_data.store(z, name="Stokes")
final_checkpoint_data.close()
