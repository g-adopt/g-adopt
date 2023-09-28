from firedrake import *
from firedrake.petsc import PETSc
from gadopt.inverse import *
from gadopt import timer_decorator
import numpy as np
import libgplates
from wrappers import collect_garbage
from firedrake.adjoint_utils import blocks


class VariableMassInvPC(AuxiliaryOperatorPC):
    """Scaled inverse pressure mass preconditioner
    Note that unlike MassInvPC this PC will update for variable viscosity."""

    # use same prefix as MassInvPC, so we can use it as drop-in replacement in options
    _prefix = "Mp_"

    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        context = P.getPythonContext()
        mu = context.appctx.get("mu", 1.0)
        # for future Augmented Lagrangian term:
        gamma = context.appctx.get("gamma", 0.0)
        return inner(-1 / (mu + gamma) * test, trial) * dx, []


class SPDAssembledPC(AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.
    For use in the fieldsplit_0 block in combination with gamg."""

    def initialize(self, pc):
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


# Quadrature degree:
dx = dx(degree=6)
ds_b = ds_b(degree=6)
ds_t = ds_t(degree=6)
ds_tb = ds_t + ds_b

# Logging fiunction
log = PETSc.Sys.Print

# Projection solver parameters for nullspaces:
iterative_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}

LinearSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
NonlinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
LinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
NonlinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters

blocks.solving.Block.evaluate_adj = collect_garbage(blocks.solving.Block.evaluate_adj)
blocks.solving.Block.recompute = collect_garbage(blocks.solving.Block.recompute)

# timer decorator for fwd and derivative calls.
ReducedFunctional.__call__ = collect_garbage(
    timer_decorator(ReducedFunctional.__call__)
)
ReducedFunctional.derivative = collect_garbage(
    timer_decorator(ReducedFunctional.derivative)
)


def my_taylor_test():
    Tic, reduced_functional = forward_problem()
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    minconv = taylor_test(reduced_functional, Tic, Delta_temp)


@collect_garbage
def forward_problem():
    # Set up geometry:
    rmin, rmax = 1.22, 2.22

    enable_disk_checkpointing()

    # Load mesh
    with CheckpointFile("../../Adjoint_CheckpointFile.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")
        Tic = f.load_function(mesh, name="Temperature")
        z = f.load_function(mesh, name="Stokes")
        Tobs = f.load_function(mesh, name="ReferenceTemperature")

    bottom_id, top_id = "bottom", "top"
    n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    q = TestFunction(Q)
    u, p = split(z)

    # Set up temperature field and initialise:
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)

    Told = Function(Q, name="OldTemp")
    Tnew = Function(Q, name="NewTemp")
    T0 = Constant(0.091)  # Non-dimensional surface temperature

    # Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
    Ttheta = 0.5 * Tnew + (1 - 0.5) * Told

    time = (409 - 10.0) * (
        libgplates.myrs2sec
        * libgplates.plate_scaling_factor
        / libgplates.time_dim_factor
    )
    time_presentday = 409 * (
        libgplates.myrs2sec
        * libgplates.plate_scaling_factor
        / libgplates.time_dim_factor
    )

    delta_t = Constant(1.0e-6)

    # Stokes Equation Solver Parameters:
    stokes_solver_parameters = {
        "mat_type": "matfree",
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 100,
        "snes_atol": 1e-10,
        "snes_rtol": 5e-2,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "preonly",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_type": "full",
        "fieldsplit_0": {
            "ksp_type": "cg",
            "ksp_rtol": 5e-4,
            "ksp_monitor": None,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "__main__.SPDAssembledPC",
            "assembled_pc_type": "gamg",
        },
        "fieldsplit_1": {
            "ksp_type": "fgmres",
            "ksp_rtol": 5e-3,
            "ksp_monitor": None,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "__main__.VariableMassInvPC",
            "Mp_ksp_type": "cg",
            "Mp_pc_type": "sor",
        },
    }

    # Energy Equation Solver Parameters:
    energy_solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_rtol": 1e-4,
        "ksp_converged_reason": None,
        "pc_type": "sor",
    }

    # GPLATES
    X_val = interpolate(X, V)

    # set up a Function for gplate velocities
    gplates_velocities = Function(V, name="GPlates_Velocity")

    # Setup Equations Stokes related constants
    Ra = Constant(5.0e7)  # Rayleigh number
    Di = Constant(0.5)  # Dissipation number.
    mu_lin = 2.0

    # A step function designed to control viscosity jumps:
    def step_func(r, center, mag, increasing=True, sharpness=30):
        """
        input:
            r: is the radius array
            center: radius of the jump
            increasing: if True, the jump happens towards lower r
                        otherwise jump happens at higher r
            sharpness: how sharp should the jump should be (larger numbers = sharper).
        """
        if increasing:
            sign = 1
        else:
            sign = -1
        return mag * (0.5 * (1 + tanh(sign * (r - center) * sharpness)))

    # assemble the depth dependence
    # for the lower mantle increase we multiply the profile with a
    # linear function
    for line, step in zip(
        [5.0 * (rmax - r), 1.0, 1.0],
        [
            step_func(r, 1.992, 30, False),
            step_func(r, 2.078, 10, False),
            step_func(r, 2.2, 10, True),
        ],
    ):
        mu_lin += line * step

    # adding temperature dependence of visc
    delta_mu_T = Constant(100.0)
    mu_lin *= exp(-ln(delta_mu_T) * Tnew)
    mu_star, sigma_y = Constant(1.0), 5.0e5 + 2.5e6 * (rmax - r)
    epsilon = sym(grad(u))  # strain-rate
    epsii = sqrt(inner(epsilon, epsilon) + 1e-10)
    mu_plast = mu_star + (sigma_y / epsii)
    mu = (2.0 * mu_lin * mu_plast) / (mu_lin + mu_plast)

    k = as_vector((X[0], X[1], X[2])) / r
    C_ip = Constant(100.0)
    p_ip = 2
    # Temperature equation related constants:
    tcond = Constant(1.0)  # Thermal conductivity
    H_int = Constant(10.0)  # Internal heating

    # Compressible reference state:
    rho_0, alpha = 1.0, 1.0
    weight = r - rmin
    rhobar = Function(Q, name="CompRefDensity").interpolate(
        rho_0 * exp(((1.0 - weight) * Di) / alpha)
    )
    Tbar = Function(Q, name="CompRefTemperature").interpolate(
        T0 * exp((1.0 - weight) * Di) - T0
    )
    alphabar = Function(Q, name="IsobaricThermalExpansivity").interpolate(
        0.3 + (weight * 0.7)
    )
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)

    # Stokes equations in UFL form:
    I = Identity(3)

    def stress(u):
        return 2 * mu * sym(grad(u)) - 2.0 / 3.0 * I * mu * div(u)

    F_stokes = (
        inner(grad(v), stress(u)) * dx
        - div(v) * p * dx
        + dot(n, v) * p * ds_tb
        - (dot(v, k) * (Ra * Ttheta * rhobar * alphabar) * dx)
    )
    F_stokes += (
        -w * div(rhobar * u) * dx + w * dot(n, rhobar * u) * ds_tb
    )  # Continuity equation

    # nitsche free-slip BCs
    F_stokes += -dot(v, n) * dot(dot(n, stress(u)), n) * ds_b
    F_stokes += -dot(u, n) * dot(dot(n, stress(v)), n) * ds_b
    F_stokes += (
        C_ip
        * mu
        * (p_ip + 1) ** 2
        * FacetArea(mesh)
        / CellVolume(mesh)
        * dot(u, n)
        * dot(v, n)
        * ds_b
    )

    # No-Slip (prescribed) boundary condition for the top surface
    bc_gplates = DirichletBC(Z.sub(0), gplates_velocities, (top_id))
    boundary_X = X_val.dat.data_ro_with_halos[bc_gplates.nodes]

    # For SU stabilisation:
    def absv(u):
        """Component-wise absolute value of vector"""
        return as_vector([abs(ui) for ui in u])

    def beta(Pe):
        """Component-wise beta formula Donea and Huerta (2.47a)"""
        return as_vector([1 / tanh(Pei + 1e-6) - 1 / (Pei + 1e-6) for Pei in Pe])

    # SU(PG) ala Donea & Huerta:
    J = Function(TensorFunctionSpace(mesh, "DQ", 1), name="Jacobian").interpolate(
        Jacobian(mesh)
    )
    Pe = absv(dot(u, J)) / 2
    nubar = dot(Pe, beta(Pe))
    q_SU = q + nubar / dot(u, u) * dot(u, grad(q))

    # Energy equation in UFL form:
    F_energy = (
        q * rhobar * cpbar * ((Tnew - Told) / delta_t) * dx
        + q_SU * rhobar * cpbar * dot(u, grad(Ttheta)) * dx
        + dot(grad(q), tcond * grad(Tbar + Ttheta)) * dx
        + q * (alphabar * rhobar * Di * dot(u, k) * Ttheta) * dx
        - q * ((Di / Ra) * inner(stress(u), grad(u))) * dx
        - q * rhobar * H_int * dx
    )

    # Temperature boundary conditions
    bctb, bctt = DirichletBC(Q, 1.0 - (T0 * exp(Di) - T0), bottom_id), DirichletBC(
        Q, 0.0, top_id
    )

    # Nullspaces and near-nullspaces:
    p_nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    Z_nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), p_nullspace])

    # Generating near_nullspaces for GAMG:
    x_rotV = Function(V).interpolate(as_vector((0, X[2], -X[1])))
    y_rotV = Function(V).interpolate(as_vector((-X[2], 0, X[0])))
    z_rotV = Function(V).interpolate(as_vector((-X[1], X[0], 0)))
    nns_x = Function(V).interpolate(Constant([1.0, 0.0, 0.0]))
    nns_y = Function(V).interpolate(Constant([0.0, 1.0, 0.0]))
    nns_z = Function(V).interpolate(Constant([0.0, 0.0, 1.0]))
    V_near_nullspace = VectorSpaceBasis(
        [nns_x, nns_y, nns_z, x_rotV, y_rotV, z_rotV], comm=mesh.comm
    )
    V_near_nullspace.orthonormalize()
    Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

    # Write output files in VTK format:
    (
        u_,
        p_,
    ) = (
        z.subfunctions
    )  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")

    NonlinearVariationalSolver.solve = collect_garbage(NonlinearVariationalSolver.solve)

    # Setup problem and solver objects so we can reuse (cache) solver setup
    stokes_problem = NonlinearVariationalProblem(
        F_stokes, z, bcs=[bc_gplates]
    )  # velocity BC for the bottom surface is handled through Nitsche, top surface through gplates_velocities
    # Add garbage collection

    stokes_solver = NonlinearVariationalSolver(
        stokes_problem,
        solver_parameters=stokes_solver_parameters,
        appctx={"mu": mu},
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
    )
    energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
    energy_solver = NonlinearVariationalSolver(
        energy_problem, solver_parameters=energy_solver_parameters
    )

    control = Control(Tic)

    project(
        Tic,
        Told,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=[bctb, bctt],
    )

    Tnew.assign(Told)

    # Now perform the time loop:
    while time < time_presentday:
        # Update gplates velocities
        libgplates.rec_model.set_time(model_time=time)
        gplates_velocities.dat.data_with_halos[
            bc_gplates.nodes
        ] = libgplates.rec_model.get_velocities(boundary_X)

        # Solve Stokes sytem:
        stokes_solver.solve()

        if time_presentday - time < float(delta_t):
            delta_t.assign(time_presentday - time)

        # Temperature system:
        energy_solver.solve()

        # Set Told = Tnew
        Told.assign(Tnew)

        time += float(delta_t)

    # Temperature misfit between solution and observation
    objective = assemble((Tnew - Tobs) ** 2 * dx)

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()
    return Tic, ReducedFunctional(objective, control)


if __name__ == "__main__":
    my_taylor_test()
