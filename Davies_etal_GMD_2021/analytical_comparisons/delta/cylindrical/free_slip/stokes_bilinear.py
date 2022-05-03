from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI  # noqa: F401
import sys
import assess
import math
PETSc.Sys.popErrorHandler()

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry and key parameters:
rmin, rmax = 1.22, 2.22
nn = Constant(int(sys.argv[1]))  # wave number (n is already used for FacetNormal)
level = int(sys.argv[2])  # refinement level
rp = (rmax + rmin) / 2.0  # radius of delta_function

# Define logging convenience functions:


def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


def log_params(f, str):
    """Log diagnostic parameters"""
    f.write(str + "\n")
    f.flush()


# Stokes Equation Solver Parameters:
stokes_solver_parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
        "ksp_rtol": 1e-14,
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
        "ksp_rtol": 1e-12,
    }
}

# Projection solver parameters for nullspaces:
project_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}


def model(disc_n):
    ncells = disc_n*256
    nlayers = disc_n*16

    # Construct a circle mesh and then extrude into a cylinder:
    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan_2(X[1], X[0])

    # Set up function spaces - currently using the Q2Q1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    P0 = FunctionSpace(mesh, "DQ", 0)
    Q1DG = FunctionSpace(mesh, "DQ", 1)
    Q1DGvec = VectorFunctionSpace(mesh, "DQ", 1)

    # Set up mixed function space and associated test functions:
    Z = MixedFunctionSpace([V, W])
    v, w = TestFunctions(Z)

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)
    u_, p_ = z.split()

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity
    g = Constant(1.0)
    C_ip = Constant(100.0)  # The fudge factor for interior penalty term used in weak imposition of BCs
    p_ip = 2  # maximum polynomial degree of the _gradient_ of velocity

    marker = Function(P0)
    marker.interpolate(conditional(r < rp, 1, 0))

    # Setup UFL, incorporating Nitsche boundary conditions:
    stress = 2 * mu * sym(grad(u))
    F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb + g * cos(nn*phi) * dot(jump(marker, n), avg(v)) * dS_h
    F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb  # Continuity equation

    # nitsche free slip BCs
    F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds_tb
    F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds_tb
    F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

    # Nullspaces and near-nullspaces:
    xy_rot = Function(V).interpolate(as_vector((-X[1], X[0])))  # rotation interpolated as vector function in V
    u_nullspace = VectorSpaceBasis([xy_rot])
    u_nullspace.orthonormalize()
    p_nullspace = VectorSpaceBasis(constant=True)  # constant nullspace for pressure
    Z_nullspace = MixedVectorSpaceBasis(Z, [u_nullspace, p_nullspace])  # combined mixed nullspace

    # Generating near_nullspaces for GAMG:
    nns_x = Function(V).interpolate(Constant([1., 0.]))
    nns_y = Function(V).interpolate(Constant([0., 1.]))
    u_near_nullspace = VectorSpaceBasis([nns_x, nns_y, xy_rot])
    u_near_nullspace.orthonormalize()
    Z_near_nullspace = MixedVectorSpaceBasis(Z, [u_near_nullspace, Z.sub(1)])

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    solve(
        F_stokes == 0, z,
        solver_parameters=stokes_solver_parameters,
        appctx={"mu": mu},
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace
    )

    # take out null modes through L2 projection from velocity and pressure
    # removing rotation from velocity:
    rot = as_vector((-X[1], X[0]))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=project_solver_parameters)

    # removing constant nullspace from pressure
    coef = assemble(p_ * dx)/assemble(Constant(1.0)*dx(domain=mesh))
    p_.project(p_ - coef, solver_parameters=project_solver_parameters)

    solution_upper = assess.CylindricalStokesSolutionDeltaFreeSlip(float(nn), +1, nu=float(mu))
    solution_lower = assess.CylindricalStokesSolutionDeltaFreeSlip(float(nn), -1, nu=float(mu))

    # compute u analytical and error
    uxy = interpolate(as_vector((X[0], X[1])), V)
    u_anal_upper = Function(V, name="AnalyticalVelocityUpper")
    u_anal_lower = Function(V, name="AnalyticalVelocityLower")
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal_upper.dat.data[:] = [solution_upper.velocity_cartesian(xyi) for xyi in uxy.dat.data]
    u_anal_lower.dat.data[:] = [solution_lower.velocity_cartesian(xyi) for xyi in uxy.dat.data]
    u_anal.interpolate(marker*u_anal_lower + (1-marker)*u_anal_upper)
    u_error = Function(V, name="VelocityError").assign(u_-u_anal)

    # compute p analytical and error
    pxy = interpolate(as_vector((X[0], X[1])), Q1DGvec)
    pdg = interpolate(p, Q1DG)
    p_anal_upper = Function(Q1DG, name="AnalyticalPressureUpper")
    p_anal_lower = Function(Q1DG, name="AnalyticalPressureLower")
    p_anal = Function(Q1DG, name="AnalyticalPressure")
    p_anal_upper.dat.data[:] = [solution_upper.pressure_cartesian(xyi) for xyi in pxy.dat.data]
    p_anal_lower.dat.data[:] = [solution_lower.pressure_cartesian(xyi) for xyi in pxy.dat.data]
    p_anal.interpolate(marker*p_anal_lower + (1-marker)*p_anal_upper)
    p_error = Function(Q1DG, name="PressureError").assign(pdg-p_anal)

    # Write output files in VTK format:
    u_.rename("Velocity")
    p_.rename("Pressure")
    u_file = File("fs_velocity_{}.pvd".format(disc_n))
    p_file = File("fs_pressure_{}.pvd".format(disc_n))

    # Write output:
    u_file.write(u_, u_anal, u_error)
    p_file.write(p_, p_anal, p_error)

    l2anal_u = math.sqrt(assemble(dot(u_anal, u_anal)*dx))
    l2anal_p = math.sqrt(assemble(dot(p_anal, p_anal)*dx))
    l2error_u = math.sqrt(assemble(dot(u_error, u_error)*dx))
    l2error_p = math.sqrt(assemble(dot(p_error, p_error)*dx))

    return l2error_u, l2error_p, l2anal_u, l2anal_p


# Run model at different levels:
f = open('errors.log', 'w')
l2error_u, l2error_p, l2anal_u, l2anal_p = model(level)
log_params(f, f"{l2error_u} {l2error_p} {l2anal_u} {l2anal_p}")
f.close()
