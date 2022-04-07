from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI  # noqa: F401
import sys
import assess
import math

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry and key parameters:
rmin, rmax = 1.22, 2.22
k = int(sys.argv[1])  # radial degree
l = int(sys.argv[2])  # spherical harmonic degree
m = int(sys.argv[3])  # spherical harmonic order
nlevels = int(sys.argv[4])  # levels of refinement of cubed sphere mesh
nlayers = int(sys.argv[5])  # vertical layers


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
        "ksp_rtol": 1e-10,
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
        "ksp_rtol": 1e-9,
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


def model(ref_level, radial_layers):
    # Mesh and associated physical boundary IDs:
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, radial_layers, (rmax-rmin)/radial_layers, extrusion_type="radial")

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    rhat = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    Vscl = FunctionSpace(mesh, "CG", 2)  # scalar version of V, used for rho'
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # vector version of W, used for coordinates in anal. pressure solution

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

    # rhs provided by assess solution: rho'=r**k Y_lm
    solution = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, nu=float(mu), Rp=rmax, Rm=rmin, g=float(g))
    u_xyz = interpolate(X, V)
    rhop = Function(Vscl)
    rhop.dat.data[:] = [solution.delta_rho_cartesian(xyzi) for xyzi in u_xyz.dat.data]

    # Setup UFL, incorporating Nitsche boundary conditions:
    stress = 2 * mu * sym(grad(u))
    F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb + g * rhop * dot(v, rhat) * dx
    F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb  # Continuity equation

    # nitsche free slip BCs
    F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds_tb
    F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds_tb
    F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

    # Nullspaces and near-nullspaces:
    x_rotV = Function(V).interpolate(as_vector((0, X[2], -X[1])))
    y_rotV = Function(V).interpolate(as_vector((-X[2], 0, X[0])))
    z_rotV = Function(V).interpolate(as_vector((-X[1], X[0], 0)))
    u_nullspace = VectorSpaceBasis([x_rotV, y_rotV, z_rotV])
    u_nullspace.orthonormalize()
    p_nullspace = VectorSpaceBasis(constant=True)  # constant nullspace for pressure
    Z_nullspace = MixedVectorSpaceBasis(Z, [u_nullspace, p_nullspace])  # combined mixed nullspace

    # Generating near_nullspaces for GAMG:
    nns_x = Function(V).interpolate(Constant([1., 0., 0.]))
    nns_y = Function(V).interpolate(Constant([0., 1., 0.]))
    nns_z = Function(V).interpolate(Constant([0., 0., 1.]))
    u_near_nullspace = VectorSpaceBasis([nns_x, nns_y, nns_z, x_rotV, y_rotV, z_rotV])
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
    # removing rotations around x, y, and z axes from velocity:
    x_coef = assemble(dot(x_rotV, u_) * dx) / assemble(dot(x_rotV, x_rotV) * dx)
    u_.project(u_ - x_rotV*x_coef, solver_parameters=project_solver_parameters)
    y_coef = assemble(dot(y_rotV, u_) * dx) / assemble(dot(y_rotV, y_rotV) * dx)
    u_.project(u_ - y_rotV*y_coef, solver_parameters=project_solver_parameters)
    z_coef = assemble(dot(z_rotV, u_) * dx) / assemble(dot(z_rotV, z_rotV) * dx)
    u_.project(u_ - z_rotV*z_coef, solver_parameters=project_solver_parameters)

    # removing constant nullspace from pressure
    p_coef = assemble(p_ * dx)/assemble(Constant(1.0)*dx(domain=mesh))
    p_.project(p_ - p_coef, solver_parameters=project_solver_parameters)

    # compute u analytical and error
    u_xyz_p2 = interpolate(X, V)
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal.dat.data[:] = [solution.velocity_cartesian(xyz) for xyz in u_xyz_p2.dat.data]
    u_error = Function(V, name="VelocityError").assign(u_-u_anal)

    # compute p analytical and error
    p_xyz = interpolate(X, Wvec)
    p_anal = Function(W, name="AnalyticalPressure")
    p_anal.dat.data[:] = [solution.pressure_cartesian(xyzi) for xyzi in p_xyz.dat.data]
    p_error = Function(W, name="PressureError").assign(p_-p_anal)

    # Write output files in VTK format:
    u_.rename("Velocity")
    p_.rename("Pressure")
    u_file = File("fs_velocity.pvd")
    p_file = File("fs_pressure.pvd")

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
l2error_u, l2error_p, l2anal_u, l2anal_p = model(nlevels, nlayers)
log_params(f, f"{l2error_u} {l2error_p} {l2anal_u} {l2anal_p}")
f.close()
