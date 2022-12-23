from gadopt import *
import sys
import numpy
import assess

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry and key parameters:
rmin, rmax = 1.22, 2.22
k = Constant(int(sys.argv[1]))  # radial degree
nn = Constant(int(sys.argv[2]))  # wave number (n is already used for FacetNormal)
level = int(sys.argv[3])  # refinement level

# SCK: should be able to get rid off:
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
    bottom_id, top_id = "bottom", "top"

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan_2(X[1], X[0])

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # vector version of W, used for coordinates in anal. pressure solution
    Z = MixedFunctionSpace([V, W])
    v, w = TestFunctions(Z)

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)
    u_, p_ = z.split()

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity
#    g = Constant(1.0)
#    C_ip = Constant(100.0)  # The fudge factor for interior penalty term used in weak imposition of BCs
#    p_ip = 2  # maximum polynomial degree of the _gradient_ of velocity

    T = r**k/rmax**k*cos(nn*phi)  # RHS

#    # Setup UFL, incorporating Nitsche boundary conditions:
#    stress = 2 * mu * sym(grad(u))
#    F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb + g * rhop * dot(v, rhat) * dx
#    F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb  # Continuity equation

#    # nitsche free slip BCs
#    F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds_tb
#    F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds_tb
#    F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

    approximation = BoussinesqApproximation(1)

    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'un': 0},
    }

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 cartesian=False,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)
    # use tighter tolerances than default to ensure convergence:
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-14
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-12

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    stokes_solver.solve()

    # take out null modes through L2 projection from velocity and pressure
    # removing rotation from velocity:
    rot = as_vector((-X[1], X[0]))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=project_solver_parameters)

    # removing constant nullspace from pressure
    coef = assemble(p_ * dx)/assemble(Constant(1.0)*dx(domain=mesh))
    p_.project(p_ - coef, solver_parameters=project_solver_parameters)

    solution = assess.CylindricalStokesSolutionSmoothFreeSlip(int(float(nn)), int(float(k)), nu=float(mu))

    # compute u analytical and error
    uxy = interpolate(as_vector((X[0], X[1])), V)
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal.dat.data[:] = [solution.velocity_cartesian(xyi) for xyi in uxy.dat.data]
    u_error = Function(V, name="VelocityError").assign(u_-u_anal)

    # compute p analytical and error
    pxy = interpolate(as_vector((X[0], X[1])), Wvec)
    p_anal = Function(W, name="AnalyticalPressure")
    p_anal.dat.data[:] = [solution.pressure_cartesian(xyi) for xyi in pxy.dat.data]
    p_error = Function(W, name="PressureError").assign(p_-p_anal)

    # Write output files in VTK format:
    u_.rename("Velocity")
    p_.rename("Pressure")
    u_file = File("fs_velocity_{}.pvd".format(disc_n))
    p_file = File("fs_pressure_{}.pvd".format(disc_n))

    # Write output:
    u_file.write(u_, u_anal, u_error)
    p_file.write(p_, p_anal, p_error)

    l2anal_u = numpy.sqrt(assemble(dot(u_anal, u_anal)*dx))
    l2anal_p = numpy.sqrt(assemble(dot(p_anal, p_anal)*dx))
    l2error_u = numpy.sqrt(assemble(dot(u_error, u_error)*dx))
    l2error_p = numpy.sqrt(assemble(dot(p_error, p_error)*dx))

    return l2error_u, l2error_p, l2anal_u, l2anal_p


# Run model at different levels:
f = open('errors.log', 'w')
l2error_u, l2error_p, l2anal_u, l2anal_p = model(level)
f.write(f"{l2error_u} {l2error_p} {l2anal_u} {l2anal_p}")
f.close()
