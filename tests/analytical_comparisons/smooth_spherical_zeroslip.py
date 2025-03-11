from gadopt import *
import numpy
import assess

# Projection solver parameters for nullspaces:
_project_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}


def model(level, l, mm, k, do_write=False):
    """The smooth initial condition, spherical domain, zero-slip boundary condition model.

    Args:
        level: refinement level
        l: spherical harmonic degree
        mm: ratio of spherical harmonic degree to spherical harmonic order
        k: radial component of forcing term
        do_write: whether to output the velocity/pressure fields
    """

    rmin, rmax = 1.22, 2.22
    m = l // mm

    l = Constant(l)
    m = Constant(m)
    k = Constant(k)

    nlayers = 2 ** level

    # Construct a cubed sphere mesh and then extrude into a sphere:
    mesh2d = CubedSphereMesh(radius=rmin, refinement_level=level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type="radial")
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    # Define geometric quantities
    X = SpatialCoordinate(mesh)

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    Vscl = FunctionSpace(mesh, "CG", 2)  # scalar version of V, used for rho'
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # vector version of W, used for coordinates in anal. pressure solution
    Z = MixedFunctionSpace([V, W])
    v, w = TestFunctions(Z)

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)
    u_, p_ = z.subfunctions

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity

    # RHS provided by assess solution: rho'=r**k Y_lm
    solution = assess.SphericalStokesSolutionSmoothZeroSlip(float(l), float(m), float(k), nu=float(mu), Rp=rmax, Rm=rmin, g=1.0)
    u_xyz = Function(V).interpolate(X)
    rhop = Function(Vscl)
    rhop.dat.data[:] = [solution.delta_rho_cartesian(xyzi) for xyzi in u_xyz.dat.data]

    T = -rhop  # RHS

    approximation = BoussinesqApproximation(1)
    stokes_bcs = {
        boundary.bottom: {'u': 0},
        boundary.top: {'u': 0},
    }

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)
    # use tighter tolerances than default to ensure convergence:
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-10
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-9

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    stokes_solver.solve()

    # take out null modes through L2 projection from velocity and pressure
    # removing rotation from velocity:
    rot = as_vector((0, X[2], -X[1]))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=_project_solver_parameters)
    rot = as_vector((-X[2], 0, X[0]))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=_project_solver_parameters)
    rot = as_vector((-X[1], X[0], 0))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=_project_solver_parameters)

    # removing constant nullspace from pressure
    coef = assemble(p_ * dx)/assemble(Constant(1.0)*dx(domain=mesh))
    p_.project(p_ - coef, solver_parameters=_project_solver_parameters)

    # compute u analytical and error
    uxzy = Function(V).interpolate(as_vector((X[0], X[1], X[2])))
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal.dat.data[:] = [solution.velocity_cartesian(xyzi) for xyzi in uxzy.dat.data]
    u_error = Function(V, name="VelocityError").assign(u_-u_anal)

    # compute p analytical and error
    pxyz = Function(Wvec).interpolate(as_vector((X[0], X[1], X[2])))
    p_anal = Function(W, name="AnalyticalPressure")
    p_anal.dat.data[:] = [solution.pressure_cartesian(xyzi) for xyzi in pxyz.dat.data]
    p_error = Function(W, name="PressureError").assign(p_-p_anal)

    if do_write:
        # Write output files in VTK format:
        u_.rename("Velocity")
        p_.rename("Pressure")
        u_file = VTKFile("zs_velocity_{}.pvd".format(level))
        p_file = VTKFile("zs_pressure_{}.pvd".format(level))

        # Write output:
        u_file.write(u_, u_anal, u_error)
        p_file.write(p_, p_anal, p_error)

    l2anal_u = numpy.sqrt(assemble(dot(u_anal, u_anal)*dx))
    l2anal_p = numpy.sqrt(assemble(dot(p_anal, p_anal)*dx))
    l2error_u = numpy.sqrt(assemble(dot(u_error, u_error)*dx))
    l2error_p = numpy.sqrt(assemble(dot(p_error, p_error)*dx))

    return l2error_u, l2error_p, l2anal_u, l2anal_p
