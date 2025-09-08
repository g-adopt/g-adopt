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


def model(level, nn, do_write=False):
    """The delta-function initial condition, cylindrical domain, zero-slip boundary condition model.

    Args:
        level: refinement level
        nn: wave number
        do_write: whether to output the velocity/pressure fields
    """

    rmin, rmax = 1.22, 2.22
    rp = Constant((rmin + rmax) / 2.)  # height of forcing: delta(r-rp)
    nn = Constant(nn)

    ncells = level * 256
    nlayers = level * 16

    # Construct a circle mesh and then extrude into a cylinder:
    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan2(X[1], X[0])

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # for coordinates used in normal stress solution
    P0 = FunctionSpace(mesh, "DQ", 0)  # used for marker field
    Q1DG = FunctionSpace(mesh, "DQ", 1)  # used for analytical (disc.) pressure solution
    Q1DGvec = VectorFunctionSpace(mesh, "DQ", 1)  # for coordinates used in analytical pressure solution

    # Set up mixed function space and associated test functions:
    Z = MixedFunctionSpace([V, W])
    v, w = TestFunctions(Z)

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)
    u_, p_ = z.subfunctions

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity
    g = Constant(1.0)  # Overall scaling of delta forcing

    approximation = BoussinesqApproximation(1)
    stokes_bcs = {
        boundary.bottom: {'u': 0},
        boundary.top: {'u': 0},
    }

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

    # Add delta forcing as an ad-hoc additional term, applied as "internal" boundary
    # integral over facets where the marker jump from 0 to 1
    marker = Function(P0).interpolate(conditional(r < rp, 1, 0))
    n = FacetNormal(mesh)
    additional_forcing_term = g * cos(nn * phi) * dot(jump(marker, n), avg(v)) * dS_h
    # Use tighter tolerances than default to ensure convergence
    solver_parameters_update = {
        "fieldsplit_0": {"ksp_rtol": 1e-13},
        "fieldsplit_1": {"ksp_rtol": 1e-11},
    }
    stokes_solver = StokesSolver(
        z,
        approximation,
        additional_forcing_term=additional_forcing_term,
        bcs=stokes_bcs,
        solver_parameters_update=solver_parameters_update,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
    )

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.
    stokes_solver.solve()

    # take out null modes through L2 projection from velocity and pressure
    # removing rotation from velocity:
    rot = as_vector((-X[1], X[0]))
    coef = assemble(dot(rot, u_)*dx) / assemble(dot(rot, rot)*dx)
    u_.project(u_ - rot*coef, solver_parameters=_project_solver_parameters)

    # removing constant nullspace from pressure
    coef = assemble(p_ * dx)/assemble(Constant(1.0)*dx(domain=mesh))
    p_.project(p_ - coef, solver_parameters=_project_solver_parameters)

    # calculating surface normal stress given the solution of the stokes problem
    ns_ = stokes_solver.force_on_boundary(boundary.top)

    solution_upper = assess.CylindricalStokesSolutionDeltaZeroSlip(float(nn), +1, nu=float(mu))
    solution_lower = assess.CylindricalStokesSolutionDeltaZeroSlip(float(nn), -1, nu=float(mu))

    # compute u analytical and error
    uxy = Function(V).interpolate(as_vector((X[0], X[1])))
    u_anal_upper = Function(V, name="AnalyticalVelocityUpper")
    u_anal_lower = Function(V, name="AnalyticalVelocityLower")
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal_upper.dat.data[:] = [solution_upper.velocity_cartesian(xyi) for xyi in uxy.dat.data]
    u_anal_lower.dat.data[:] = [solution_lower.velocity_cartesian(xyi) for xyi in uxy.dat.data]
    u_anal.interpolate(marker*u_anal_lower + (1-marker)*u_anal_upper)
    u_error = Function(V, name="VelocityError").assign(u_-u_anal)

    # compute p analytical and error
    pxy = Function(Q1DGvec).interpolate(as_vector((X[0], X[1])))
    pdg = Function(Q1DG).interpolate(p)
    p_anal_upper = Function(Q1DG, name="AnalyticalPressureUpper")
    p_anal_lower = Function(Q1DG, name="AnalyticalPressureLower")
    p_anal = Function(Q1DG, name="AnalyticalPressure")
    p_anal_upper.dat.data[:] = [solution_upper.pressure_cartesian(xyi) for xyi in pxy.dat.data]
    p_anal_lower.dat.data[:] = [solution_lower.pressure_cartesian(xyi) for xyi in pxy.dat.data]
    p_anal.interpolate(marker*p_anal_lower + (1-marker)*p_anal_upper)
    p_error = Function(Q1DG, name="PressureError").assign(pdg-p_anal)

    # compute ns_ analytical and error (note we are using the same space as pressure)
    nsxy = Function(Wvec).interpolate(as_vector((X[0], X[1])))
    ns_anal_upper = Function(W, name="AnalyticalNormalStressUpper")
    ns_anal_lower = Function(W, name="AnalyticalNormalStressLower")
    ns_anal_upper.dat.data[:] = [-solution_upper.radial_stress_cartesian(xyi) for xyi in nsxy.dat.data]
    ns_anal_lower.dat.data[:] = [-solution_lower.radial_stress_cartesian(xyi) for xyi in nsxy.dat.data]
    ns_anal = Function(W, name="AnalyticalNormalStress").interpolate(marker * ns_anal_lower + (1 - marker) * ns_anal_upper)
    ns_error = Function(W, name="NormalStressError").assign(ns_ - ns_anal)

    if do_write:
        # Write output files in VTK format:
        u_.rename("Velocity")
        p_.rename("Pressure")
        u_file = VTKFile("fs_velocity_{}.pvd".format(level))
        p_file = VTKFile("fs_pressure_{}.pvd".format(level))
        ns_file = VTKFile("fs_normalstress_{}.pvd".format(level))

        # Write output:
        u_file.write(u_, u_anal, u_error)
        p_file.write(p_, p_anal, p_error)
        ns_file.write(ns_, ns_anal, ns_error)

    l2anal_u = numpy.sqrt(assemble(dot(u_anal, u_anal)*dx))
    l2anal_p = numpy.sqrt(assemble(dot(p_anal, p_anal)*dx))
    l2anal_ns = numpy.sqrt(assemble(dot(ns_anal, ns_anal) * ds_t))
    l2error_u = numpy.sqrt(assemble(dot(u_error, u_error)*dx))
    l2error_p = numpy.sqrt(assemble(dot(p_error, p_error)*dx))
    l2error_ns = numpy.sqrt(assemble(dot(ns_error, ns_error) * ds_t))

    return l2error_u, l2error_p, l2error_ns, l2anal_u, l2anal_p, l2anal_ns
