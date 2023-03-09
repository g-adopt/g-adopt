from gadopt import *
import sys
import numpy
import assess

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry and key parameters:
rmin, rmax = 1.22, 2.22
rp = Constant((rmin + rmax) / 2.)  # height of forcing: delta(r-rp)
nn = Constant(int(sys.argv[1]))  # wave number (n is already used for FacetNormal)
level = int(sys.argv[2])  # refinement level

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
    T = Constant(0.0)

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

    # add delta forcing as ad-hoc aditional term
    # forcing is applied as "internal" boundary integral over facets
    # where the marker jump from 0 to 1
    marker = Function(P0)
    marker.interpolate(conditional(r < rp, 1, 0))
    n = FacetNormal(mesh)
    stokes_solver.F += g * cos(nn*phi) * dot(jump(marker, n), avg(v)) * dS_h

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
