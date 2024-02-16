from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
import numpy
import assess

# Quadrature degree:
_dx = dx(degree=6)
# Projection solver parameters for nullspaces:
_project_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}


def model(level, k, nn, dt_factor, do_write=True):
    """The smooth initial condition, cylindrical domain, free-slip boundary condition model.

    Args:
        level: refinement level
        k: radial degree
        nn: wave number per radial degree
        do_write: whether to output the velocity/pressure fields
    """

    rmin, rmax = 1.22, 2.22
    nn = k * nn

    k = Constant(k)
    nn = Constant(nn)

    ncells = level * 64
    nlayers = level * 8

    # Construct a circle mesh and then extrude into a cylinder:
    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")
    bottom_id, top_id = "bottom", "top"

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan2(X[1], X[0])

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # vector version of W, used for coordinates in anal. pressure solution
    Z = MixedFunctionSpace([V, W, W])  # Mixed function space.
    v, w, w1 = TestFunctions(Z)

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p, eta = split(z)
    u_, p_, eta_ = z.subfunctions

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity

    T = -r**k/rmax**k*cos(nn*phi)  # RHS

    approximation = BoussinesqApproximation(1)
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'free_surface': {}},
    }

    rho0 = approximation.rho
    g = approximation.g
    tau0 = Constant(2 * nn * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)
    dt = Constant(dt_factor*tau0)  # timestep (dimensionless)
    log("dt (dimensionless)", dt)

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, cartesian=False, free_surface_dt=dt,
                                 free_surface_variable_rho=False)

    # use tighter tolerances than default to ensure convergence:
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-13
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-11

    time = Constant(0.0)
    max_timesteps = round(20*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.

    # Modify ds to work with extruded mesh
    ds = CombinedSurfaceMeasure(mesh, degree=6)

    if do_write:
        # Write output files in VTK format:
        u_.rename("Velocity")
        p_.rename("Pressure")
        eta_.rename("Eta")
        u_file = File("fs_velocity_{}.pvd".format(level))
        p_file = File("fs_pressure_{}.pvd".format(level))
        eta_file = File("fs_eta_{}.pvd".format(level))
        temp_file = File("fs_temp_{}.pvd".format(level))
        temp_file.write(Function(W, name="T").interpolate(T), Function(W, name="Density").interpolate(approximation.rho_field(0, T)))

    solution = assess.CylindricalStokesSolutionSmoothFreeSlip(int(float(nn)), int(float(k)), nu=float(mu))

    # compute u analytical and error
    uxy = interpolate(as_vector((X[0], X[1])), V)
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal.dat.data[:] = [solution.velocity_cartesian(xyi) for xyi in uxy.dat.data]

    # compute p analytical and error
    pxy = interpolate(as_vector((X[0], X[1])), Wvec)
    p_anal = Function(W, name="AnalyticalPressure")
    p_anal.dat.data[:] = [solution.pressure_cartesian(xyi) for xyi in pxy.dat.data]

    # compute eta analytical and error
    etaxy = interpolate(as_vector((X[0], X[1])), Wvec)
    eta_anal = Function(W, name="AnalyticalEta")
    eta_anal.dat.data[:] = [-solution.radial_stress_cartesian(xyi) for xyi in etaxy.dat.data]

    # Now perform the time loop:
    for timestep in range(1, max_timesteps+1):

        # Solve Stokes sytem:
        stokes_solver.solve()
        time.assign(time + dt)

        u_error = Function(V, name="VelocityError").assign(u_-u_anal)
        p_error = Function(W, name="PressureError").assign(p_-p_anal)
        eta_error = Function(W, name="EtaError").assign(eta_-eta_anal)

        if do_write:
            # Write output:
            u_file.write(u_, u_anal, u_error)
            p_file.write(p_, p_anal, p_error)
            eta_file.write(eta_, eta_anal, eta_error)

    # take out null modes through L2 projection from velocity and pressure
    # removing rotation from velocity:
    rot = as_vector((-X[1], X[0]))
    coef = assemble(dot(rot, u_)*_dx) / assemble(dot(rot, rot)*_dx)
    u_.project(u_ - rot*coef, solver_parameters=_project_solver_parameters)

    # removing constant nullspace from pressure
    coef = assemble(p_ * _dx)/assemble(Constant(1.0)*_dx(domain=mesh))
    p_.project(p_ - coef, solver_parameters=_project_solver_parameters)

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

    # compute eta analytical and error
    etaxy = interpolate(as_vector((X[0], X[1])), Wvec)
    eta_anal = Function(W, name="AnalyticalEta")
    eta_anal.dat.data[:] = [-solution.radial_stress_cartesian(xyi) for xyi in etaxy.dat.data]
    eta_error = Function(W, name="EtaError").assign(eta_-eta_anal)

    l2anal_u = numpy.sqrt(assemble(dot(u_anal, u_anal)*_dx))
    l2anal_p = numpy.sqrt(assemble(dot(p_anal, p_anal)*_dx))
    l2anal_eta = numpy.sqrt(assemble(dot(eta_anal, eta_anal)*ds(top_id)))
    l2error_u = numpy.sqrt(assemble(dot(u_error, u_error)*_dx))
    l2error_p = numpy.sqrt(assemble(dot(p_error, p_error)*_dx))
    l2error_eta = numpy.sqrt(assemble(dot(eta_error, eta_error)*ds(top_id)))

    return l2error_u, l2error_p, l2error_eta, l2anal_u, l2anal_p, l2anal_eta


if __name__ == "__main__":
    # default case run with nx = 80 for four dt factors
    dt_factors = 2 / (2**np.arange(2))
    levels = [2**i for i in [1, 2]]
    # Rerun with iterative solvers
    errors = np.array([model(l, 2, 4, 1) for l in levels])
    # errors = np.array([model(3, 2, 4, dtf) for dtf in dt_factors])
    log(errors)
    log("u errors", errors[:, 0])
    # use the highest resolution analytical solutions as the reference in scaling
    ref = errors[:, 0][-1]
    relative_errors = errors[:, 0] / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    log("u convergence:", convergence)

    log("p errors", errors[:, 1])
    ref = errors[:, 1][-1]
    relative_errors = errors[:, 1] / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    log("p convergence:", convergence)

    log("eta errors", errors[:, 2])
    ref = errors[:, 2][-1]
    relative_errors = errors[:, 2] / ref
    convergence = np.log2(relative_errors[:-1] / relative_errors[1:])
    log("eta convergence:", convergence)
