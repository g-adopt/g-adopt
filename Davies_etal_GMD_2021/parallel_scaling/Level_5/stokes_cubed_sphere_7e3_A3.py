from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI  # noqa: F401
import scipy.special
import math

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
rmin, rmax, ref_level, nlayers = 1.22, 2.22, 5, 16

# Construct a CubedSphere mesh and then extrude into a sphere - note that unlike cylindrical case, popping is done internally here:
mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')
bottom_id, top_id = "bottom", "top"
n = FacetNormal(mesh)  # Normals, required for Nusselt number calculation


# Define logging convenience functions:
def log(*args):
    """Log output to stdout from root processor only"""
    PETSc.Sys.Print(*args)


# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
W2d = FunctionSpace(mesh2d, "CG", 1)  # Shell function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.
# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
mu_field = Function(W, name="Viscosity")

# Output function space information:
log("Number of 2D shell mesh DOF:", W2d.dim())
log("Number of Velocity DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
log("Number of Temperature DOF:", Q.dim())

# Timing info:
stokes_stage = PETSc.Log.Stage("stokes_solve")
energy_stage = PETSc.Log.Stage("energy_solve")

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
theta = atan_2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
phi = atan_2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)
k = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)
Told, Tnew = Function(Q, name="OldTemp"), Function(Q, name="NewTemp")
conductive_term = rmin*(rmax - r) / (r*(rmax - rmin))

# evaluate P_lm node-wise using scipy lpmv
l, m, eps_c, eps_s = 3, 2, 0.01, 0.01
Plm = Function(Q, name="P_lm")
cos_phi = interpolate(cos(phi), Q)
Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
if m == 0:
    Plm.assign(Plm/math.sqrt(2))

Told.interpolate(conductive_term +
                 (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))
Tnew.assign(Told)

# Temporal discretisation - Using a Crank-Nicholson scheme where theta = 0.5:
Ttheta = 0.5 * Tnew + (1-0.5) * Told

# Time stepping:
max_timesteps = 20
time = 0.0

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
        "ksp_rtol": 1e-7,
        "ksp_converged_reason": None,
        "ksp_view": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-5,
        "ksp_converged_reason": None,
        "ksp_view": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
    }
}

# Temperature Equation Solver Parameters:
energy_solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-7,
    "ksp_converged_reason": None,
    "ksp_view": None,
    "pc_type": "sor",
}


# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(7e3)  # Rayleigh number
E = Constant(4.605170185988092)  # Activation energy for temperature dependent viscosity.
mu = exp(E * (0.5 - Tnew))
k = as_vector((X[0]/r, X[1]/r, X[2]/r))  # Radial unit vector (in direction opposite to gravity)
C_ip = Constant(100.0)  # Fudge factor for interior penalty term used in weak imposition of BCs
p_ip = 2  # Maximum polynomial degree of the _gradient_ of velocity

# Temperature equation related constants:
delta_t = Constant(5.0e-8)  # Initial time-step
kappa = Constant(1.0)  # Thermal diffusivity

# Stokes equations in UFL form:
stress = 2 * mu * sym(grad(u))
F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb - (dot(v, k) * Ra * Ttheta) * dx
F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb  # Continuity equation

# nitsche free-slip BCs
F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds_tb
F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds_tb
F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx + dot(grad(q), kappa * grad(Ttheta)) * dx

# Temperature boundary conditions
bctb, bctt = DirichletBC(Q, 1.0, bottom_id), DirichletBC(Q, 0.0, top_id)

# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((0, X[2], -X[1])))
y_rotV = Function(V).interpolate(as_vector((-X[2], 0, X[0])))
z_rotV = Function(V).interpolate(as_vector((-X[1], X[0], 0)))
V_nullspace = VectorSpaceBasis([x_rotV, y_rotV, z_rotV])
V_nullspace.orthonormalize()
p_nullspace = VectorSpaceBasis(constant=True)  # Constant nullspace for pressure
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])  # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
nns_x = Function(V).interpolate(Constant([1., 0., 0.]))
nns_y = Function(V).interpolate(Constant([0., 1., 0.]))
nns_z = Function(V).interpolate(Constant([0., 0., 1.]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, nns_z, x_rotV, y_rotV, z_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z)  # velocity BC's handled through Nitsche
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=stokes_solver_parameters, appctx={"mu": mu}, nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, near_nullspace=Z_near_nullspace)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=energy_solver_parameters)

# Now perform the time loop:
for timestep in range(0, max_timesteps):

    time += float(delta_t)

    # Solve Stokes sytem:
    with stokes_stage: stokes_solver.solve()

    # Temperature system:
    with energy_stage: energy_solver.solve()

    # Set Told = Tnew - assign the values of Tnew to Told
    Told.assign(Tnew)
