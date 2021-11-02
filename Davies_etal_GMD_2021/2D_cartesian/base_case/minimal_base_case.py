from firedrake import *

# Mesh - use a built in meshing function:
mesh = UnitSquareMesh(40, 40, quadrilateral=True)

# Function spaces:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space

# Test functions and functions to hold solutions:
v, w = TestFunctions(Z)
q = TestFunction(Q)
z = Function(Z)
u, p = split(z)  # Returns symbolic UFL expression for u and p
Told, Tnew = Function(Q, name="OldTemp"), Function(Q, name="NewTemp")
Ttheta = 0.5 * Tnew + 0.5 * Told  # Temporal discretisation through Crank-Nicholson

# Initialise temperature field:
X = SpatialCoordinate(mesh)
Told.interpolate(1.0 - X[1] + 0.05 * cos(pi * X[0]) * sin(pi * X[1]))
Tnew.assign(Told)

# Important constants:
Ra = Constant(1e4)  # Rayleigh number
mu = Constant(1.0)  # Viscosity - constant for this isoviscous case
kappa = Constant(1.0)  # Thermal diffusivity
delta_t = Constant(1e-4)  # Time-step
k = Constant((0, 1))  # Unit vector (in direction opposite to gravity)

# Stokes equations in UFL form:
stress = 2 * mu * sym(grad(u))
F_stokes = inner(grad(v), stress) * dx + dot(v, grad(p)) * dx - (dot(v, k) * Ra * Ttheta) * dx
F_stokes += dot(grad(w), u) * dx  # Continuity equation
# Energy equation in UFL form:
F_energy = q * (Tnew - Told) / delta_t * dx + q * dot(u, grad(Ttheta)) * dx + dot(grad(q), kappa * grad(Ttheta)) * dx

# Set up boundary conditions and deal with nullspaces:
bcvx, bcvy = DirichletBC(Z.sub(0).sub(0), 0, (1, 2)), DirichletBC(Z.sub(0).sub(1), 0, (3, 4))
bctb, bctt = DirichletBC(Q, 1.0, 3), DirichletBC(Q, 0.0, 4)
p_nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# Initialise output:
output_file = File('output.pvd')  # Create output file
u, p = z.split()
u.rename("Velocity"), p.rename("Pressure")

# Solver dictionary:
solver_parameters = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Setup problem and solver objects so we can reuse (cache) solver setup
stokes_problem = NonlinearVariationalProblem(F_stokes, z, bcs=[bcvx, bcvy])
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=solver_parameters, nullspace=p_nullspace, transpose_nullspace=p_nullspace)
energy_problem = NonlinearVariationalProblem(F_energy, Tnew, bcs=[bctb, bctt])
energy_solver = NonlinearVariationalSolver(energy_problem, solver_parameters=solver_parameters)

for timestep in range(0, 1000):  # Perform time loop for 1000 steps
    if timestep % 10 == 0:
        output_file.write(u, p, Tnew)
    stokes_solver.solve()
    energy_solver.solve()
    Told.assign(Tnew)
