# Mesh Generation:
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")

---------------------------------------------------------------------------------------------
# Constants, unit vector, initial condition
Ra = Constant(1e5)
r = sqrt(X[0]**2 + X[1]**2)
k = as_vector((X[0], X[1])) / r
Told.interpolate(rmax-r + 0.02*cos(4.*atan_2(X[1],X[0]))*sin((r-rmin)*pi))

---------------------------------------------------------------------------------------------
# UFL for Stokes equations incorporating Nitsche:
C_ip = Constant(100.0) # Fudge factor for interior penalty term used in weak imposition of BCs
p_ip = 2 # Maximum polynomial degree of the _gradient_ of velocity

# Stokes equations in UFL form:
stress = 2 * mu * sym(grad(u))
F_stokes = inner(grad(v), stress) * dx - div(v) * p * dx + dot(n, v) * p * ds_tb - (dot(v, k) * Ra * Ttheta) * dx
F_stokes += -w * div(u) * dx + w * dot(n, u) * ds_tb  # Continuity equation

# nitsche free-slip BCs
F_stokes += -dot(v, n) * dot(dot(n, stress), n) * ds_tb
F_stokes += -dot(u, n) * dot(dot(n, 2 * mu * sym(grad(v))), n) * ds_tb
F_stokes += C_ip * mu * (p_ip + 1)**2 * FacetArea(mesh) / CellVolume(mesh) * dot(u, n) * dot(v, n) * ds_tb

---------------------------------------------------------------------------------------------
# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((-X[1], X[0])))
V_nullspace = VectorSpaceBasis([x_rotV])
V_nullspace.orthonormalize()
p_nullspace = VectorSpaceBasis(constant=True) # Constant nullspace for pressure n
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace]) # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
nns_x  = Function(V).interpolate(Constant([1., 0.]))
nns_y  = Function(V).interpolate(Constant([0., 1.]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, x_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1)])

---------------------------------------------------------------------------------------------
# Updated solve calls:
stokes_problem = NonlinearVariationalProblem(F_stokes, z) # velocity BC's handled through Nitsche
stokes_solver = NonlinearVariationalSolver(stokes_problem, solver_parameters=stokes_solver_parameters, appctx={"mu": mu}, nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, near_nullspace=Z_near_nullspace)
