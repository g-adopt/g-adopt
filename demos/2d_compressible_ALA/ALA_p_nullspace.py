from gadopt import *
# Set up geometry:
mesh = UnitSquareMesh(40, 40, quadrilateral=True)  # Square mesh generated via firedrake

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Function spaces
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Compressible reference fields

# Compressible reference state:
X = SpatialCoordinate(mesh)
T0 = Constant(0.091)  # Non-dimensional surface temperature
Di = Constant(0.5)  # Dissipation number.
Ra = Constant(1e5)  # Rayleigh number
gruneisen = 1.0
rhobar = Function(Q, name="CompRefDensity").interpolate(exp(((1.0 - X[1]) * Di) / gruneisen))
Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - X[1]) * Di) - T0)
# why do we have these as functions?
alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

ala = AnelasticLiquidApproximation(Ra, Di, rho=rhobar, Tbar=Tbar, alpha=alphabar, chi=chibar, cp=cpbar)

q = TestFunction(W)
p = Function(W, name="pressure_nullspace")
bc = DirichletBC(W, 1., top_id)
r"""
Trying to solve the equation:
    $$ - \nabla p + g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p = 0 $$
Taking the divergence:
    $$ - \nabla \cdot(\nabla p) + \nabla \cdot(g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p) $$
And then testing it with q:
    $$ - q * \nabla \cdot(\nabla p) * dx + q * \nabla \cdot(g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p) * dx $$
Doing integration by parts:
    $$ - \nabla \cdot( q * \nabla(p)) * dx + \nabla(q). \nabla(p) * dx + \nabla \cdot(q * g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p) * dx - \nabla(q) . (g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p) * dx $$

Dropping the boundary condition terms:
    $$ \nabla(q) \cdot \nabla(p) * dx - \nabla q \cdot (g * Di * \rho * \chi * cp/(cv * \gamma) * \hat{k} * p) * dx $$
"""
F = inner(grad(q), grad(p)) * dx - inner(grad(q), as_vector((0, 1)) * ala.Di * ala.cp0 / ala.cv0 / ala.gamma0 * ala.g * ala.rho * ala.chi * p) * dx

solve(F == 0, p, bcs=bc)

vtk_file = VTKFile("output.pvd")
vtk_file.write(p)
