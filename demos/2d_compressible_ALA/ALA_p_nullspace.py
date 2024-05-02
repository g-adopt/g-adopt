from gadopt import *
from gadopt.stokes_integrators import ala_right_nullspace
# Set up geometry:
mesh = UnitSquareMesh(40, 40, quadrilateral=True)  # Square mesh generated via firedrake

left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Function spaces

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

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

p = ala_right_nullspace(W, approximation=ala, top_subdomain_id=top_id)

vtk_file = VTKFile("output.pvd")
vtk_file.write(p)
