r""" This is a script to solve the Poisson equation for gravity.

The strong form of the equation is given by:
        $$ -\nabla^2 \phi = 4 R_earth \pi G \rho $$
To non-dimensionalise this equation, we choose charactristic scales of length, potential, and density:
        $$ L = R_earth, \quad \rho_0 = 1000, \quad \phi_0 = G \rho_0 L^2 $$
Then in non-dimensional form the equation becomes:
        $$ - \nabla^2 \phi = 4 \pi \rho $$
For the weak formulation we have:
        $$ \int_{V} (\nabla^2 \phi) v \,dx = \int_{V} 4 \pi \rho v \,dx $$
Using integration by parts:
        $$ \int_{V} \nabla v \cdot \nabla \phi \,dx - \int_{V} v \nabla \phi \cdot n \,ds = \int_{V} 4 \pi \rho v \,dx $$
"""
from firedrake import *

# A special mesh generated just for this purpose (see mesh.geo)
mesh = Mesh("mesh.msh")

r_surface = 2.22
r_cmb = 1.22

X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)

# Define the function space, trial and test functions
V = FunctionSpace(mesh, "CG", 1)
phi = TrialFunction(V)
v = TestFunction(V)
solution = Function(V, name="phi")

# For the forcing term we use DG to capture the density jump
Q = FunctionSpace(mesh, "DG", 0)
rho = Function(Q, name="rho").interpolate(conditional(r < r_surface, conditional(r > r_cmb, 1.0, 0.0), 0))

bcs = [DirichletBC(V, 0.0, 4)]

L = inner(grad(v), grad(phi)) * dx  # - v * dot(grad(phi), FacetNormal(mesh)) * ds(4)
R = 4 * pi * rho * v * dx
# solve(L == R, solution, bcs=bcs)
#solve(L == R, solution, bcs=bcs)
p = LinearVariationalProblem(L,R,solution,bcs=bcs)
s = LinearVariationalSolver(p)
s.solve()

fi = VTKFile("poisson.pvd")
fi.write(solution, rho)
