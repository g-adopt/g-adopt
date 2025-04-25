from firedrake import *
from firedrake.adjoint import *

continue_annotation()

solver_parameters = {
    "snes_type": "newtonls",
    "snes_converged_reason": None,
    "snes_max_it": 50,
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-9,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    "gadopt_test": None,
}

mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
v = TestFunction(V)
u0 = Function(V)
u1 = Function(V)

ui = Function(R, val=2.0)
c = Control(ui)
u0.assign(ui)
F = dot(v, (u1 - u0)) * dx - dot(v, u0 * u1) * dx
problem = NonlinearVariationalProblem(F, u1)
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()
u0.assign(u1)
solver.solve()
J = assemble(dot(u1, u1) * dx)
rf = ReducedFunctional(J, c)
print("BEGIN REDUCEDFUNCTIONAL CALL")
rf([ui])
