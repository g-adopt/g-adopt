from .time_stepper import TimeIntegratorBase
from .momentum_equation import StokesEquations
from .utility import upward_normal, ensure_constant
from .utility import log_level, INFO, DEBUG
import firedrake as fd

iterative_stokes_solver_parameters = {
    "mat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_type": "full",
    "fieldsplit_0": {
        "ksp_type": "cg",
        "ksp_rtol": 1e-5,
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-4,
        "pc_type": "python",
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_ksp_rtol": 1e-5,
        "Mp_ksp_type": "cg",
        "Mp_pc_type": "sor",
    }
}

direct_stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
}

def create_stokes_nullspace(Z, closed=True, rotational=False, translations=None):
    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    V, W = Z.split()
    if rotational:
        if dim == 2:
            rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(V).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(V).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(V).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []
    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(V).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    if closed:
        p_nullspace = fd.VectorSpaceBasis(constant=True)
    else:
        p_nullspace = W

    return fd.MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])


class StokesSolver:
    name = 'Stokes'

    def __init__(self, z, T, delta_t, bcs=None, Ra=1, g=1, mu=1,
                 quad_degree=6, cartesian=True, solver_parameters=None,
                 closed=True, rotational=False,
                 **kwargs):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree)
        self.solution = z
        self.delta_t = ensure_constant(delta_t)
        self.Ra = ensure_constant(Ra)
        self.g = ensure_constant(g)
        self.mu = ensure_constant(mu)
        self.solver_parameters = solver_parameters
        self.linear = True

        self.solver_kwargs = kwargs
        u, p = fd.split(self.solution)
        self.k = upward_normal(self.Z.mesh(), cartesian)
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(6.25),  # matches C_ip=100. in "old" code for Q2Q1 in 2d
            'source': self.Ra * self.g * T * self.k,
        }

        self.weak_bcs = {}
        self.strong_bcs = []
        for id, bc in bcs.items():
            weak_bc = {}
            for type, value in bc.items():
                if type == 'u':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0), value, id))
                elif type == 'ux':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(0), value, id))
                elif type == 'uy':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(1), value, id))
                elif type == 'uz':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(2), value, id))
                else:
                    weak_bc[type] = value
            self.weak_bcs[id] = weak_bc

        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if self.solver_parameters is None:
            if self.linear:
                self.solver_parameters = {"snes_type": "ksponly"}
            else:
                self.solver_parameters = newton_stokes_solver_parameters.copy()
            if INFO >= log_level:
                self.solver_parameters['snes_monitor'] = None

            if self.mesh.topological_dimension() == 2 and cartesian:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)
                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None

        self.problem = fd.NonlinearVariationalProblem(self.F, self.solution, bcs=self.strong_bcs)
        self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                           solver_parameters=self.solver_parameters,
                                                           options_prefix=self.name,
                                                           appctx={"mu": self.mu},
                                                           **self.solver_kwargs)
        self._initialized = True

    def solve(self):
        self.solver.solve()
