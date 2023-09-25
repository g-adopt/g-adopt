from .momentum_equation import StokesEquations
from .utility import upward_normal, ensure_constant
from .utility import log_level, INFO, DEBUG, depends_on
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
        "assembled_mg_levels_pc_type": "sor",
        "assembled_mg_levels_pc_sor_diagonal_shift": 1e-100,
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
p2p0_stokes_solver_parameters = {
    'mat_type': 'nest',
    'ksp_atol': 1e-10,
    'ksp_converged_reason': None,
    'ksp_max_it': 500,
    'ksp_monitor_true_residual': None,
    'ksp_rtol': 1e-09,
    'ksp_type': 'fgmres',
    'pc_fieldsplit_schur_factorization_type': 'full',
    'pc_fieldsplit_schur_precondition': 'user',
    'pc_fieldsplit_type': 'schur',
    'pc_type': 'fieldsplit',
    'fieldsplit_0': {
        'ksp_convergence_test': 'skip',
        'ksp_max_it': 1,
        'ksp_norm_type': 'unpreconditioned',
        'ksp_richardson_self_scale': False,
        'ksp_type': 'richardson',
        'mg_coarse_assembled': {
            'mat_type': 'aij',
            'pc_telescope_reduction_factor': 1,
            'pc_telescope_subcomm_type': 'contiguous',
            'pc_type': 'telescope',
            'telescope_pc_factor_mat_solver_type': 'superlu_dist',
            'telescope_pc_type': 'lu'
        },
        'mg_coarse_pc_python_type': 'firedrake.AssembledPC',
        'mg_coarse_pc_type': 'python',
        'mg_levels': {
            'ksp_convergence_test': 'skip',
            'ksp_max_it': 6,
            'ksp_norm_type': 'unpreconditioned',
            'ksp_type': 'fgmres',
            'patch_pc_patch_construct_dim': 0,
            'patch_pc_patch_construct_type': 'star',
            'patch_pc_patch_dense_inverse': True,
            'patch_pc_patch_local_type': 'additive',
            'patch_pc_patch_partition_of_unity': False,
            'patch_pc_patch_precompute_element_tensors': True,
            'patch_pc_patch_save_operators': True,
            'patch_pc_patch_statistics': False,
            'patch_pc_patch_sub_mat_type': 'seqdense',
            'patch_pc_patch_symmetrise_sweep': False,
            'patch_sub_ksp_type': 'preonly',
            'patch_sub_pc_factor_mat_solver_type': 'petsc',
            'patch_sub_pc_type': 'lu',
            'pc_python_type': 'firedrake.PatchPC',
            'pc_type': 'python'
        },
        'pc_mg_log': None,
        'pc_mg_type': 'full',
        'pc_type': 'mg'
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_python_type': 'gadopt.P0MassInvPC',
        'pc_type': 'python'
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
    """
    Create nullspace for the mixed Stokes system.

    :arg closed: if closed include constant pressure nullspace
    :arg rotational: if rotational include all rotational modes
    :translations: list of dimensions (0 to dim-1) corresponding to translations to include
    """
    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    V, W = Z.subfunctions
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
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = V

    if closed:
        p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = W

    return fd.MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])


class StokesSolver:
    name = 'Stokes'

    def __init__(self, z, T, approximation, bcs=None, mu=1,
                 quad_degree=6, cartesian=True, solver_parameters=None,
                 closed=True, rotational=False, J=None,
                 gamma=None,
                 **kwargs):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.approximation = approximation
        self.mu = ensure_constant(mu)
        self.gamma = ensure_constant(gamma)
        self.solver_parameters = solver_parameters
        self.J = J
        self.linear = not depends_on(self.mu, self.solution)

        self.solver_kwargs = kwargs
        u, p = fd.split(self.solution)
        self.k = upward_normal(self.Z.mesh(), cartesian)
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(6.25),  # matches C_ip=100. in "old" code for Q2Q1 in 2d
            'source': self.approximation.buoyancy(p, T) * self.k,
            'rho_continuity': self.approximation.rho_continuity(),
        }
        self.appctx = {"mu": self.mu}
        if self.gamma is not None:
            self.fields['gamma'] = self.gamma
            self.appctx['gamma'] = self.gamma

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

            if self.gamma is not None:
                # Augmented Lagrangian only implemted for P2-P0 at the moment
                assert self.Z.sub(0).ufl_element().degree() == 2
                assert self.Z.sub(1).ufl_element().degree() == 0
                # only for isoviscous
                # assert isinstance(self.mu, fd.Constant)
                self.solver_parameters.update(p2p0_stokes_solver_parameters)
            elif self.mesh.topological_dimension() == 2 and cartesian:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)
                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
        # solver is setup only last minute
        # so people can overwrite parameters we've setup here
        self._solver_setup = False

    def setup_solver(self):
        self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                      bcs=self.strong_bcs, J=self.J)
        self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                    solver_parameters=self.solver_parameters,
                                                    options_prefix=self.name,
                                                    appctx=self.appctx,
                                                    **self.solver_kwargs)
        if self.gamma is not None:
            from .mg_transfers import VariablePkP0SchoeberlTransfer, NullTransfer
            V = self.Z.sub(0)
            Q = self.Z.sub(1)
            tdim = self.mesh.topological_dimension()
            hierarchy = "uniform"
            restriction = False
            vtransfer = VariablePkP0SchoeberlTransfer(tdim, hierarchy)
            qtransfer = NullTransfer()
            transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict if restriction else fd.restrict, fd.inject),
                         Q.ufl_element(): (fd.prolong, fd.restrict, qtransfer.inject)}
            transfermanager = fd.TransferManager(native_transfers=transfers)
            self.solver.set_transfer_manager(transfermanager)

        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        self.solver.solve()
