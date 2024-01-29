from .momentum_equation import StokesEquations, FreeSurfaceStokesEquations
from .utility import upward_normal, ensure_constant, InteriorBC
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
        "ksp_max_it": 1000,
        "pc_type": "python",
        "pc_python_type": "gadopt.SPDAssembledPC",
        "assembled_pc_type": "gamg",
        "assembled_mg_levels_pc_type": "sor",
        "assembled_pc_gamg_threshold": 0.01,
        "assembled_pc_gamg_square_graph": 100,
        "assembled_pc_gamg_coarse_eq_limit": 1000,
        "assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
    },
    "fieldsplit_1": {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-4,
        "ksp_max_it": 200,
        "pc_type": "python",
        "pc_python_type": "gadopt.VariableMassInvPC",
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

    def __init__(self, z, T, approximation, bcs=None, mu=1, quad_degree=6,
                 cartesian=True, solver_parameters=None, closed=True,
                 rotational=False, J=None, constant_jacobian=False, **kwargs):

        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.solution = z
        self.solution_old = fd.Function(self.solution)  # This is used for the implicit free surface coupling
        self.T = T
        self.approximation = approximation
        self.mu = ensure_constant(mu)
        self.quad_degree = quad_degree
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)
        self.solver_kwargs = kwargs
        self.k = upward_normal(self.Z.mesh(), cartesian)

        # Note that velocity, pressure and the buoyancy term are added to fields later in initialise().
        # This allows there to be three trial functions when 'splitting' z for the implicit free surface coupling.
        self.fields = {
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'rho_continuity': self.approximation.rho_continuity(),
        }

        self.weak_bcs = {}
        self.strong_bcs = []
        for id, bc in bcs.items():
            weak_bc = {}
            for bc_type, value in bc.items():
                if bc_type == 'u':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0), value, id))
                elif bc_type == 'ux':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(0), value, id))
                elif bc_type == 'uy':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(1), value, id))
                elif bc_type == 'uz':
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0).sub(2), value, id))
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

        # Still need to set up the equations and add trial functions to fields in initialise()
        # to allow for a free surface coupling with an extra trial function
        self._initialised = False

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
        # solver is setup only last minute
        # so people can overwrite parameters we've setup here
        self._solver_setup = False

    def initialise(self):

        # Add velocity, pressure and buoyancy term to the fields dictionary
        u, p = fd.split(self.solution)
        self.fields['velocity'] = u
        self.fields['pressure'] = p
        self.fields['source'] = self.approximation.buoyancy(p, self.T) * self.k

        # Initialise StokesEquations
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=self.quad_degree,
                                         compressible=self.approximation.compressible)

        # Add terms to StokesEquations
        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        self._initialised = True

    def setup_solver(self):
        if self.constant_jacobian:
            z_tri = fd.TrialFunction(self.Z)
            F_stokes_lin = fd.replace(self.F, {self.solution: z_tri})
            a, L = fd.lhs(F_stokes_lin), fd.rhs(F_stokes_lin)
            self.problem = fd.LinearVariationalProblem(a, L, self.solution,
                                                       bcs=self.strong_bcs,
                                                       constant_jacobian=True)
            self.solver = fd.LinearVariationalSolver(self.problem,
                                                     solver_parameters=self.solver_parameters,
                                                     options_prefix=self.name,
                                                     appctx={"mu": self.mu},
                                                     **self.solver_kwargs)
        else:
            self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                          bcs=self.strong_bcs, J=self.J)
            self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                        solver_parameters=self.solver_parameters,
                                                        options_prefix=self.name,
                                                        appctx={"mu": self.mu},
                                                        **self.solver_kwargs)
        self._solver_setup = True

    def solve(self):
        if not self._initialised:
            self.initialise()
        if not self._solver_setup:
            self.setup_solver()

        self.solution_old.assign(self.solution)  # Need to update old solution for implicit free surface

        self.solver.solve()


class FreeSurfaceStokesSolver(StokesSolver):
    name = 'FreeSurfaceStokesSolver'

    def __init__(self, z, T, approximation, free_surface_dt, free_surface_id, bcs=None, mu=1,
                 quad_degree=6, cartesian=True, solver_parameters=None, closed=True,
                 rotational=False, J=None, constant_jacobian=False, **kwargs):

        super().__init__(z, T, approximation, bcs=bcs, mu=mu, quad_degree=quad_degree,
                         cartesian=cartesian, solver_parameters=solver_parameters, closed=closed,
                         rotational=rotational, J=J, constant_jacobian=constant_jacobian, **kwargs)

        self.free_surface_dt = free_surface_dt
        self.free_surface_id = free_surface_id

    def initialise(self):
        # Solution is made up of 3 trial functions so we have a separate initialise()
        u, p, eta = fd.split(self.solution)
        u_old, p_old, eta_old = fd.split(self.solution_old)

        # Theta timestepping scheme for free surface variable
        # Forward Euler (theta = 0), Crank Nicholson (thetat = 0.5), Backward Euler (theta = 1)
        theta = 0.5
        eta_theta = (1-theta)*eta_old + theta*eta

        # Add velocity, pressure and buoyancy term to the fields dictionary
        self.fields['velocity'] = u
        self.fields['pressure'] = p
        self.fields['source'] = self.approximation.buoyancy(p, self.T) * self.k

        # Initialise Stokes equations with a free surface
        self.equations = FreeSurfaceStokesEquations(self.Z, self.Z, quad_degree=self.quad_degree,
                                                    compressible=self.approximation.compressible,
                                                    free_surface_id=self.free_surface_id)

        # Add free surface stress term
        self.weak_bcs[self.free_surface_id] = {'normal_stress': self.approximation.rho * self.approximation.g * eta_theta}

        # Add default Stokes terms to the equation
        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        # Set internal dofs to zero to prevent singular matrix for free surface equation
        self.strong_bcs.append(InteriorBC(self.Z.sub(2), 0, self.free_surface_id))

        # Set free surface mass term
        self.F += self.equations[2].mass_term(self.test[2], (eta-eta_old)/self.free_surface_dt)

        self._initialised = True
