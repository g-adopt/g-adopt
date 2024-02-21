from .momentum_equation import StokesEquations
from .free_surface_equation import FreeSurfaceEquation
from .viscoelastic_equation import ViscoelasticEquations
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

    def __init__(self, z, T, approximation, bcs=None, mu=1,
                 quad_degree=6, cartesian=True, solver_parameters=None,
                 closed=True, rotational=False, J=None, constant_jacobian=False,
                 iterative_2d=False, free_surface_dt=None, free_surface_variable_rho=True,
                 equations=StokesEquations, **kwargs):

        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = equations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.T = T
        self.approximation = approximation

        self.mu = ensure_constant(mu)
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)

        self.free_surface_dt = free_surface_dt
        self.free_surface_variable_rho = free_surface_variable_rho

        self.solver_kwargs = kwargs
        self.stokes_vars = fd.split(self.solution)  # For default stokes this is a tuple of (velocity, pressure)
        self.k = upward_normal(self.Z.mesh(), cartesian)
        
        self.setup_fields()

        # Setup boundary conditions
        self.weak_bcs = {}
        self.strong_bcs = []
        self.free_surface_dict = {}  # Separate dictionary for copying free surface information
        self.free_surface = False
        
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
                elif bc_type == 'free_surface':
                    # Copy free surface information to a new dictionary
                    self.free_surface_dict[id] = value
                    self.free_surface = True
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

        if self.free_surface:
            self.setup_free_surface()

        # Add terms to Stokes Equations
        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if self.free_surface:
            for i in range(len(eta)):
                # Add free surface time derivative term
                # Multiply by theta to keep the block system symmetric for the implicit coupling case
                # (N.b. we already have two equations from StokesEquations)
                self.F += self.equations[2+i].mass_term(self.test[2+i], theta*(eta[i]-eta_old[i])/free_surface_dt)

        if self.solver_parameters is None:
            if self.linear:
                self.solver_parameters = {"snes_type": "ksponly"}
            else:
                self.solver_parameters = newton_stokes_solver_parameters.copy()
            if INFO >= log_level:
                self.solver_parameters['snes_monitor'] = None

            if self.mesh.topological_dimension() == 2 and cartesian and not iterative_2d:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)
                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None

                if self.free_surface:
                    # Merge free surface fields with pressure field for Schur complement solve
                    self.solver_parameters.update({'pc_fieldsplit_0_fields': '0',
                                                   'pc_fieldsplit_1_fields': '1,'+','.join(str(2+i) for i in range(len(eta)))})
        # solver is setup only last minute
        # so people can overwrite parameters we've setup here
        self._solver_setup = False


    def setup_fields(self):
        # Add velocity, pressure and buoyancy term to the fields dictionary
        self.fields = {
            'velocity': self.stokes_vars[0],
            'pressure': self.stokes_vars[1],
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'source': self.approximation.buoyancy(self.stokes_vars[1], self.T) * self.k,
            'rho_continuity': self.approximation.rho_continuity(),
        }



    def setup_free_surface(self):
    
        if self.free_surface_dt is None:
            raise TypeError("Please provide a timestep to advance the free surface, currently free_surface_dt=None.")

        self.solution_old = fd.Function(self.solution)
        eta = []
        eta_old = []
        eta_theta = []
        self.free_surface_id_list = []

        c = 0  # Counter for free surfaces (N.b. we already have two equations from StokesEquations)
        for id, value in self.free_surface_dict.items():
            if 'exterior_density' in value:
                exterior_density = value['exterior_density']
            else:
                exterior_density = 0

            # Define free surface variables for timestepping
            eta.append(self.stokes_vars[2+c])
            eta_old.append(fd.split(self.solution_old)[2+c])
            theta = 0.5  # This gives a second order in time integration scheme
            eta_theta.append((1-theta)*eta_old[c] + theta*eta[c])

            # Add free surface equation
            self.equations.append(FreeSurfaceEquation(self.Z.sub(2+c), self.Z.sub(2+c), quad_degree=quad_degree,
                                  free_surface_id=id, free_surface_dt=free_surface_dt, theta=theta, k=self.k))
            # Add free surface stress term
            if free_surface_variable_rho:
                # Use actual density
                surface_rho = self.approximation.rho_field(self.stokes_vars[1], T)
            else:
                # Use reference density (needed for analytical cylindrical cases)
                surface_rho = self.approximation.rho
            self.weak_bcs[id] = {'normal_stress': (surface_rho - exterior_density) * self.approximation.g * eta_theta[c]}

            # Set internal dofs to zero to prevent singular matrix for free surface equation
            self.strong_bcs.append(InteriorBC(self.Z.sub(2+c), 0, id))

            self.free_surface_id_list.append(id)

            c += 1

    def setup_solver(self):
        appctx = {"mu": self.mu,
                  "dx": self.equations[0].dx}

        if self.free_surface:
            appctx["free_surface_id_list"] = self.free_surface_id_list
            appctx["ds"] = self.equations[2].ds

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
                                                     appctx=appctx,
                                                     **self.solver_kwargs)
        else:
            self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                          bcs=self.strong_bcs, J=self.J)
            self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                        solver_parameters=self.solver_parameters,
                                                        options_prefix=self.name,
                                                        appctx=appctx,
                                                        **self.solver_kwargs)

        self._solver_setup = True

    def solve(self):
        if not self._solver_setup:
            self.setup_solver()
        if self.free_surface:
            self.solution_old.assign(self.solution)  # Need to update old solution for implicit free surface
        self.solver.solve()


class ViscoelasticStokesSolver(StokesSolver):
    name = "ViscoelasticStokesSolver"


    def __init__(self, z, viscosity, shear_modulus, density, deviatoric_stress, displacement, approximation, dt, bcs=None,
                 quad_degree=6, cartesian=True, solver_parameters=None,
                 closed=True, rotational=False, J=None, constant_jacobian=False,
                 iterative_2d=False, free_surface_dt=None, free_surface_variable_rho=True,
                 **kwargs):

        self.density = density
        self.deviatoric_stress = deviatoric_stress  # Temporary function to store deviatoric stress from previous time step
        self.displacement = displacement

        
        maxwell_time = viscosity / shear_modulus
        self.effective_viscosity = self.effective_viscosity(viscosity, maxwell_time, dt)
        self.prefactor_prestress = self.prefactor_prestress(viscosity, maxwell_time, dt)
        
        self.previous_stress = fd.Function(deviatoric_stress, name="previous stress").interpolate(self.prefactor_prestress * self.deviatoric_stress)  # History stress term from previous time step (explicit RHS forcing)


        super().__init__(z, self.density, approximation, bcs=bcs, mu=self.effective_viscosity,
                 quad_degree=quad_degree, cartesian=cartesian, solver_parameters=solver_parameters,
                 closed=closed, rotational=rotational, J=J, constant_jacobian=constant_jacobian,
                 iterative_2d=iterative_2d, free_surface_dt=free_surface_dt, free_surface_variable_rho=free_surface_variable_rho,
                 equations=ViscoelasticEquations, **kwargs)
        
        scale_mu = fd.Constant(1e10)  # this is a scaling factor roughly size of mantle maxwell time to make sure that solve converges with strong bcs in parallel...
        self.F = (1/ scale_mu)*self.F

    def effective_viscosity(self, viscosity, maxwell_time, dt):
        return viscosity / (maxwell_time + dt / 2)

    def prefactor_prestress(self, viscosity, maxwell_time, dt):
        return (maxwell_time - dt / 2) / (maxwell_time + dt / 2)


    def setup_fields(self):
        super().setup_fields()
        # Add additional viscoelastic fields 
        self.fields["displacement"] = self.displacement
        self.fields["previous_stress"] = self.previous_stress

    def setup_free_surface(self):
        # Overload method 
        for id, value in self.free_surface_dict.items():
            if 'exterior_density' in value:
                print("hello exterior density")
                exterior_density = value['exterior_density']
            else:
                exterior_density = 0
            # First, make the displacement term implicit by incorporating 
            # the unknown `incremental displacement' (stokes_vars[0]) that 
            # we are solving for
            implicit_displacement = self.stokes_vars[0] + self.displacement
            implicit_displacement_up = fd.dot(implicit_displacement, self.k)
            # Add free surface stress term. This is also referred to as the Hydrostatic Prestress advection term in the GIA literature. 
            normal_stress = (self.density - exterior_density) * self.approximation.g * implicit_displacement_up 
            if 'normal_stress' in self.weak_bcs[id]:
                # Usually there will be also an ice/water loadi acting as a normal stress in the GIA problem
                print("hello normal stress")
                existing_value = self.weak_bcs[id]['normal_stress']
                self.weak_bcs[id]['normal_stress'] = existing_value + normal_stress
            else:
                self.weak_bcs[id] = {'normal_stress': normal_stress}
        
        # Turn off free surface flag because the viscoelastic free surface (for the small displacement approximation) is now setup
        self.free_surface = False

    def solve(self):
        super().solve()

        # Update deviatoric stress for the timestep that has just been solved for
        self.deviatoric_stress.interpolate(2 * self.effective_viscosity * fd.sym(fd.grad(self.stokes_vars[0])) + self.prefactor_prestress * self.deviatoric_stress)
        # Update history stress term for using as a RHS explicit forcing in the next timestep
        self.previous_stress.interpolate(self.prefactor_prestress * self.deviatoric_stress)
        self.displacement.interpolate(self.displacement+self.stokes_vars[0])
