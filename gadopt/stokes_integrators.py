from .momentum_equation import StokesEquations
from .free_surface_equation import FreeSurfaceEquation
from .approximations import BaseApproximation
from .utility import upward_normal, ensure_constant, InteriorBC, DEBUG, INFO, depends_on, log_level
from typing import Optional
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
        "pc_python_type": "firedrake.MassInvPC",
        "Mp_pc_type": "ksp",
        "Mp_ksp_ksp_rtol": 1e-5,
        "Mp_ksp_ksp_type": "cg",
        "Mp_ksp_pc_type": "sor",
    }
}
"""Default solver parameters for iterative solvers"""

direct_stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default solver parameters for direct solvers"""

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
}
"""Default solver parameters for non-linear systems"""


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: Optional[list[int]] = None
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z: Firedrake mixed function space associated with the Stokes system
      closed: Whether to include a constant pressure null space
      rotational: Whether to include all rotational modes
      translations: List of translations to include

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    stokes_funcspace = Z.subfunctions

    if rotational:
        if dim == 2:
            rotV = fd.Function(stokes_funcspace[0]).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(stokes_funcspace[0]).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(stokes_funcspace[0]).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(stokes_funcspace[0]).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(stokes_funcspace[0]).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = stokes_funcspace[0]

    if closed:
        p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = stokes_funcspace[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface nullspace
    null_space += stokes_funcspace[2:]

    return fd.MixedVectorSpaceBasis(Z, null_space)


class StokesSolver:
    """Solves the Stokes system.

    Arguments:
      z: Firedrake function representing the mixed Stokes system
      T: Firedrake function representing the temperature
      approximation: Approximation describing the system of equations
      bcs: Dictionary of identifier-value pairs specifying boundary conditions
      mu: Firedrake function representing dynamic viscosity
      quad_degree: Quadrature degree. Default value is `2p + 1`, where
                     p is the polynomial degree of the trial space
      cartesian: Whether to use Cartesian coordinates
      solver_parameters: Dictionary of solver parameters provided to PETSc
      J: Firedrake function representing the Jacobian of the system
      constant_jacobian: Whether the Jacobian of the system is constant

    """
    name = 'Stokes'

    def __init__(
        self,
        z: fd.Function,
        T: fd.Function,
        approximation: BaseApproximation,
        bcs: Optional[dict[int, dict[str, int | float]]] = None,
        mu: fd.Function | int | float = 1,
        quad_degree: int = 6,
        cartesian: bool = True,
        solver_parameters: Optional[dict[str, str | float]] = None,
        J: Optional[fd.Function] = None,
        constant_jacobian: bool = False,
        iterative_2d: bool = False,
        free_surface_dt: Optional[float] = None,
        free_surface_variable_rho: bool = True,
        free_surface_theta: float = 0.5,
        **kwargs
    ):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.approximation = approximation

        self.mu = ensure_constant(mu)
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)

        self.solver_kwargs = kwargs
        self.stokes_vars = fd.split(self.solution)  # For default stokes this is a tuple of (velocity, pressure)
        self.k = upward_normal(self.Z.mesh(), cartesian)

        # Add velocity, pressure and buoyancy term to the fields dictionary
        self.fields = {
            'velocity': self.stokes_vars[0],
            'pressure': self.stokes_vars[1],
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'source': approximation.buoyancy(self.stokes_vars[1], T) * self.k,
            'rho_continuity': approximation.rho_continuity(),
        }

        # Setup boundary conditions
        self.weak_bcs = {}
        self.strong_bcs = []
        free_surface_dict = {}  # Separate dictionary for copying free surface information
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
                    free_surface_dict[id] = value
                    self.free_surface = True
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

        if self.free_surface:
            if free_surface_dt is None:
                raise TypeError("Please provide a timestep to advance the free surface, currently free_surface_dt=None.")

            self.solution_old = fd.Function(self.solution)
            eta = []
            eta_old = []
            eta_theta = []
            self.free_surface_id_list = []

            c = 0  # Counter for free surfaces (N.b. we already have two equations from StokesEquations)
            for id, value in free_surface_dict.items():
                exterior_density = value.get('exterior_density', 0)

                # Define free surface variables for timestepping
                eta.append(self.stokes_vars[2+c])
                eta_old.append(fd.split(self.solution_old)[2+c])
                eta_theta.append((1-free_surface_theta)*eta_old[c] + free_surface_theta*eta[c])

                # Add free surface equation
                self.equations.append(FreeSurfaceEquation(self.Z.sub(2+c), self.Z.sub(2+c), quad_degree=quad_degree,
                                      free_surface_id=id, free_surface_dt=free_surface_dt, theta=free_surface_theta, k=self.k))
                # Add free surface stress term
                if free_surface_variable_rho:
                    # Use actual density
                    surface_rho = approximation.rho_field(self.stokes_vars[1], T)
                else:
                    # Use reference density (needed for analytical cylindrical cases)
                    surface_rho = approximation.rho
                self.weak_bcs[id] = {'normal_stress': (surface_rho - exterior_density) * approximation.g * eta_theta[c]}

                # Set internal dofs to zero to prevent singular matrix for free surface equation
                self.strong_bcs.append(InteriorBC(self.Z.sub(2+c), 0, id))

                self.free_surface_id_list.append(id)

                c += 1

        # Add terms to Stokes Equations
        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if self.free_surface:
            for i in range(len(eta)):
                # Add free surface time derivative term
                # Multiply by theta to keep the block system symmetric for the implicit coupling case
                # (N.b. we already have two equations from StokesEquations)
                self.F += self.equations[2+i].mass_term(self.test[2+i], free_surface_theta*(eta[i]-eta_old[i])/free_surface_dt)

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

    def setup_solver(self):
        """Sets up the solver."""
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
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()
        if self.free_surface:
            self.solution_old.assign(self.solution)  # Need to update old solution for implicit free surface
        self.solver.solve()
