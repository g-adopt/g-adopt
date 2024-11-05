r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

import abc
from collections import defaultdict
from numbers import Number
from statistics import geometric_mean

from firedrake import *

from .approximations import Approximation
from .equations import Equation
from .free_surface_equation import free_surface_term
from .free_surface_equation import mass_term as mass_term_fs
from .momentum_equation import residual_terms_stokes
from .utility import DEBUG, INFO, InteriorBC, depends_on, log_level, vertical_component

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
    },
}
"""Default iterative solver parameters for solution of stokes system.

We configure the Schur complement approach as described in Section of 4.3 of Davies et
al. (2022), using PETSc's fieldsplit preconditioner type, which provides a class of
preconditioners for mixed problems that allows a user to apply different preconditioners
to different blocks of the system.

The fieldsplit_0 entries configure solver options for the velocity block. The linear
systems associated with this matrix are solved using a combination of the Conjugate
Gradient (cg) method and an algebraic multigrid preconditioner (GAMG).

The fieldsplit_1 entries contain solver options for the Schur complement solve itself.
For preconditioning, we approximate the Schur complement matrix with a mass matrix
scaled by viscosity. Since this preconditioner step involves an iterative solve, the
Krylov method used for the Schur complement needs to be of flexible type, and we use
FGMRES by default.

We note that our default solver parameters can be augmented or adjusted by accessing the
solver_parameter dictionary, for example:

```
   stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
   stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
   stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
   stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
   stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2
```

Note:
  G-ADOPT defaults to iterative solvers in 3-D.
"""

direct_stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
"""Default direct solver parameters for solution of Stokes system.

Configured to use LU factorisation, using the MUMPS library.

Note:
  G-ADOPT defaults to direct solvers in 2-D.
"""

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
}
"""Default solver parameters for non-linear systems.

We use a setup based on Newton's method (newtonls) with a secant line search over the
L2-norm of the function.
"""


class MetaPostInit(abc.ABCMeta):
    """Calls the implemented __post_init__ method after __init__ returns.

    The implemented behaviour allows any subclass __init__ method to first call its
    parent class's __init__ through super(), then execute its own code, and finally call
    __post_init__. The latter call is automatic and does not require any attention from
    the developer or user.
    """

    def __call__(cls, *args, **kwargs):
        class_instance = super().__call__(*args, **kwargs)
        class_instance.__post_init__()

        return class_instance


class MassMomentumBase(abc.ABC, metaclass=MetaPostInit):
    """Solver for a system involving mass and momentum conservation.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space.
      approximation:
        G-ADOPT approximation defining terms in the system of equations.
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value).
      quad_degree:
        Integer denoting the quadrature degree. Default value is `2p + 1`, where p is
        the polynomial degree of the trial space.
      solver_parameters:
        Either a dictionary of PETSc solver parameters or a string specifying a default
        set of parameters defined in G-ADOPT.
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system.
      constant_jacobian:
        A boolean specifying whether the Jacobian of the system is constant.
      nullspace:
        Dictionary of nullspace options, including transpose and near nullspaces.
    """

    name = "MassMomentum"

    def __init__(
        self,
        solution: Function,
        approximation: Approximation,
        /,
        *,
        bcs: dict[int, dict[str, Number]] = {},
        quad_degree: int = 6,
        solver_parameters: dict[str, str | Number] | str | None = None,
        J: Function | None = None,
        constant_jacobian: bool = False,
        nullspace: dict[str, float] = {},
    ) -> None:
        self.solution = solution
        self.approximation = approximation
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.nullspace = nullspace

        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.tests = TestFunctions(self.solution_space)
        self.solution_split = split(solution)

        self.F = 0.0  # Weak form of the system

        # Solver object is set up later to permit editing default solver options.
        self._solver_ready = False

    def __post_init__(self) -> None:
        self.set_boundary_conditions()
        self.set_equation()
        self.set_solver_options()

    def set_boundary_conditions(self) -> None:
        """Sets up boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        normal_stress_fs = None

        for bc_id, bc in self.bcs.items():
            weak_bc = defaultdict(lambda: 0.0)

            for bc_type, value in bc.items():
                if bc_type == "u":
                    self.strong_bcs.append(
                        DirichletBC(self.solution_space.sub(0), value, bc_id)
                    )
                elif bc_type == "ux":
                    self.strong_bcs.append(
                        DirichletBC(self.solution_space.sub(0).sub(0), value, bc_id)
                    )
                elif bc_type == "uy":
                    self.strong_bcs.append(
                        DirichletBC(self.solution_space.sub(0).sub(1), value, bc_id)
                    )
                elif bc_type == "uz":
                    self.strong_bcs.append(
                        DirichletBC(self.solution_space.sub(0).sub(2), value, bc_id)
                    )
                elif bc_type == "free_surface":
                    normal_stress_fs = self.set_free_surface_boundary(value, bc_id)
                else:
                    weak_bc[bc_type] = value

            if normal_stress_fs is not None:
                weak_bc["normal_stress"] += normal_stress_fs
                normal_stress_fs = None

            self.weak_bcs[bc_id] = weak_bc

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> ufl.algebra.Product | ufl.algebra.Sum:
        """Sets the given boundary as a free surface."""
        pass

    @abc.abstractmethod
    def set_equation(self):
        """Sets up the term contributions from each equation."""
        raise NotImplementedError

    def set_solver_options(self) -> None:
        """Sets PETSc solver parameters."""
        # Application context for the inverse mass matrix preconditioner
        self.appctx = {"mu": self.approximation.mu / self.approximation.rho}

        if isinstance(solver_preset := self.solver_parameters, dict):
            return

        if not depends_on(self.approximation.mu, self.solution):
            self.solver_parameters = {"snes_type": "ksponly"}
        else:
            self.solver_parameters = newton_stokes_solver_parameters.copy()

        if INFO >= log_level:
            self.solver_parameters["snes_monitor"] = None

        if solver_preset is not None:
            match solver_preset:
                case "direct":
                    self.solver_parameters.update(direct_stokes_solver_parameters)
                case "iterative":
                    self.solver_parameters.update(iterative_stokes_solver_parameters)
                case _:
                    raise ValueError(f"Solver type '{solver_preset}' not implemented.")
        elif self.mesh.topological_dimension() == 2 and self.mesh.cartesian:
            self.solver_parameters.update(direct_stokes_solver_parameters)
        else:
            self.solver_parameters.update(iterative_stokes_solver_parameters)

        # extra options for iterative solvers
        if self.solver_parameters.get("pc_type") == "fieldsplit":
            if DEBUG >= log_level:
                self.solver_parameters["fieldsplit_0"]["ksp_converged_reason"] = None
                self.solver_parameters["fieldsplit_1"]["ksp_monitor"] = None
            elif INFO >= log_level:
                self.solver_parameters["fieldsplit_1"]["ksp_converged_reason"] = None

    def setup_solver(self) -> None:
        """Sets up the solver."""
        if self.constant_jacobian:
            z_tri = TrialFunction(self.solution_space)
            F_stokes_lin = replace(self.F, {self.solution: z_tri})
            a, L = lhs(F_stokes_lin), rhs(F_stokes_lin)
            self.problem = LinearVariationalProblem(
                a, L, self.solution, bcs=self.strong_bcs, constant_jacobian=True
            )
            self.solver = LinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                **self.nullspace,
                appctx=self.appctx,
            )
        else:
            self.problem = NonlinearVariationalProblem(
                self.F, self.solution, bcs=self.strong_bcs, J=self.J
            )
            self.solver = NonlinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                **self.nullspace,
                appctx=self.appctx,
            )
        self._solver_ready = True

    def solver_callback(self) -> None:
        """Optional instructions to execute right after a solve."""
        pass

    def solve(self) -> None:
        """Solves the system."""
        if not self._solver_ready:
            self.setup_solver()

        self.solver.solve()

        self.solver_callback()


class StokesSolver(MassMomentumBase):
    """Solver for the Stokes system.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space.
      approximation:
        G-ADOPT approximation defining terms in the system of equations.
      T:
        Firedrake function representing the temperature field.
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value).
      quad_degree:
        Integer denoting the quadrature degree. Default value is `2p + 1`, where p is
        the polynomial degree of the trial space.
      solver_parameters:
        Either a dictionary of PETSc solver parameters or a string specifying a default
        set of parameters defined in G-ADOPT.
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system.
      constant_jacobian:
        A boolean specifying whether the Jacobian of the system is constant.
      nullspace:
        Dictionary of nullspace options, including transpose and near nullspaces.
      timestep_fs:
        Float representing the timestep used in the free surface equation.
      theta_fs:
        Float quantifying the implicit contribution in the free-surface timestepping.
        theta_fs = 0: Forward Euler
        theta_fs = 0.5: Crank-Nicolson (default)
        theta_fs = 1: Backward Euler
    """

    name = "Stokes"

    def __init__(
        self,
        solution: Function,
        approximation: Approximation,
        T: Function = Constant(0.0),
        /,
        *,
        timestep_fs: float | None = None,
        theta_fs: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, **kwargs)

        self.T = T
        self.timestep_fs = timestep_fs
        self.theta_fs = theta_fs

        self.eta_old = [None] * (len(solution.subfunctions) - 2)
        self.buoyancy_fs = self.eta_old.copy()
        self.free_surface_map = {}

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> ufl.algebra.Product | ufl.algebra.Sum:
        """Sets the given boundary as a free surface.

        This method calculates the normal stress at the free surface boundary and sets
        a zero-interior strong condition away from that boundary. It also populates the
        `free_surface_map` dictionary used to calculate the free-surface contribution to
        the momentum weak form.

        Args:
          params_fs:
            A dictionary holding information about the free surface boundary.
            Valid dictionary keys:
              eta_index:
                Index of eta function in mixed space (starts from 0)
              rho_ext:
                Exterior density along the free surface
              Ra_fs:
                Free-surface Rayleigh number (buoyancy along the free surface)
              include_buoyancy_effects:
                Whether the interior density should vary according to buoyancy effects
          bc_id:
            An integer representing the index of the mesh boundary.

        Returns:
          A UFL expression for the normal stress at the free surface boundary.
        """
        eta_ind = params_fs["eta_index"]
        self.free_surface_map[bc_id] = eta_ind

        p = self.solution_split[1]
        eta = self.solution_split[2 + eta_ind]
        func_space_eta = self.solution_space[2 + eta_ind]
        self.eta_old[eta_ind] = Function(func_space_eta).interpolate(eta)

        # Set internal degrees of freedom to zero to prevent singular matrix
        self.strong_bcs.append(InteriorBC(func_space_eta, 0, bc_id))

        self.buoyancy_fs[eta_ind] = self.approximation.buoyancy(
            p=p, T=self.T, params_fs=params_fs
        )
        eta_theta = self.theta_fs * eta + (1 - self.theta_fs) * self.eta_old[eta_ind]

        normal_stress = self.buoyancy_fs[eta_ind] * eta_theta

        return normal_stress

    def set_equation(self) -> None:
        """Sets up UFL forms for the Stokes equations residual."""
        eqs_attrs = [
            {"p": self.solution_split[1], "T": self.T},
            {"u": self.solution_split[0]},
        ]
        for i, (terms_eq, eq_attrs) in enumerate(zip(residual_terms_stokes, eqs_attrs)):
            eq = Equation(
                self.tests[i],
                self.solution_space[i],
                terms_eq,
                eq_attrs=eq_attrs,
                approximation=self.approximation,
                bcs=self.weak_bcs,
                quad_degree=self.quad_degree,
            )
            self.F -= eq.residual(self.solution_split[i])

        for bc_id, eta_ind in self.free_surface_map.items():
            eta = self.solution_split[2 + eta_ind]
            func_space_eta = self.solution_space[2 + eta_ind]

            # To ensure the top right and bottom left corners of the block matrix remain
            # symmetric we need to multiply the free surface equation (kinematic bc)
            # with `-theta * delta_rho * g`. This is similar to rescaling eta ->
            # eta_tilde in Kramer et al. (2012); see block matrix shown in Equation 23.
            buoyancy_scale = -self.theta_fs * self.buoyancy_fs[eta_ind]

            eq_attrs = {
                "boundary_id": bc_id,
                "buoyancy_scale": buoyancy_scale,
                "u": self.solution_split[0],
            }

            eq_fs = Equation(
                self.tests[2 + eta_ind],
                func_space_eta,
                free_surface_term,
                mass_term=mass_term_fs,
                eq_attrs=eq_attrs,
                quad_degree=self.quad_degree,
            )
            self.F += eq_fs.mass((eta - self.eta_old[eta_ind]) / self.timestep_fs)
            self.F -= eq_fs.residual(eta)

            self.ds_fs = eq_fs.ds

    def set_solver_options(self) -> None:
        super().set_solver_options()

        if (
            self.free_surface_map
            and self.solver_parameters.get("pc_type") == "fieldsplit"
        ):
            # Gather pressure and free surface fields for Schur complement solve
            fields = ",".join(map(str, range(1, len(self.solution_split))))
            self.solver_parameters.update(
                {"pc_fieldsplit_0_fields": "0", "pc_fieldsplit_1_fields": fields}
            )
            # Update mass inverse preconditioner
            self.solver_parameters["fieldsplit_1"].update(
                {"pc_python_type": "gadopt.FreeSurfaceMassInvPC"}
            )

            # Update application context
            self.appctx["free_surface"] = self.free_surface_map
            self.appctx["ds"] = self.ds_fs

    def solver_callback(self) -> None:
        # The old free surface height must be updated for an implicit free surface.
        for eta, eta_old in zip(self.solution_split[2:], self.eta_old):
            eta_old.interpolate(eta)


class ViscoelasticSolver(MassMomentumBase):
    """Solves the Stokes system assuming a Maxwell viscoelastic rheology.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space.
      approximation:
        G-ADOPT approximation defining terms in the system of equations.
      displ:
        Firedrake function representing the total displacement.
      tau_old:
        Firedrake function representing the deviatoric stress at the previous time step.
      dt:
        Float representing the simulation's timestep.
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value).
      quad_degree:
        Integer denoting the quadrature degree. Default value is `2p + 1`, where p is
        the polynomial degree of the trial space.
      solver_parameters:
        Either a dictionary of PETSc solver parameters or a string specifying a default
        set of parameters defined in G-ADOPT.
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system.
      constant_jacobian:
        A boolean specifying whether the Jacobian of the system is constant.
      nullspace:
        Dictionary of nullspace options, including transpose and near nullspaces.
    """

    name = "Viscoelastic"

    def __init__(
        self,
        solution: Function,
        displ: Function,
        tau_old: Function,
        approximation: Approximation,
        dt: float | Constant | Function,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, **kwargs)

        self.displ = displ
        self.tau_old = tau_old

        # Characteristic relaxation time
        self.maxwell_time = approximation.mu / approximation.G
        # Effective viscosity
        approximation.mu = approximation.mu / (self.maxwell_time + dt / 2)
        # Scaling factor for the previous stress
        self.stress_scale = (self.maxwell_time - dt / 2) / (self.maxwell_time + dt / 2)

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> ufl.algebra.Product | ufl.algebra.Sum:
        """Sets the given boundary as a free surface.

        Args:
          params_fs:
            A dictionary holding information about the free surface boundary.
            Valid dictionary keys:
              rho_ext:
                Exterior density along the free surface
              Ra_fs:
                Free-surface Rayleigh number (buoyancy along the free surface)
              include_buoyancy_effects:
                Whether the interior density should vary according to buoyancy effects
          bc_id:
            An integer representing the index of the mesh boundary.

        Returns:
          A UFL expression for the normal stress at the free surface boundary.
        """
        # First, make the displacement term implicit by incorporating the unknown
        # `incremental displacement' (u) that we are solving for. Then, calculate the
        # free surface stress term. This is also referred to as the Hydrostatic
        # Prestress advection term in the GIA literature.
        u, p = self.solution_split[:2]
        buoyancy_fs = self.approximation.buoyancy(
            p=p, displ=self.displ, params_fs=params_fs
        )
        normal_stress = buoyancy_fs * vertical_component(u + self.displ)

        return normal_stress

    def set_equation(self) -> None:
        """Sets up UFL forms for the viscoelastic Stokes equations residual."""
        eqs_attrs = [
            {
                "p": self.solution_split[1],
                "displ": self.displ,
                "stress_old": self.tau_old,
            },
            {"u": self.solution_split[0]},
        ]

        # Rescale equations using the Maxwell time to make sure that the solver
        # converges with strong boundary conditions in parallel.
        func = Function(FunctionSpace(self.mesh, "CG", 2))
        rescale_factor = geometric_mean(func.interpolate(self.maxwell_time).dat.data)

        for i, (terms_eq, eq_attrs) in enumerate(zip(residual_terms_stokes, eqs_attrs)):
            eq = Equation(
                self.tests[i],
                self.solution_space[i],
                terms_eq,
                eq_attrs=eq_attrs,
                approximation=self.approximation,
                bcs=self.weak_bcs,
                quad_degree=self.quad_degree,
                rescale_factor=rescale_factor,
            )
            self.F -= eq.residual(self.solution_split[i])

    def solver_callback(self) -> None:
        u = self.solution_split[0]
        self.displ.interpolate(self.displ + u)
        # History stress term (explicit RHS forcing)
        self.tau_old.interpolate(
            self.stress_scale * self.approximation.stress(u, self.tau_old)
        )


def create_stokes_nullspace(
    Z: functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: list[int] | None = None,
    approximation: Approximation | None = None,
    top_subdomain_id: str | int | None = None,
) -> nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Args:
      Z:
        Firedrake mixed function space associated with the Stokes system
      closed:
        Whether to include a constant pressure null space
      rotational:
        Whether to include all rotational modes
      translations:
        List of translations to include
      approximation:
        Approximation (ALA) for calculating (non-constant) right nullspace
      top_subdomain_id:
        Boundary id of top surface; required when providing approximation

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    # approximation and top_subdomain_id are both needed when calculating right
    # nullspace for ala
    if (approximation is None) != (top_subdomain_id is None):
        raise ValueError(
            "Both approximation and top_subdomain_id must be provided, or both "
            "must be None."
        )

    X = SpatialCoordinate(Z.mesh())
    dim = len(X)

    if rotational:
        if dim == 2:
            rotV = Function(Z[0]).interpolate(as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = Function(Z[0]).interpolate(as_vector((0, -X[2], X[1])))
            y_rotV = Function(Z[0]).interpolate(as_vector((X[2], 0, -X[0])))
            z_rotV = Function(Z[0]).interpolate(as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(Function(Z[0]).interpolate(as_vector(vec)))

    if basis:
        V_nullspace = VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = Z[0]

    if closed:
        if approximation and approximation.preset == "ALA":
            p = ala_right_nullspace(Z[1], approximation, top_subdomain_id)
            p_nullspace = VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = Z[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface nullspace
    null_space += Z[2:]

    return MixedVectorSpaceBasis(Z, null_space)


def ala_right_nullspace(
    W: functionspaceimpl.WithGeometry,
    approximation: Approximation,
    top_subdomain_id: str | int,
) -> Function:
    r"""Compute pressure nullspace for Anelastic Liquid Approximation.

    Args:
      W: pressure function space
      approximation: AnelasticLiquidApproximation with equation parameters
      top_subdomain_id: boundary id of top surface

    Returns:
      pressure nullspace solution

    To obtain the pressure nullspace solution for the Stokes equation in the Anelastic
    Liquid Approximation (which includes a pressure-dependent buoyancy term), we try to
    solve the equation:

    $$
      -nabla p + g "Di" rho chi c_p/(c_v gamma) hatk p = 0
    $$

    Taking the divergence:

    $$
      -nabla * nabla p + nabla * (g "Di" rho chi c_p/(c_v gamma) hatk p) = 0,
    $$

    then testing it with q:

    $$
        int_Omega -q nabla * nabla p dx + int_Omega q nabla * (g "Di" rho chi c_p/(c_v gamma) hatk p) dx = 0
    $$

    followed by integration by parts:

    $$
        int_Gamma -bb n * q nabla p ds + int_Omega nabla q cdot nabla p dx +
        int_Gamma bb n * hatk q g "Di" rho chi c_p/(c_v gamma) p dx -
        int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0
    $$

    This elliptic equation can be solved with natural boundary conditions by imposing
    our original equation above, which eliminates all boundary terms:

    $$
      int_Omega nabla q * nabla p dx - int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0.
    $$

    However, if we do so on all boundaries we end up with a system that has the same
    nullspace as the one we are after (note that we ended up merely testing the original
    equation with $nabla q$). Instead, we use the fact that the gradient of the null
    mode is always vertical, and thus the null mode is constant at any horizontal level
    (geoid), such as the top surface. Choosing any nonzero constant for this surface
    fixes the arbitrary scalar multiplier of the null mode. We choose the value of one
    and apply it as a Dirichlet boundary condition.

    Note that this procedure does not necessarily compute the exact nullspace of the
    *discretised* Stokes system. In particular, since not every test function $v$ in
    $V$, the velocity test space, can be written as $v=nabla q$ with $q in W$, the
    pressure test space, the two terms do not necessarily exactly cancel when tested
    with $v$ instead of $nabla q$ as in our final equation. However, in practice, the
    discrete error appears to be small enough, and providing this nullspace gives an
    improved convergence of the iterative Stokes solver.
    """
    W = FunctionSpace(mesh=W.mesh(), family=W.ufl_element())
    q = TestFunction(W)
    p = Function(W, name="Pressure nullspace")

    F = inner(grad(q), grad(p)) * dx
    F += vertical_component(grad(q) * approximation.buoyancy(p=p)) * dx

    # Fix the solution at the top boundary
    solve(F == 0, p, bcs=DirichletBC(W, 1.0, top_subdomain_id))

    return p
