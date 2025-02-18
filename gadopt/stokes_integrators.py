r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

import abc
from numbers import Number
from typing import Optional
from typing import Any

import firedrake as fd

from .approximations import BaseApproximation, AnelasticLiquidApproximation
from .equations import Equation
from .free_surface_equation import free_surface_term
from .free_surface_equation import mass_term as mass_term_fs
from .momentum_equation import residual_terms_stokes, residual_terms_compressible_viscoelastic
from .scalar_equation import mass_term, residual_terms_internal_variable
from .utility import DEBUG, INFO, InteriorBC, depends_on, log_level, upward_normal

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
"""Default iterative solver parameters for solution of stokes system.

We configure the Schur complement approach as described in Section of
4.3 of Davies et al. (2022), using PETSc's fieldsplit preconditioner
type, which provides a class of preconditioners for mixed problems
that allows a user to apply different preconditioners to different
blocks of the system.

The fieldsplit_0 entries configure solver options for the velocity
block. The linear systems associated with this matrix are solved using
a combination of the Conjugate Gradient (cg) method and an algebraic
multigrid preconditioner (GAMG).

The fieldsplit_1 entries contain solver options for the Schur
complement solve itself. For preconditioning, we approximate the Schur
complement matrix with a mass matrix scaled by viscosity, with the
viscosity provided through the optional mu keyword argument to Stokes
solver. Since this preconditioner step involves an iterative solve,
the Krylov method used for the Schur complement needs to be of
flexible type, and we use FGMRES by default.

We note that our default solver parameters can be augmented or
adjusted by accessing the solver_parameter dictionary, for example:

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

We use a setup based on Newton's method (newtonls) with a secant line
search over the L2-norm of the function.
"""


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: Optional[list[int]] = None,
    ala_approximation: Optional[AnelasticLiquidApproximation] = None,
    top_subdomain_id: Optional[str | int] = None,
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z: Firedrake mixed function space associated with the Stokes system
      closed: Whether to include a constant pressure null space
      rotational: Whether to include all rotational modes
      translations: List of translations to include
      ala_approximation: AnelasticLiquidApproximation for calculating (non-constant) right nullspace
      top_subdomain_id: Boundary id of top surface. Required when providing
                        ala_approximation.

    Returns:
      A Firedrake mixed vector space basis incorporating the null space components

    """
    # ala_approximation and top_subdomain_id are both needed when calculating right nullspace for ala
    if (ala_approximation is None) != (top_subdomain_id is None):
        raise ValueError("Both ala_approximation and top_subdomain_id must be provided, or both must be None.")

    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    stokes_subspaces = Z.subfunctions

    if rotational:
        if dim == 2:
            rotV = fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = stokes_subspaces[0]

    if closed:
        if ala_approximation:
            p = ala_right_nullspace(W=stokes_subspaces[1], approximation=ala_approximation, top_subdomain_id=top_subdomain_id)
            p_nullspace = fd.VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = stokes_subspaces[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface nullspace
    null_space += stokes_subspaces[2:]

    return fd.MixedVectorSpaceBasis(Z, null_space)


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

    ### Valid keys for boundary conditions
    |   Condition   |  Type  |                 Description                  |
    | :------------ | :----- | :------------------------------------------: |
    | u             | Strong | Solution                                     |
    | ux            | Strong | Solution along the first Cartesian axis      |
    | uy            | Strong | Solution along the second Cartesian axis     |
    | uz            | Strong | Solution along the third Cartesian axis      |
    | un            | Weak   | Solution along the normal to the boundary    |
    | stress        | Weak   | Traction across the boundary                 |
    | normal_stress | Weak   | Stress component normal to the boundary      |
    | free_surface  | Weak   | Free-surface characteristics of the boundary |

    ### Valid keys describing the free surface boundary:
    |     Argument     | Required |                  Description                   |
    | :--------------- | :------: | :--------------------------------------------: |
    | eta_index        | True     | Function index in mixed space                  |
    | rho_ext          | False    | Exterior density along the free surface        |
    | Ra_fs            | False    | Rayleigh number at the free surface            |
    | include_buoyancy | False    | Whether the interior density includes buoyancy |

    ### Classic theta values for coupled implicit time integration
    | Theta |        Scheme         |
    | :---- | :-------------------: |
    | 0.5   | Crank-Nicolson method |
    | 1.0   | Backward Euler method |

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      coupled_dt:
        Float quantifying the time step used in a coupled time integration
      theta:
        Float quantifying the implicit contribution in a coupled time integration
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      quad_degree:
        Integer denoting the quadrature degree
      solver_parameters:
        Dictionary of PETSc solver options or string matching a default set thereof
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system
      constant_jacobian:
        Boolean specifying whether the Jacobian of the system is constant
    """

    name = "MassMomentum"

    def __init__(
        self,
        solution: fd.Function,
        approximation: BaseApproximation,
        /,
        *,
        coupled_dt: float | None = None,
        theta: float = 0.5,
        bcs: dict[int, dict[str, Any]] = {},
        quad_degree: int = 6,
        solver_parameters: dict[str, str | Number] | str | None = None,
        J: fd.Function | None = None,
        constant_jacobian: bool = False,
        **kwargs,
    ) -> None:
        self.solution = solution
        self.approximation = approximation
        self.coupled_dt = coupled_dt
        self.theta = theta
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.solver_kwargs = kwargs

        self.solution_old = solution.copy(deepcopy=True)
        self.solution_split = fd.split(solution)
        self.solution_old_split = fd.split(self.solution_old)
        self.solution_theta_split = [
            theta * sol + (1 - theta) * sol_old
            for sol, sol_old in zip(self.solution_split, self.solution_old_split)
        ]

        self.Z = solution.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.k = upward_normal(self.mesh)

        self.equations = []
        self.F = 0.0  # Weak form of the system

        # Solver object is set up later to permit editing default solver options.
        self._solver_setup = False

    def __post_init__(self) -> None:
        """Executes selected methods after the class is instantiated."""
        self.set_boundary_conditions()
        self.set_equations()
        if self.free_surface_dict:
            self.setup_free_surface()
        self.set_form()
        self.set_solver_options()

    def set_boundary_conditions(self) -> None:
        """Sets strong and weak boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}
        self.free_surface_dict = {}

        for id, bc in self.bcs.items():
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
                    # N.b. stokes_integrators assumes that the order of the bcs matches the order of the
                    # free surfaces defined in the mixed space. This is not ideal - python dictionaries
                    # are ordered by insertion only since recently (since 3.7) - so relying on their order
                    # is fraught and not considered pythonic. At the moment let's consider having more
                    # than one free surface a bit of a niche case for now, and leave it as is...

                    # Copy free surface information to a new dictionary
                    self.free_surface_dict[id] = value
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[id] = weak_bc

    @abc.abstractmethod
    def set_equations(self):
        """Sets Equation instances for each equation in the system.

        Equations must be ordered like solutions in the mixed space.
        """

    def set_form(self) -> None:
        """Sets the weak form including linear and bilinear terms."""
        for equation, solution, solution_old in zip(
            self.equations, self.solution_split, self.solution_old_split
        ):
            if equation.mass_term:
                self.F += equation.mass((solution - solution_old) / self.coupled_dt)
            self.F -= equation.residual(solution)

    def set_solver_options(self) -> None:
        """Sets PETSc solver parameters."""
        # Application context for the inverse mass matrix preconditioner
        self.appctx = {"mu": self.approximation.mu / self.approximation.rho_continuity()}

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
                    raise ValueError("Solver type must be 'direct' or 'iterative'.")
        elif self.mesh.topological_dimension() == 2 and self.mesh.cartesian:
            self.solver_parameters.update(direct_stokes_solver_parameters)
        else:
            self.solver_parameters.update(iterative_stokes_solver_parameters)

        # Extra options for iterative solvers
        if self.solver_parameters.get("pc_type") == "fieldsplit":
            if DEBUG >= log_level:
                self.solver_parameters["fieldsplit_0"]["ksp_converged_reason"] = None
                self.solver_parameters["fieldsplit_1"]["ksp_monitor"] = None
            elif INFO >= log_level:
                self.solver_parameters["fieldsplit_1"]["ksp_converged_reason"] = None

    def setup_solver(self) -> None:
        """Sets up the solver."""
        if self.constant_jacobian:
            trial = fd.TrialFunction(self.Z)
            F = fd.replace(self.F, {self.solution: trial})
            a, L = fd.lhs(F), fd.rhs(F)

            self.problem = fd.LinearVariationalProblem(
                a, L, self.solution, bcs=self.strong_bcs, constant_jacobian=True
            )
            self.solver = fd.LinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                appctx=self.appctx,
                **self.solver_kwargs,
            )
        else:
            self.problem = fd.NonlinearVariationalProblem(
                self.F, self.solution, bcs=self.strong_bcs, J=self.J
            )
            self.solver = fd.NonlinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                appctx=self.appctx,
                **self.solver_kwargs,
            )

        self._solver_setup = True

    def solver_callback(self) -> None:
        """Instructions to execute right after a solve."""
        self.solution_old.assign(self.solution)

    def solve(self):
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()

        self.solver.solve()
        self.solver_callback()


class StokesSolver(MassMomentumBase):
    """Solver for the Stokes system.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space
      T:
        Firedrake function representing the temperature field
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      coupled_fields:
        Dictionary in a coupled time integration
      coupled_dt:
        Float quantifying the time step used in a coupled time integration
      theta:
        Float quantifying the implicit contribution in a coupled time integration
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      quad_degree:
        Integer denoting the quadrature degree
      solver_parameters:
        Dictionary of PETSc solver options or string matching a default set thereof
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system
      constant_jacobian:
        Boolean specifying whether the Jacobian of the system is constant
      nullspace:
        Dictionary of nullspace options, including transpose and near nullspaces
    """

    name = "Stokes"

    def __init__(
        self,
        solution: fd.Function,
        T: fd.Function | float,
        approximation: BaseApproximation,
        /,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, **kwargs)

        self.u, self.p = self.solution_split[:2]
        self.T = T

        self.free_surface_map = {}
        self.buoyancy_fs = [None] * len(self.solution_split)

    def set_equations(self) -> None:
        stress = self.approximation.stress(self.u)
        source = self.approximation.buoyancy(self.p, self.T) * self.k
        rho_mass = self.approximation.rho_continuity()
        eqs_attrs = [{"p": self.p, "stress": stress, "source": source}, {"u": self.u, "rho_mass": rho_mass}]

        for i in range(len(residual_terms_stokes)):
            self.equations.append(
                Equation(
                    self.test[i],
                    self.Z[i],
                    residual_terms_stokes[i],
                    eq_attrs=eqs_attrs[i],
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                )
            )

    def setup_free_surface(self):
        if self.coupled_dt is None:
            raise TypeError(
                "Please provide a timestep to advance the free surface, currently free_surface_dt=None."
            )

        self.free_surface_id_list = []

        c = 0  # Counter for free surfaces (N.b. we already have two equations from StokesEquations)
        for free_surface_id, free_surface_params in self.free_surface_dict.items():
            # Normal stress #
            # Depending on variable_free_surface_density flag provided to approximation the
            # interior density below the free surface is either set to a constant density or
            # varies spatially according to the buoyancy field
            # N.b. constant reference density is needed for analytical cylindrical cases
            # Prefactor #
            # To ensure the top right and bottom left corners of the block matrix remains symmetric we need to
            # multiply the free surface equation (kinematic bc) with -theta * delta_rho * g. This is similar
            # to rescaling eta -> eta_tilde in Kramer et al. 2012 (e.g. see block matrix shown in Eq 23)
            # N.b. in the case where the density contrast across the free surface is spatially variant due to
            # interior buoyancy changes then the matrix will not be exactly symmetric.
            normal_stress, prefactor = self.approximation.free_surface_terms(
                self.p, self.T, self.solution_theta_split[2+c], self.theta, **free_surface_params
            )

            # Add free surface stress term
            if 'normal_stress' in self.weak_bcs[free_surface_id]:
                # Usually there will be also an ice/water loadi acting as a normal stress in the GIA problem
                existing_value = self.weak_bcs[free_surface_id]['normal_stress']
                self.weak_bcs[free_surface_id]['normal_stress'] = existing_value + normal_stress
            else:
                self.weak_bcs[free_surface_id] = {'normal_stress': normal_stress}

            eq_attrs = {
                "boundary_id": free_surface_id,
                "buoyancy_scale": prefactor,
                "u": self.u,
            }

            # Add the free surface equation
            self.equations.append(
                Equation(
                    self.test[2 + c],
                    self.Z[2 + c],
                    free_surface_term,
                    mass_term=mass_term_fs,
                    eq_attrs=eq_attrs,
                    quad_degree=self.quad_degree,
                )
            )

            # Set internal dofs to zero to prevent singular matrix for free surface equation
            self.strong_bcs.append(
                InteriorBC(self.Z.sub(2 + c), 0, free_surface_id)
            )

            self.free_surface_id_list.append(free_surface_id)

            c += 1

    def set_solver_options(self) -> None:
        super().set_solver_options()

        if (
            self.free_surface_dict
            and self.solver_parameters.get("pc_type") == "fieldsplit"
        ):
            # Update application context
            self.appctx["free_surface_id_list"] = self.free_surface_id_list
            self.appctx["ds"] = self.equations[2].ds

            # Gather pressure and free surface fields for Schur complement solve
            fields_ind = ",".join(map(str, range(1, len(self.solution_split))))
            self.solver_parameters.update(
                {"pc_fieldsplit_0_fields": "0", "pc_fieldsplit_1_fields": fields_ind}
            )
            # Update mass inverse preconditioner
            self.solver_parameters["fieldsplit_1"].update(
                {"pc_python_type": "gadopt.FreeSurfaceMassInvPC"}
            )


def ala_right_nullspace(
        W: fd.functionspaceimpl.WithGeometry,
        approximation: AnelasticLiquidApproximation,
        top_subdomain_id: str | int):
    r"""Compute pressure nullspace for Anelastic Liquid Approximation.

        Arguments:
          W: pressure function space
          approximation: AnelasticLiquidApproximation with equation parameters
          top_subdomain_id: boundary id of top surface

        Returns:
          pressure nullspace solution

        To obtain the pressure nullspace solution for the Stokes equation in Anelastic Liquid Approximation,
        which includes a pressure-dependent buoyancy term, we try to solve the equation:

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

        This elliptic equation can be solved with natural boundary conditions by imposing our original equation above, which eliminates
        all boundary terms:

        $$
          int_Omega nabla q * nabla p dx - int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0.
        $$

        However, if we do so on all boundaries we end up with a system that has the same nullspace, as the one we are after (note that
        we ended up merely testing the original equation with $nabla q$). Instead we use the fact that the gradient of the null mode
        is always vertical, and thus the null mode is constant at any horizontal level (geoid), specifically the top surface. Choosing
        any nonzero constant for this surface fixes the arbitrary scalar multiplier of the null mode. We choose the value of one
        and apply it as a Dirichlet boundary condition.

        Note that this procedure does not necessarily compute the exact nullspace of the *discretised* Stokes system. In particular,
        since not every test function $v in V$, the velocity test space, can be written as $v=nabla q$ with $q in W$, the
        pressure test space, the two terms do not necessarily exactly cancel when tested with $v$ instead of $nabla q$ as in our
        final equation. However, in practice the discrete error appears to be small enough, and providing this nullspace gives
        an improved convergence of the iterative Stokes solver.
    """
    W = fd.FunctionSpace(mesh=W.mesh(), family=W.ufl_element())
    q = fd.TestFunction(W)
    p = fd.Function(W, name="pressure_nullspace")

    # Fix the solution at the top boundary
    bc = fd.DirichletBC(W, 1., top_subdomain_id)

    F = fd.inner(fd.grad(q), fd.grad(p)) * fd.dx

    k = upward_normal(W.mesh())

    F += - fd.inner(fd.grad(q), k * approximation.dbuoyancydp(p, fd.Constant(1.0)) * p) * fd.dx

    fd.solve(F == 0, p, bcs=bc)
    return p


class ViscoelasticStokesSolver(MassMomentumBase):
    """Solves the Stokes system assuming a Maxwell viscoelastic rheology.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space.
      stress_old:
        Firedrake function representing the deviatoric stress at the previous time step.
      displacement:
        Firedrake function representing the total displacement.
      approximation:
        G-ADOPT approximation defining terms in the system of equations.
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
        solution: fd.Function,
        stress_old: fd.Function,
        displacement: fd.Function,
        approximation: BaseApproximation,
        dt: float | fd.Constant | fd.Function,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, coupled_dt=dt, **kwargs)

        self.u, self.p = self.solution_split[:2]
        self.stress_old = stress_old
        self.displacement = displacement

        # Effective viscosity
        self.approximation.mu = approximation.effective_viscosity(self.coupled_dt)

    def setup_free_surface(self):
        # Overload method
        for free_surface_id, free_surface_params in self.free_surface_dict.items():
            # First, make the displacement term implicit by incorporating
            # the unknown `incremental displacement' (u) that
            # we are solving for
            implicit_displacement = self.u + self.displacement
            implicit_displacement_up = fd.dot(implicit_displacement, self.k)
            # Add free surface stress term. This is also referred to as the Hydrostatic Prestress advection term in the GIA literature.
            normal_stress, _ = self.approximation.free_surface_terms(
                implicit_displacement_up, **free_surface_params
            )
            if 'normal_stress' in self.weak_bcs[free_surface_id]:
                # Usually there will be also an ice/water loadi acting as a normal stress in the GIA problem
                existing_value = self.weak_bcs[free_surface_id]['normal_stress']
                self.weak_bcs[free_surface_id]['normal_stress'] = existing_value + normal_stress
            else:
                self.weak_bcs[free_surface_id] = {'normal_stress': normal_stress}

    def set_equations(self) -> None:
        """Sets up UFL forms for the viscoelastic Stokes equations residual."""
        stress = self.approximation.stress(self.u, self.stress_old, self.coupled_dt)
        source = self.approximation.buoyancy(self.displacement) * self.k
        rho_mass = self.approximation.rho_continuity()
        eqs_attrs = [
            {"p": self.p, "stress": stress, "source": source},
            {"u": self.u, "rho_mass": rho_mass}
        ]

        scale_mu = fd.Constant(1 / 1e10)  # this is a scaling factor roughly size of mantle maxwell time to make sure that solve converges with strong bcs in parallel...
        for i in range(len(residual_terms_stokes)):
            self.equations.append(
                Equation(
                    self.test[i],
                    self.Z[i],
                    residual_terms_stokes[i],
                    eq_attrs=eqs_attrs[i],
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                    rescale_factor=scale_mu,
                )
            )

    def solver_callback(self) -> None:
        # Update history stress term for using as a RHS explicit forcing in the next timestep
        # Interpolating with adjoint seems to need subfunction...
        # otherwise 'map toset must be same as Dataset' error
        u_sub = self.solution.subfunctions[0]
        self.stress_old.interpolate(
            self.approximation.prefactor_prestress(self.coupled_dt) * self.approximation.stress(u_sub, self.stress_old, self.coupled_dt)
        )
        self.displacement.interpolate(self.displacement+u_sub)


class InternalVariableSolver(MassMomentumBase):
    """ Solver for internal variable viscoelastic formulation.

    Args:
      solution:
        Firedrake function representing the field over the mixed dipslacement, internal variable space
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      coupled_dt:
        Float quantifying the time step used in a coupled time integration
      theta:
        Float quantifying the implicit contribution in a coupled time integration
      bcs:
        Dictionary specifying boundary conditions (identifier, type, and value)
      quad_degree:
        Integer denoting the quadrature degree
      solver_parameters:
        Dictionary of PETSc solver options or string matching a default set thereof
      J:
        Firedrake function representing the Jacobian of the mixed Stokes system
      constant_jacobian:
        Boolean specifying whether the Jacobian of the system is constant
    """

    name = "InternalVariable"

    def __init__(
        self,
        solution: fd.Function,
        approximation: BaseApproximation,
        coupled_dt,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, coupled_dt=coupled_dt, **kwargs)

        self.u, *self.m = self.solution_split

        # Effective viscosity THIS IS A HACK. need to update SIPG terms for compressibility?
        self.approximation.mu = approximation.effective_viscosity(self.coupled_dt)

    def set_equations(self) -> None:
        stress = self.approximatioin.stress(self.u, self.m)
        source = self.approximation.buoyancy(self.u) * self.k
        strain = self.approximation.deviatoric_strain(self.u)
        maxwell_time = self.approximation.maxwell_time

        residual_terms = [
            residual_terms_compressible_viscoelastic,
            residual_terms_internal_variable,
        ]
        eqs_attrs = [
            {"stress": stress, "source": source},
            {"source": strain / maxwell_time, "sink_coeff": 1 / maxwell_time},
        ]
        mass_terms = [None, mass_term]
        scaling_factors = [1, -self.theta]

        for i in range(len(self.tests)):
            self.equations.append(
                Equation(
                    self.tests[i],
                    self.solution_space[i],
                    residual_terms[i],
                    mass_term=mass_terms[i],
                    eq_attrs=eqs_attrs[i],
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                    scaling_factor=scaling_factors[i],
                )
            )

    def setup_free_surface(self):
        # Overload method
        for free_surface_id, free_surface_params in self.free_surface_dict.items():
            # First, make the displacement term implicit by incorporating
            # the unknown `incremental displacement' (u) that
            # we are solving for
            implicit_displacement_up = fd.dot(self.u, self.k)
            # Add free surface stress term. This is also referred to as the Hydrostatic Prestress advection term in the GIA literature.
            normal_stress, _ = self.approximation.free_surface_terms(
                implicit_displacement_up, **free_surface_params
            )
            if 'normal_stress' in self.weak_bcs[free_surface_id]:
                # Usually there will be also an ice/water loadi acting as a normal stress in the GIA problem
                existing_value = self.weak_bcs[free_surface_id]['normal_stress']
                self.weak_bcs[free_surface_id]['normal_stress'] = existing_value + normal_stress
            else:
                self.weak_bcs[free_surface_id] = {'normal_stress': normal_stress}
