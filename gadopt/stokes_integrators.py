r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

import abc
from collections import defaultdict
from numbers import Number
from typing import Any, Optional
from warnings import warn

import firedrake as fd
from ufl.core.expr import Expr

from .approximations import AnelasticLiquidApproximation, BaseApproximation
from .equations import Equation
from .free_surface_equation import free_surface_term
from .free_surface_equation import mass_term as mass_term_fs
from .momentum_equation import compressible_viscoelastic_terms, stokes_terms
from .utility import (
    DEBUG,
    INFO,
    InteriorBC,
    depends_on,
    is_continuous,
    log_level,
    upward_normal,
    vertical_component,
)

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
        raise ValueError(
            "Both ala_approximation and top_subdomain_id must be provided, or both must be None."
        )

    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)
    stokes_subspaces = Z.subspaces

    if rotational:
        if dim == 2:
            rotV = fd.Function(stokes_subspaces[0]).interpolate(
                fd.as_vector((-X[1], X[0]))
            )
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(stokes_subspaces[0]).interpolate(
                fd.as_vector((0, -X[2], X[1]))
            )
            y_rotV = fd.Function(stokes_subspaces[0]).interpolate(
                fd.as_vector((X[2], 0, -X[0]))
            )
            z_rotV = fd.Function(stokes_subspaces[0]).interpolate(
                fd.as_vector((-X[1], X[0], 0))
            )
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(
                fd.Function(stokes_subspaces[0]).interpolate(fd.as_vector(vec))
            )

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = stokes_subspaces[0]

    if closed:
        if ala_approximation:
            p = ala_right_nullspace(
                W=stokes_subspaces[1],
                approximation=ala_approximation,
                top_subdomain_id=top_subdomain_id,
            )
            p_nullspace = fd.VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = stokes_subspaces[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface nullspace
    null_space += stokes_subspaces[2:]

    return V_nullspace  # fd.MixedVectorSpaceBasis(Z, null_space)


class MetaPostInit(abc.ABCMeta):
    """Calls the implemented `prepare_solver` method after `__init__` returns.

    The implemented behaviour allows any subclass `__init__` method to first call its
    parent class's `__init__` through super(), then execute its own code, and finally
    call `prepare_solver`. The latter call is automatic and does not require any
    attention from the developer or user.
    """

    def __call__(cls, *args, **kwargs):
        class_instance = super().__call__(*args, **kwargs)
        class_instance.prepare_solver()

        return class_instance


class SolverBase(abc.ABC, metaclass=MetaPostInit):
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
    | eta_index        | Yes/No   | Function index in mixed space formulation      |
    | rho_ext          | No       | Exterior density along the free surface        |
    | Ra_fs            | No       | Rayleigh number at the free surface            |
    | include_buoyancy | No       | Whether the interior density includes buoyancy |

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
      dt:
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

    name = "CoupledMomentum"

    def __init__(
        self,
        solution: fd.Function,
        approximation: BaseApproximation,
        /,
        *,
        dt: float | None = None,
        theta: float = 0.5,
        bcs: dict[int, dict[str, Any]] = {},
        quad_degree: int = 6,
        solver_parameters: dict[str, str | Number] | str | None = None,
        J: fd.Function | None = None,
        constant_jacobian: bool = False,
        nullspace: fd.MixedVectorSpaceBasis = None,
        transpose_nullspace: fd.MixedVectorSpaceBasis = None,
        near_nullspace: fd.MixedVectorSpaceBasis = None,
    ) -> None:
        self.solution = solution
        self.approximation = approximation
        self.dt = dt
        self.theta = theta
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.solver_parameters = solver_parameters
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self.near_nullspace = near_nullspace

        self.solution_old = solution.copy(deepcopy=True)
        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.k = upward_normal(self.mesh)

        self.is_mixed_space = isinstance(
            self.solution_space.topological, fd.functionspaceimpl.MixedFunctionSpace
        )
        if self.is_mixed_space:
            self.tests = fd.TestFunctions(self.solution_space)
            self.solution_split = fd.split(solution)
            self.solution_old_split = fd.split(self.solution_old)
        else:
            self.test = fd.TestFunction(self.solution_space)
            self.solution_split = (solution,)
            self.solution_old_split = (self.solution_old,)

        self.solution_theta_split = [
            theta * sol + (1 - theta) * sol_old
            for sol, sol_old in zip(self.solution_split, self.solution_old_split)
        ]

        self.equations = []
        self.F = 0.0  # Weak form of the system

        # Solver object is set up later to permit editing default solver options.
        self._solver_ready = False

    def prepare_solver(self) -> None:
        """Runs methods that set up arguments for the variational problem and solver."""
        self.set_boundary_conditions()
        self.set_equations()
        self.set_form()
        self.set_solver_options()

    def set_boundary_conditions(self) -> None:
        """Sets strong and weak boundary conditions."""
        self.strong_bcs = []
        self.weak_bcs = {}

        if self.is_mixed_space:
            bc_map = {"u": self.solution_space.sub(0)}
        else:
            bc_map = {"u": self.solution_space}
        if self.mesh.cartesian:
            bc_map["ux"] = bc_map["u"].sub(0)
            bc_map["uy"] = bc_map["u"].sub(1)
            if self.mesh.geometric_dimension() == 3:
                bc_map["uz"] = bc_map["u"].sub(2)

        for bc_id, bc in self.bcs.items():
            weak_bc = defaultdict(float)

            for bc_type, val in bc.items():
                match bc_type:
                    case "u" | "ux" | "uy" | "uz":
                        self.strong_bcs.append(
                            fd.DirichletBC(bc_map[bc_type], val, bc_id)
                        )
                    case "free_surface":
                        weak_bc["normal_stress"] += self.set_free_surface_boundary(
                            val, bc_id
                        )
                    case _:
                        weak_bc[bc_type] += val

            self.weak_bcs[bc_id] = weak_bc

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> fd.ufl.algebra.Product | fd.ufl.algebra.Sum:
        """Sets the given boundary as a free surface.

        This method calculates the normal stress at the free surface boundary. In the
        coupled approach, it also sets a zero-interior strong condition away from that
        boundary and populates the `free_surface_map` dictionary used to calculate the
        free-surface contribution to the momentum weak form.

        Args:
          params_fs:
            Dictionary holding information about the free surface boundary
          bc_id:
            Integer representing the index of the mesh boundary

        Returns:
          UFL expression for the normal stress at the free surface boundary
        """

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
                assert equation.scaling_factor == -self.theta
                self.F += equation.mass((solution - solution_old) / self.dt)
            self.F -= equation.residual(solution)

    def set_solver_options(self) -> None:
        """Sets PETSc solver parameters."""
        # Application context for the inverse mass matrix preconditioner
        self.appctx = {
            "mu": self.approximation.mu / self.approximation.rho_continuity()
        }

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
            trial = fd.TrialFunction(self.solution_space)
            F = fd.replace(self.F, {self.solution: trial})
            a, L = fd.lhs(F), fd.rhs(F)

            self.problem = fd.LinearVariationalProblem(
                a, L, self.solution, bcs=self.strong_bcs, constant_jacobian=True
            )
            self.solver = fd.LinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                nullspace=self.nullspace,
                transpose_nullspace=self.transpose_nullspace,
                near_nullspace=self.near_nullspace,
                appctx=self.appctx,
                options_prefix=self.name,
            )
        else:
            self.problem = fd.NonlinearVariationalProblem(
                self.F, self.solution, bcs=self.strong_bcs, J=self.J
            )
            self.solver = fd.NonlinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                nullspace=self.nullspace,
                transpose_nullspace=self.transpose_nullspace,
                near_nullspace=self.near_nullspace,
                appctx=self.appctx,
                options_prefix=self.name,
            )

        self._solver_ready = True

    def solve(self) -> None:
        """Solves the system."""
        if not self._solver_ready:
            self.setup_solver()

        self.solver.solve()
        self.solution_old.assign(self.solution)


class StokesSolver(SolverBase):
    """Solver for the Stokes system.

    Args:
      solution:
        Firedrake function representing the field over the mixed Stokes space
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      T:
        Firedrake function representing the temperature field
      coupled_fields:
        Dictionary in a coupled time integration
      dt:
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
        approximation: BaseApproximation,
        T: fd.Function | float = 0.0,
        /,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, **kwargs)

        self.T = T

        self.free_surface_map = {}
        self.eta_ind = 2
        self.buoyancy_fs = [None] * len(self.solution_split)

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> Expr:
        # Associate the free-surface index with the boundary id
        # Note: This assumes that ordering of the free-surface boundary conditions is
        # the same as that of free-surface functions in the mixed space.
        self.free_surface_map[bc_id] = self.eta_ind

        # Set internal degrees of freedom to zero to prevent singular matrix
        self.strong_bcs.append(InteriorBC(self.solution_space[self.eta_ind], 0, bc_id))

        normal_stress, self.buoyancy_fs[self.eta_ind] = (
            self.approximation.free_surface_terms(
                self.solution_split[1],
                self.T,
                self.solution_theta_split[self.eta_ind],
                **params_fs,
            )
        )
        self.eta_ind += 1
        return normal_stress

    def set_equations(self) -> None:
        u, p = self.solution_split[:2]
        stress = self.approximation.stress(u)
        source = self.approximation.buoyancy(p, self.T) * upward_normal(self.mesh)
        eqs_attrs = [
            {"p": p, "stress": stress, "source": source},
            {"u": u, "rho_mass": self.approximation.rho_continuity()},
        ]

        for i in range(len(stokes_terms)):
            self.equations.append(
                Equation(
                    self.tests[i],
                    self.solution_space[i],
                    stokes_terms[i],
                    eq_attrs=eqs_attrs[i],
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                )
            )

        for bc_id, eta_ind in self.free_surface_map.items():
            eq_attrs = {
                "boundary_id": bc_id,
                "buoyancy_scale": self.buoyancy_fs[eta_ind],
                "u": u,
            }

            self.equations.append(
                Equation(
                    self.tests[eta_ind],
                    self.solution_space[eta_ind],
                    free_surface_term,
                    mass_term=mass_term_fs,
                    eq_attrs=eq_attrs,
                    quad_degree=self.quad_degree,
                    scaling_factor=-self.theta,
                )
            )

    def set_solver_options(self) -> None:
        super().set_solver_options()

        if (
            self.free_surface_map
            and self.solver_parameters.get("pc_type") == "fieldsplit"
        ):
            # Update application context
            self.appctx["free_surface"] = self.free_surface_map
            self.appctx["ds"] = self.equations[-1].ds

            # Gather pressure and free surface fields for Schur complement solve
            fields_ind = ",".join(map(str, range(1, len(self.solution_split))))
            self.solver_parameters.update(
                {"pc_fieldsplit_0_fields": "0", "pc_fieldsplit_1_fields": fields_ind}
            )
            # Update mass inverse preconditioner
            self.solver_parameters["fieldsplit_1"].update(
                {"pc_python_type": "gadopt.FreeSurfaceMassInvPC"}
            )

    def force_on_boundary(self, subdomain_id: int | str, **kwargs) -> fd.Function:
        """Computes the force acting on a boundary.

        Arguments:
          subdomain_id: The subdomain ID of a physical boundary.

        Returns:
          The force acting on the boundary.

        """
        if not hasattr(self, "BoundaryNormalStressSolvers"):
            self.BoundaryNormalStressSolvers = {}

        if subdomain_id not in self.BoundaryNormalStressSolvers:
            self.BoundaryNormalStressSolvers[subdomain_id] = BoundaryNormalStressSolver(
                self, subdomain_id, **kwargs
            )

        return self.BoundaryNormalStressSolvers[subdomain_id].solve()


def ala_right_nullspace(
    W: fd.functionspaceimpl.WithGeometry,
    approximation: AnelasticLiquidApproximation,
    top_subdomain_id: str | int,
):
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
    bc = fd.DirichletBC(W, 1.0, top_subdomain_id)

    F = fd.inner(fd.grad(q), fd.grad(p)) * fd.dx

    k = upward_normal(W.mesh())

    F += (
        -fd.inner(fd.grad(q), k * approximation.dbuoyancydp(p, fd.Constant(1.0)) * p)
        * fd.dx
    )

    fd.solve(F == 0, p, bcs=bc)
    return p


class ViscoelasticStokesSolver(SolverBase):
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
        super().__init__(solution, approximation, dt=dt, **kwargs)

        self.u, self.p = self.solution_split[:2]
        self.stress_old = stress_old
        self.displacement = displacement

        # Effective viscosity
        self.approximation.mu = approximation.effective_viscosity(self.dt)

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
            if "normal_stress" in self.weak_bcs[free_surface_id]:
                # Usually there will be also an ice/water loadi acting as a normal stress in the GIA problem
                existing_value = self.weak_bcs[free_surface_id]["normal_stress"]
                self.weak_bcs[free_surface_id]["normal_stress"] = (
                    existing_value + normal_stress
                )
            else:
                self.weak_bcs[free_surface_id] = {"normal_stress": normal_stress}

    def set_equations(self) -> None:
        """Sets up UFL forms for the viscoelastic Stokes equations residual."""
        stress = self.approximation.stress(self.u, self.stress_old, self.dt)
        source = self.approximation.buoyancy(self.displacement) * self.k
        rho_mass = self.approximation.rho_continuity()
        eqs_attrs = [
            {"p": self.p, "stress": stress, "source": source},
            {"u": self.u, "rho_mass": rho_mass},
        ]

        scale_mu = fd.Constant(
            1 / 1e10
        )  # this is a scaling factor roughly size of mantle maxwell time to make sure that solve converges with strong bcs in parallel...
        for i in range(len(stokes_terms)):
            self.equations.append(
                Equation(
                    self.tests[i],
                    self.solution_space[i],
                    stokes_terms[i],
                    eq_attrs=eqs_attrs[i],
                    approximation=self.approximation,
                    bcs=self.weak_bcs,
                    quad_degree=self.quad_degree,
                    scaling_factor=scale_mu,
                )
            )

    def solver_callback(self) -> None:
        # Update history stress term for using as a RHS explicit forcing in the next timestep
        # Interpolating with adjoint seems to need subfunction...
        # otherwise 'map toset must be same as Dataset' error
        u_sub = self.solution.subfunctions[0]
        self.stress_old.interpolate(
            self.approximation.prefactor_prestress(self.dt)
            * self.approximation.stress(u_sub, self.stress_old, self.dt)
        )
        self.displacement.interpolate(self.displacement + u_sub)


class InternalVariableSolver(SolverBase):
    """Solver for internal variable viscoelastic formulation.

    Args:
      solution:
        Firedrake function representing the displacement solution
      approximation:
        G-ADOPT approximation defining terms in the system of equations
      m_list:
        List of internal variables
    """

    name = "InternalVariable"

    def __init__(
        self,
        solution: fd.Function,
        approximation: BaseApproximation,
        /,
        m_list: list,
        **kwargs,
    ) -> None:
        super().__init__(solution, approximation, **kwargs)
        self.m_list = m_list

        # Effective viscosity THIS IS A HACK. need to update SIPG terms for compressibility?
        self.approximation.mu = approximation.viscosity[0]/(self.approximation.maxwell_times[0]+self.dt)
        print('hello gadopt')

    def set_equations(self) -> None:
        self.strain = self.approximation.deviatoric_strain(self.solution)

        m_new_list = [self.update_m(m, alpha) for m, alpha in zip(self.m_list, self.approximation.maxwell_times)]

        stress = self.approximation.stress(self.solution, m_new_list)
        source = self.approximation.buoyancy(self.solution) * self.k

        eqs_attrs = {"stress": stress, "source": source}

        self.equations.append(
            Equation(
                self.test,
                self.solution_space,
                compressible_viscoelastic_terms,
                eq_attrs=eqs_attrs,
                approximation=self.approximation,
                bcs=self.weak_bcs,
                quad_degree=self.quad_degree,
            )
        )

    def set_free_surface_boundary(
        self, params_fs: dict[str, int | bool], bc_id: int
    ) -> Expr:
        normal_stress = params_fs.get("normal_stress", 0.0)
        # Add free surface stress term. This is also referred to as the Hydrostatic
        # Prestress advection term in the GIA literature.
        normal_stress += self.approximation.hydrostatic_prestress_advection(
            vertical_component(self.solution)
        )

        return normal_stress

    def update_m(self, m, alpha):
        # Return updated internal variable using Backward Euler formula
        m_new = (m + self.dt / alpha * self.strain) / (1 + self.dt / alpha)
        return m_new

    def solve(self) -> None:
        super().solve()
        # Update internal variable term for using as a RHS explicit forcing in the next timestep
        for m, alpha in zip(self.m_list, self.approximation.maxwell_times):
            m.interpolate(self.update_m(m, alpha))


class BoundaryNormalStressSolver:
    r"""A class for calculating surface forces acting on a boundary.

    This solver computes topography on boundaries using the equation:

    $$
    h = sigma_(rr) / (g delta rho)
    $$

    where $sigma_(rr)$ is defined as:

    $$
    sigma_(rr) = [-p I + 2 mu (nabla u + nabla u^T)] . hat n . hat n
    $$

    Instead of assuming a unit normal vector $hat n$, this solver uses `FacetNormal`
    from Firedrake to accurately determine the normal vectors, which is particularly
    useful for complex meshes like icosahedron meshes in spherical simulations.

    Arguments:
        stokes_solver (StokesSolver): The Stokes solver providing the necessary fields for calculating stress.
        subdomain_id (str | int): The subdomain ID of a physical boundary.
        **kwargs: Optional keyword arguments. You can provide:
            - solver_parameters (dict[str, str | Number]): Parameters to control the variational solver.
            If not provided, defaults are chosen based on whether the Stokes solver is direct or iterative.
    """

    direct_solve_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    iterative_solver_parameters = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-9,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
        "ksp_converged_reason": None,
    }

    name = "BoundaryNormalStressSolver"

    def __init__(self, stokes_solver: StokesSolver, subdomain_id: int | str, **kwargs):
        # pressure and velocity together with viscosity are needed
        self.u, self.p, *self.eta = stokes_solver.solution.subfunctions

        # geometry
        self.mesh = stokes_solver.mesh
        self.dim = self.mesh.geometric_dimension()

        # approximation tells us if we need to consider compressible formulation or not
        self.approximation = stokes_solver.approximation

        # Domain id that we want to use for boundary force
        self.subdomain_id = subdomain_id

        self._kwargs = kwargs

        self.solver_parameters = self._kwargs.get(
            "solver_parameters",
            BoundaryNormalStressSolver.direct_solve_parameters
            if stokes_solver.solver_parameters == direct_stokes_solver_parameters
            else BoundaryNormalStressSolver.iterative_solver_parameters,
        )

        self._solver_is_set_up = False

    def solve(self):
        """
        Solves a linear system for the force and applies necessary boundary conditions.

        Returns:
            The modified force after solving the linear system and applying boundary conditions.
        """
        # Solve a linear system
        if not self._solver_is_set_up:
            self.setup_solver()
        # Solve for the force
        self.solver.solve()

        # Take the average out
        vave = fd.assemble(self.force * self.ds) / fd.assemble(1 * self.ds)
        self.force.assign(self.force - vave)

        # Re-apply the zero condition everywhere except for the boundary
        self.interior_null_bc.apply(self.force)

        return self.force

    def setup_solver(self):
        # Define the solution in the pressure function space
        # Pressure is chosen as it has a lower rank compared to velocity
        # If pressure is discontinuous, we need to use a continuous equivalent
        if not is_continuous(self.p):
            warn(
                "BoundaryNormalStressSolver: Pressure field is discontinuous. Using an equivalent continous lagrange element."
            )
            Q = fd.FunctionSpace(
                self.mesh, "Lagrange", self.p.function_space().ufl_element().degree()
            )
        else:
            Q = fd.FunctionSpace(self.mesh, self.p.ufl_element())

        self.force = fd.Function(Q, name=f"force_{self.subdomain_id}")

        # Normal vector
        n = fd.FacetNormal(self.mesh)

        phi = fd.TestFunction(Q)
        v = fd.TrialFunction(Q)

        stress_with_pressure = self.approximation.stress(self.u) - self.p * fd.Identity(
            self.dim
        )

        ds_kwargs = {
            "domain": self.mesh,
            "degree": self._kwargs.get("quad_degree", None),
        }
        if self.mesh.extruded and self.subdomain_id in ["top", "bottom"]:
            self.ds = (fd.ds_t if self.subdomain_id == "top" else fd.ds_b)(**ds_kwargs)
        else:
            self.ds = fd.ds(self.subdomain_id, **ds_kwargs)

        # Setting up the variational problem
        a = phi * v * self.ds
        L = -phi * fd.dot(fd.dot(stress_with_pressure, n), n) * self.ds

        # Setting up boundary condition, problem and solver
        # The field is only meaningful on the boundary, so set zero everywhere else
        self.interior_null_bc = InteriorBC(Q, 0.0, [self.subdomain_id])

        self.problem = fd.LinearVariationalProblem(
            a, L, self.force, bcs=self.interior_null_bc, constant_jacobian=True
        )
        self.solver = fd.LinearVariationalSolver(
            self.problem,
            solver_parameters=self.solver_parameters,
            options_prefix=f"{BoundaryNormalStressSolver.name}_{self.subdomain_id}",
        )
        self._solver_is_set_up = True
