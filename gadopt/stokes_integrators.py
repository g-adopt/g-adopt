r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

from numbers import Number
from typing import Optional

import firedrake as fd

from .approximations import BaseApproximation, AnelasticLiquidApproximation
from .momentum_equation import StokesEquations
from .utility import DEBUG, INFO, depends_on, ensure_constant, log_level, upward_normal

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
        if ala_approximation:
            p = ala_right_nullspace(W=W, approximation=ala_approximation, top_subdomain_id=top_subdomain_id)
            p_nullspace = fd.VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = W

    return fd.MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace])


class StokesSolver:
    """Solves the Stokes system in place.

    Arguments:
      z: Firedrake function representing mixed Stokes system
      T: Firedrake function representing temperature
      approximation: Approximation describing system of equations
      bcs: Dictionary of identifier-value pairs specifying boundary conditions
      mu: Firedrake function representing dynamic viscosity
      quad_degree: Quadrature degree. Default value is `2p + 1`, where
                   p is the polynomial degree of the trial space
      solver_parameters: Either a dictionary of PETSc solver parameters or a string
                         specifying a default set of parameters defined in G-ADOPT
      J: Firedrake function representing the Jacobian of the system
      constant_jacobian: Whether the Jacobian of the system is constant

    """

    name = "Stokes"

    def __init__(
        self,
        z: fd.Function,
        T: fd.Function,
        approximation: BaseApproximation,
        bcs: dict[int, dict[str, Number]] = {},
        mu: fd.Function | Number = 1,
        quad_degree: int = 6,
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        J: Optional[fd.Function] = None,
        constant_jacobian: bool = False,
        **kwargs,
    ):
        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.test = fd.TestFunctions(self.Z)
        self.equations = StokesEquations(self.Z, self.Z, quad_degree=quad_degree,
                                         compressible=approximation.compressible)
        self.solution = z
        self.solution_old = None
        self.approximation = approximation
        self.mu = ensure_constant(mu)
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.linear = not depends_on(self.mu, self.solution)

        self.solver_kwargs = kwargs
        u, p = fd.split(self.solution)
        self.k = upward_normal(self.Z.mesh())
        self.fields = {
            'velocity': u,
            'pressure': p,
            'viscosity': self.mu,
            'interior_penalty': fd.Constant(2.0),  # allows for some wiggle room in imposition of weak BCs
                                                   # 6.25 matches C_ip=100. in "old" code for Q2Q1 in 2d.
            'source': self.approximation.buoyancy(p, T) * self.k,
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

        self.F = 0
        for test, eq, u in zip(self.test, self.equations, fd.split(self.solution)):
            self.F -= eq.residual(test, u, u, self.fields, bcs=self.weak_bcs)

        if isinstance(solver_parameters, dict):
            self.solver_parameters = solver_parameters
        else:
            if self.linear:
                self.solver_parameters = {"snes_type": "ksponly"}
            else:
                self.solver_parameters = newton_stokes_solver_parameters.copy()

            if INFO >= log_level:
                self.solver_parameters["snes_monitor"] = None

            if isinstance(solver_parameters, str):
                match solver_parameters:
                    case "direct":
                        self.solver_parameters.update(direct_stokes_solver_parameters)
                    case "iterative":
                        self.solver_parameters.update(
                            iterative_stokes_solver_parameters
                        )
                    case _:
                        raise ValueError(
                            f"Solver type '{solver_parameters}' not implemented."
                        )
            elif self.mesh.topological_dimension() == 2 and self.mesh.cartesian:
                self.solver_parameters.update(direct_stokes_solver_parameters)
            else:
                self.solver_parameters.update(iterative_stokes_solver_parameters)

                if DEBUG >= log_level:
                    self.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
                    self.solver_parameters['fieldsplit_1']['ksp_monitor'] = None
                elif INFO >= log_level:
                    self.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None

        # solver object is set up later to permit editing default solver parameters specified above
        self._solver_setup = False

    def setup_solver(self):
        """Sets up the solver."""
        # mu used in MassInvPC:
        mu_over_rho = self.mu / self.approximation.rho_continuity()
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
                                                     appctx={"mu": mu_over_rho},
                                                     **self.solver_kwargs)
        else:
            self.problem = fd.NonlinearVariationalProblem(self.F, self.solution,
                                                          bcs=self.strong_bcs, J=self.J)
            self.solver = fd.NonlinearVariationalSolver(self.problem,
                                                        solver_parameters=self.solver_parameters,
                                                        options_prefix=self.name,
                                                        appctx={"mu": mu_over_rho},
                                                        **self.solver_kwargs)
        self._solver_setup = True

    def solve(self):
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()
        self.solution_old = self.solution.copy(deepcopy=True)
        self.solver.solve()


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

    k = upward_normal(W.mesh(), cartesian=approximation.cartesian)

    F += - fd.inner(fd.grad(q), k * approximation.dbuoyancydp(p, fd.Constant(1.0)) * p) * fd.dx

    fd.solve(F == 0, p, bcs=bc)
    return p
