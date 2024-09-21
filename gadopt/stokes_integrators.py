r"""This module provides a fine-tuned solver class for the Stokes system of conservation
equations and a function to automatically set the associated null spaces. Users
instantiate the `StokesSolver` class by providing relevant parameters and call the
`solve` method to request a solver update.

"""

from numbers import Number
from typing import Optional

import firedrake as fd

from .approximations import Approximation
from .equations import Equation
from .free_surface_equation import free_surface_term
from .free_surface_equation import mass_term as mass_term_fs
from .momentum_equation import residual_terms_stokes as terms_stokes
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


def create_stokes_nullspace(
    Z: fd.functionspaceimpl.WithGeometry,
    closed: bool = True,
    rotational: bool = False,
    translations: Optional[list[int]] = None,
    approximation: Optional[Approximation] = None,
    top_subdomain_id: Optional[str | int] = None,
) -> fd.nullspace.MixedVectorSpaceBasis:
    """Create a null space for the mixed Stokes system.

    Arguments:
      Z:
        Firedrake mixed function space associated with the Stokes system
      closed:
        Whether to include a constant pressure null space
      rotational:
        Whether to include all rotational modes
      translations:
        List of translations to include
      approximation:
        AnelasticLiquidApproximation for calculating (non-constant) right nullspace
      top_subdomain_id:
        Boundary id of top surface. Required when providing approximation.

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

    X = fd.SpatialCoordinate(Z.mesh())
    dim = len(X)

    if rotational:
        if dim == 2:
            rotV = fd.Function(Z[0]).interpolate(fd.as_vector((-X[1], X[0])))
            basis = [rotV]
        elif dim == 3:
            x_rotV = fd.Function(Z[0]).interpolate(fd.as_vector((0, -X[2], X[1])))
            y_rotV = fd.Function(Z[0]).interpolate(fd.as_vector((X[2], 0, -X[0])))
            z_rotV = fd.Function(Z[0]).interpolate(fd.as_vector((-X[1], X[0], 0)))
            basis = [x_rotV, y_rotV, z_rotV]
        else:
            raise ValueError("Unknown dimension")
    else:
        basis = []

    if translations:
        for tdim in translations:
            vec = [0] * dim
            vec[tdim] = 1
            basis.append(fd.Function(Z[0]).interpolate(fd.as_vector(vec)))

    if basis:
        V_nullspace = fd.VectorSpaceBasis(basis, comm=Z.mesh().comm)
        V_nullspace.orthonormalize()
    else:
        V_nullspace = Z[0]

    if closed:
        if approximation and approximation.preset == "ALA":
            p = ala_right_nullspace(Z[1], approximation, top_subdomain_id)
            p_nullspace = fd.VectorSpaceBasis([p], comm=Z.mesh().comm)
            p_nullspace.orthonormalize()
        else:
            p_nullspace = fd.VectorSpaceBasis(constant=True, comm=Z.mesh().comm)
    else:
        p_nullspace = Z[1]

    null_space = [V_nullspace, p_nullspace]

    # If free surface unknowns, add dummy free surface nullspace
    null_space += Z[2:]

    return fd.MixedVectorSpaceBasis(Z, null_space)


class StokesSolver:
    """Solves the Stokes system in place.

    Arguments:
      approximation:
        Approximation describing system of equations.
      z:
        Firedrake function representing mixed Stokes system.
      T:
        Firedrake function representing temperature.
      bcs:
        Dictionary of identifier-value pairs specifying boundary conditions.
      quad_degree:
        Quadrature degree. Default value is `2p + 1`, where p is the polynomial degree
        of the trial space.
      solver_parameters:
        Either a dictionary of PETSc solver parameters or a string specifying a default
        set of parameters defined in G-ADOPT
      J:
        Firedrake function representing the Jacobian of the system.
      constant_jacobian:
        Whether the Jacobian of the system is constant.
      nullspace:
        Dictionary of nullspace options.
      timestep_fs:
        Timestep for advancing free surface equation.
      theta_fs:
        Timestepping prefactor for free surface equation.
        theta = 0: Forward Euler
        theta = 0.5: Crank-Nicolson (default)
        theta = 1: Backward Euler
    """

    name = "Stokes"

    def __init__(
        self,
        approximation: Approximation,
        z: fd.Function,
        T: fd.Function = None,
        bcs: dict[int, dict[str, Number]] = {},
        quad_degree: int = 6,
        solver_parameters: Optional[dict[str, str | Number] | str] = None,
        J: Optional[fd.Function] = None,
        constant_jacobian: bool = False,
        nullspace: Optional[dict[str, float]] = {},
        timestep_fs: Optional[float] = None,
        theta_fs: float = 0.5,
    ) -> None:
        self.approximation = approximation
        self.solution = z
        self.T = T or fd.Constant(0)
        self.bcs = bcs
        self.quad_degree = quad_degree
        self.J = J
        self.constant_jacobian = constant_jacobian
        self.nullspace = nullspace
        self.timestep_fs = timestep_fs
        self.theta_fs = theta_fs

        self.Z = z.function_space()
        self.mesh = self.Z.mesh()
        self.tests = fd.TestFunctions(self.Z)
        self.split_solution = fd.split(z)

        self.F = 0

        self.set_boundary_conditions()

        if isinstance(solver_parameters, dict):
            self.solver_parameters = solver_parameters
        else:
            self.set_solver_parameters(solver_parameters)

        self.set_equation()

        # Solver object is set up later to permit editing default solver parameters.
        self._solver_setup = False

    def set_boundary_conditions(self) -> None:
        """Sets up boundary conditions."""
        self.weak_bcs = {}
        self.strong_bcs = []

        self.free_surface = {}
        self.eta_old = [None] * (len(self.split_solution) - 2)

        for bc_id, bc in self.bcs.items():
            weak_bc = {}
            for bc_type, value in bc.items():
                if bc_type == "u":
                    self.strong_bcs.append(fd.DirichletBC(self.Z.sub(0), value, bc_id))
                elif bc_type == "ux":
                    self.strong_bcs.append(
                        fd.DirichletBC(self.Z.sub(0).sub(0), value, bc_id)
                    )
                elif bc_type == "uy":
                    self.strong_bcs.append(
                        fd.DirichletBC(self.Z.sub(0).sub(1), value, bc_id)
                    )
                elif bc_type == "uz":
                    self.strong_bcs.append(
                        fd.DirichletBC(self.Z.sub(0).sub(2), value, bc_id)
                    )
                elif bc_type == "free_surface":
                    weak_bc["normal_stress"] = self.set_free_surface(bc_id, value)
                else:
                    weak_bc[bc_type] = value
            self.weak_bcs[bc_id] = weak_bc

    def set_free_surface(
        self, bc_id: int, params_fs: dict[str, int | bool]
    ) -> fd.ufl.algebra.Product | fd.ufl.algebra.Sum:
        """Sets the given boundary as a free surface.

        This method populates the `free_surface` dictionary (used for preconditioning),
        and `eta_old` list (used to track free surface evolution). It also sets an
        interior boundary condition away from the free surface boundary. Finally, it
        adds the free surface contribution to the Stokes residual.

        Arguments:
          bc_id;
            An integer representing the index of the mesh boundary.
          params_fs:
            A dictionary holding information about the free surface boundary.

        Returns:
          A UFL expression for the normal stress at the free surface boundary.
        """
        eta_ind = params_fs["eta_index"]
        self.free_surface[bc_id] = eta_ind

        u, p = self.split_solution[:2]
        eta = self.split_solution[2 + eta_ind]
        func_space_eta = self.Z[2 + eta_ind]
        self.eta_old[eta_ind] = fd.Function(func_space_eta).interpolate(eta)

        # Set internal dofs to zero to prevent singular matrix for free
        # surface equation.
        self.strong_bcs.append(InteriorBC(func_space_eta, 0, bc_id))

        # Depending on the variable_rho_fs boolean, the interior density below the free
        # surface is either set to a constant density or varies spatially according to
        # the buoyancy field.
        # N.B. A constant reference density is needed for analytical cylindrical cases.
        eta_theta = self.theta_fs * eta + (1 - self.theta_fs) * self.eta_old[eta_ind]
        if self.approximation.dimensional:
            buoyancy = params_fs["rho_diff"] * self.approximation.g
        else:
            buoyancy = params_fs["Ra_fs"]
        normal_stress = buoyancy * eta_theta
        if params_fs.get("variable_rho_fs", True):
            normal_stress -= self.approximation.buoyancy(p, self.T) * eta_theta

        # To ensure the top right and bottom left corners of the block matrix remain
        # symmetric we need to multiply the free surface equation (kinematic bc) with
        # `-theta * delta_rho * g`. This is similar to rescaling eta -> eta_tilde in
        # Kramer et al. (2012); e.g. see block matrix shown in Equation 23.
        # N.B. In the case where the density contrast across the free surface is
        # spatially variant due to interior buoyancy changes then the matrix will not be
        # exactly symmetric.
        rescale_factor = -self.theta_fs * buoyancy

        terms_kwargs = {"free_surface_id": bc_id, "u": u}

        eq_fs = Equation(
            self.tests[2 + eta_ind],
            func_space_eta,
            free_surface_term,
            mass_term=mass_term_fs,
            terms_kwargs=terms_kwargs,
            quad_degree=self.quad_degree,
            rescale_factor=rescale_factor,
        )
        self.F -= eq_fs.residual(eta)
        self.F += eq_fs.mass((eta - self.eta_old[eta_ind]) / self.timestep_fs)

        self.ds_fs = eq_fs.ds

        return normal_stress

    def set_solver_parameters(self, solver_parameters) -> None:
        """Sets PETSc solver parameters."""
        if not depends_on(self.approximation.mu, self.solution):
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
                    self.solver_parameters.update(iterative_stokes_solver_parameters)
                case _:
                    raise ValueError(
                        f"Solver type '{solver_parameters}' not implemented."
                    )
        elif self.mesh.topological_dimension() == 2 and self.mesh.cartesian:
            self.solver_parameters.update(direct_stokes_solver_parameters)
        else:
            self.solver_parameters.update(iterative_stokes_solver_parameters)

        if self.solver_parameters.get("pc_type") == "fieldsplit":
            # extra options for iterative solvers
            if DEBUG >= log_level:
                self.solver_parameters["fieldsplit_0"]["ksp_converged_reason"] = None
                self.solver_parameters["fieldsplit_1"]["ksp_monitor"] = None
            elif INFO >= log_level:
                self.solver_parameters["fieldsplit_1"]["ksp_converged_reason"] = None

            if self.free_surface:
                # merge free surface fields with pressure field for Schur complement solve
                fields = ",".join(map(str, range(1, len(self.split_solution))))
                self.solver_parameters.update(
                    {"pc_fieldsplit_0_fields": "0", "pc_fieldsplit_1_fields": fields}
                )

                # update keys for GADOPT's free surface mass inverse preconditioner
                self.solver_parameters["fieldsplit_1"].update(
                    {"pc_python_type": "gadopt.FreeSurfaceMassInvPC"}
                )

    def set_equation(self) -> None:
        """Sets up UFL forms for the Stokes equations residual."""
        terms_kwargs = {
            "u": self.split_solution[0],
            "p": self.split_solution[1],
            "T": self.T,
        }
        for i, terms_eq in enumerate(terms_stokes):
            eq = Equation(
                self.tests[i],
                self.Z[i],
                terms_eq,
                terms_kwargs=terms_kwargs,
                approximation=self.approximation,
                bcs=self.weak_bcs,
                quad_degree=self.quad_degree,
            )
            self.F -= eq.residual(self.split_solution[i])

    def setup_solver(self) -> None:
        """Sets up the solver."""
        appctx = {"mu": self.approximation.mu / self.approximation.rho}  # fd.MassInvPC
        if self.free_surface:
            appctx["free_surface"] = self.free_surface
            appctx["ds"] = self.ds_fs

        if self.constant_jacobian:
            z_tri = fd.TrialFunction(self.Z)
            F_stokes_lin = fd.replace(self.F, {self.solution: z_tri})
            a, L = fd.lhs(F_stokes_lin), fd.rhs(F_stokes_lin)
            self.problem = fd.LinearVariationalProblem(
                a, L, self.solution, bcs=self.strong_bcs, constant_jacobian=True
            )
            self.solver = fd.LinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                **self.nullspace,
                appctx=appctx,
            )
        else:
            self.problem = fd.NonlinearVariationalProblem(
                self.F, self.solution, bcs=self.strong_bcs, J=self.J
            )
            self.solver = fd.NonlinearVariationalSolver(
                self.problem,
                solver_parameters=self.solver_parameters,
                options_prefix=self.name,
                **self.nullspace,
                appctx=appctx,
            )
        self._solver_setup = True

    def solve(self) -> None:
        """Solves the system."""
        if not self._solver_setup:
            self.setup_solver()

        # Need to update old free surface height for implicit free surface
        for eta, eta_old in zip(self.split_solution[2:], self.eta_old):
            eta_old.interpolate(eta)

        self.solver.solve()


def ala_right_nullspace(
    W: fd.functionspaceimpl.WithGeometry,
    approximation: Approximation,
    top_subdomain_id: str | int,
) -> fd.Function:
    r"""Compute pressure nullspace for Anelastic Liquid Approximation.

    Arguments:
      W: pressure function space
      approximation: AnelasticLiquidApproximation with equation parameters
      top_subdomain_id: boundary id of top surface

    Returns:
      pressure nullspace solution

    To obtain the pressure nullspace solution for the Stokes equation in the
    Anelastic Liquid Approximation (which includes a pressure-dependent buoyancy
    term), we try to solve the equation:

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

    This elliptic equation can be solved with natural boundary conditions by
    imposing our original equation above, which eliminates all boundary terms:

    $$
      int_Omega nabla q * nabla p dx - int_Omega nabla q * hatk g "Di" rho chi c_p/(c_v gamma) p dx = 0.
    $$

    However, if we do so on all boundaries we end up with a system that has the
    same nullspace as the one we are after (note that we ended up merely testing the
    original equation with $nabla q$). Instead, we use the fact that the gradient of
    the null mode is always vertical, and thus the null mode is constant at any
    horizontal level (geoid), such as the top surface. Choosing any nonzero constant
    for this surface fixes the arbitrary scalar multiplier of the null mode. We
    choose the value of one and apply it as a Dirichlet boundary condition.

    Note that this procedure does not necessarily compute the exact nullspace of the
    *discretised* Stokes system. In particular, since not every test function $v$ in
    $V$, the velocity test space, can be written as $v=nabla q$ with $q in W$, the
    pressure test space, the two terms do not necessarily exactly cancel when tested
    with $v$ instead of $nabla q$ as in our final equation. However, in practice,
    the discrete error appears to be small enough, and providing this nullspace
    gives an improved convergence of the iterative Stokes solver.
    """
    W = fd.FunctionSpace(mesh=W.mesh(), family=W.ufl_element())
    q = fd.TestFunction(W)
    p = fd.Function(W, name="Pressure nullspace")
    k = upward_normal(W.mesh())

    F = fd.inner(fd.grad(q), fd.grad(p)) * fd.dx
    F -= fd.inner(fd.grad(q), -k * approximation.compressible_buoyancy * p) * fd.dx

    # Fix the solution at the top boundary
    fd.solve(F == 0, p, bcs=fd.DirichletBC(W, 1.0, top_subdomain_id))

    return p
