import firedrake as fd
from ufl.algebra import Operator

from gadopt import GenericTransportSolver, rigid_body_modes
from gadopt.stokes_integrators import direct_stokes_solver_parameters
from gadopt.transport_solver import direct_energy_solver_parameters
from gadopt.utility import upward_normal


class Approximation:
    def __init__(
        self, name: str, ref_state: dict[str, float], ref_profiles: dict[str, float]
    ) -> None:
        self.name = name
        self.ref_state = ref_state
        self.ref_profiles = ref_state | ref_profiles

    def density(self, p: fd.Function, T: fd.Function, delta_rho: Operator) -> Operator:
        match self.name:
            case "BA" | "EBA" | "TALA":
                return (
                    self.ref_profiles["rho"] * (1.0 - self.ref_profiles["alpha"] * T)
                    + delta_rho
                )
            case "ALA":
                return (
                    self.ref_profiles["rho"]
                    * (
                        1.0
                        + p / self.ref_profiles["K"]
                        - self.ref_profiles["alpha"] * T
                    )
                    + delta_rho
                )
            case "ICA" | "HCA" | "PDA":
                return (self.ref_state["rho"] + delta_rho) * fd.exp(
                    (self.ref_profiles["p"] + p - self.ref_state["p"])
                    / self.ref_profiles["K"]
                    - self.ref_profiles["alpha"]
                    * (self.ref_profiles["T"] + T - self.ref_state["T"])
                )

    def momentum_buoyancy(
        self, p: fd.Function, T: fd.Function, delta_rho: Operator
    ) -> Operator:
        return (
            self.density(p, T, delta_rho) - self.ref_profiles["rho"]
        ) * self.ref_profiles["g"]

    def strain_rate(self, u: fd.Function) -> Operator:
        return fd.sym(fd.grad(u))

    def shear_stress(self, u: fd.Function, effective_viscosity: Operator) -> Operator:
        return 2.0 * effective_viscosity * fd.dev(self.strain_rate(u))


class StokesSolver:
    def __init__(
        self,
        solution: fd.Function,
        apx: Approximation,
        viscosity=None,
        T=0.0,
        T_old=0.0,
        delta_rho=0.0,
        delta_rho_old=0.0,
        time_step=None,
        strong_bcs=None,
        weak_bcs=None,
        free_surface_id=None,
        solver_parameters=None,
        nullspace=None,
        transpose_nullspace=None,
    ) -> None:
        self.solution = solution
        self.solution_old = fd.Function(solution)
        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.apx = apx
        self.viscosity = viscosity or apx.ref_profiles["eta"]
        self.T = T
        self.T_old = T_old
        self.delta_rho = delta_rho
        self.delta_rho_old = delta_rho_old
        self.time_step = time_step
        self.weak_bcs = weak_bcs or {}
        self.free_surface_id = free_surface_id

        self.set_weak_form()

        solver_parameters = (
            solver_parameters
            or {"snes_type": "ksponly", "snes_monitor": None}
            | direct_stokes_solver_parameters
        )

        if nullspace is not None:
            nullspace = self.null_space(**nullspace)
        if transpose_nullspace is not None:
            transpose_nullspace = self.null_space(**transpose_nullspace)

        variational_problem = fd.NonlinearVariationalProblem(
            self.residual(), solution, bcs=strong_bcs
        )
        self.solver = fd.NonlinearVariationalSolver(
            variational_problem,
            solver_parameters=solver_parameters,
            options_prefix="Stokes",
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
        )

    def set_weak_form(self) -> None:
        self.tests = fd.TestFunctions(self.solution_space)
        self.n = fd.FacetNormal(self.mesh)
        self.n_up = upward_normal(self.mesh)
        self.dx = fd.dx(degree=6)
        self.ds = fd.ds(degree=6)

        u, p = fd.split(self.solution)[:2]

        match self.apx.name:
            case "BA" | "EBA" | "TALA" | "ALA":
                density_kwargs = {"p": p, "T": self.T, "delta_rho": self.delta_rho}

                self.rho = self.apx.ref_profiles["rho"]
            case "ICA":
                density_kwargs = {"p": 0.0, "T": self.T, "delta_rho": self.delta_rho}

                self.K = self.apx.ref_profiles["K"]
                self.rho = self.apx.density(**density_kwargs)
            case "HCA":
                density_kwargs = {"p": 0.0, "T": self.T, "delta_rho": self.delta_rho}

                self.alpha = self.apx.ref_profiles["alpha"]
                self.K = self.apx.ref_profiles["K"]
                self.T_ref = self.apx.ref_profiles["T"]
                self.rho = self.apx.density(**density_kwargs)
            case "PDA":
                density_kwargs = {"p": 0.0, "T": self.T, "delta_rho": self.delta_rho}

                self.rho = self.apx.density(**density_kwargs)
                self.rho_old = self.apx.density(0.0, self.T_old, self.delta_rho_old)

        self.g = -self.apx.ref_profiles["g"] * self.n_up

        self.buoyancy = self.apx.momentum_buoyancy(**density_kwargs)
        self.stress = self.apx.shear_stress(u, self.viscosity)

    def mass_equation(self) -> fd.Form:
        u = fd.split(self.solution)[0]
        u_old = fd.split(self.solution_old)[0]

        match self.apx.name:
            case "BA" | "EBA":
                return self.tests[1] * fd.div(u) * self.dx
            case "TALA" | "ALA":
                return self.tests[1] * fd.div(self.rho * u) * self.dx
            case "ICA":
                return (
                    self.tests[1]
                    * (fd.div(u) + self.rho / self.K * fd.dot(u, self.g))
                    * self.dx
                )
            case "HCA":
                return (
                    self.tests[1]
                    * (
                        fd.div(u)
                        + self.rho / self.K * fd.dot(u_old, self.g)
                        - self.alpha * fd.dot(u_old, fd.grad(self.T_ref + self.T))
                    )
                    * self.dx
                )
            case "PDA":
                drho_dt = (self.rho - self.rho_old) / self.time_step
                return (
                    self.tests[1]
                    * (drho_dt + fd.dot(u, fd.grad(self.rho)) + self.rho * fd.div(u))
                    * self.dx
                )

    def momentum_equation(self) -> fd.Form:
        p = fd.split(self.solution)[1]

        weak_form = (
            fd.div(self.tests[0]) * p
            - fd.inner(fd.nabla_grad(self.tests[0]), self.stress)
            + fd.dot(self.tests[0], -self.buoyancy * self.n_up)
        ) * self.dx
        # weak_form += -fd.dot(self.tests[0], self.n) * p * self.ds
        for bc_id, bc_dict in self.weak_bcs.items():
            if "traction" in bc_dict:
                weak_form += fd.dot(self.tests[0], bc_dict["traction"]) * self.ds(bc_id)

        return weak_form

    def free_surface_equation(self) -> fd.Form:
        u, _, phi = fd.split(self.solution)
        phi_old = fd.split(self.solution)[2]

        weak_form = -0.5 * (
            -self.tests[2]
            * (self.rho * fd.dot(-self.g, self.n) * (phi - phi_old) / self.time_step)
            * self.ds(self.free_surface_id)
            + self.tests[2]
            * (fd.dot(u, self.n) * self.rho * self.apx.ref_profiles["g"])
            * self.ds(self.free_surface_id)
        ) + fd.dot(self.tests[0], self.n) * (
            self.rho * self.apx.ref_profiles["g"] - self.buoyancy
        ) * 0.5 * (phi + phi_old) * self.ds(self.free_surface_id)

        return weak_form

    def residual(self) -> fd.Form:
        residual = self.mass_equation() + self.momentum_equation()
        if self.free_surface_id is not None:
            residual += self.free_surface_equation()

        return residual

    def null_space(
        self,
        closed: bool,
        rotational: bool,
        translations: list[int] | None,
        boundary_id=None,
    ):
        V_nullspace = rigid_body_modes(
            self.solution_space.subspaces[0],
            rotational=rotational,
            translations=translations,
        )

        if closed:
            if boundary_id is None:
                p_nullspace = fd.VectorSpaceBasis(constant=True, comm=self.mesh.comm)
            else:
                pressure_space = fd.FunctionSpace(
                    mesh=self.mesh,
                    family=self.solution_space.subspaces[1].ufl_element(),
                )
                test = fd.TestFunction(pressure_space)
                kernel = fd.Function(pressure_space, name="Pressure null space")
                buoyancy = -self.apx.momentum_buoyancy(kernel, 0.0, 0.0) * self.n_up

                F = fd.dot(fd.grad(test), fd.grad(kernel) - buoyancy) * self.dx
                bcs = fd.DirichletBC(pressure_space, 1.0, boundary_id)
                fd.solve(F == 0.0, kernel, bcs=bcs)

                p_nullspace = fd.VectorSpaceBasis([kernel], comm=self.mesh.comm)
                p_nullspace.orthonormalize()
        else:
            p_nullspace = self.solution_space.subspaces[1]

        nullspace = [V_nullspace, p_nullspace]
        nullspace += self.solution_space.subspaces[2:]

        return fd.MixedVectorSpaceBasis(self.solution_space, nullspace)

    def solve(self) -> None:
        self.solver.solve()
        self.solution_old.assign(self.solution)


class EnergySolver:
    def __init__(
        self,
        solution: fd.Function,
        apx: Approximation,
        u: fd.Function,
        time_step: fd.Function | float,
        time_stepper,
        viscosity=None,
        delta_rho=0.0,
        strong_bcs=None,
        solver_parameters=None,
        disable_shear_heating=False,
    ) -> None:
        self.solution = solution
        self.solution_space = solution.function_space()
        self.mesh = self.solution_space.mesh()
        self.apx = apx
        self.u = u
        self.viscosity = viscosity or apx.ref_profiles["eta"]
        self.delta_rho = delta_rho
        self.disable_shear_heating = disable_shear_heating

        solver_parameters = (
            solver_parameters
            or {"ksp_converged_reason": None} | direct_energy_solver_parameters
        )

        self.set_weak_form()

        self.trial_space = self.solution_space
        self.timestepper = time_stepper(
            self,
            solution,
            time_step,
            solver_parameters=solver_parameters,
            strong_bcs=strong_bcs,
        )

    def set_weak_form(self) -> None:
        self.test = fd.TestFunction(self.solution_space)
        self.n = fd.FacetNormal(self.mesh)
        self.dx = fd.dx(degree=5)
        self.ds = fd.ds(degree=5)

        for profile in ["alpha", "cp", "H", "p", "T"]:
            setattr(self, profile, self.apx.ref_profiles[profile])
        match self.apx.name:
            case "BA" | "EBA" | "TALA" | "ALA":
                self.rho = self.apx.ref_profiles["rho"]
            case "ICA" | "HCA" | "PDA":
                self.rho = self.apx.density(0.0, self.solution, self.delta_rho)
        self.k = self.apx.ref_profiles["k"] * fd.Identity(2)
        self.stress = self.apx.shear_stress(self.u, self.viscosity)

    def residual(self, trial: fd.Function) -> fd.Form:
        weak_form = (
            fd.div(self.test * self.rho * self.cp * self.u) * trial
            - fd.dot(fd.grad(self.test), fd.dot(self.k, fd.grad(self.T + trial)))
            + self.test * self.alpha * fd.dot(self.u, fd.grad(self.p)) * trial
            + self.test * self.rho * self.H
        ) * self.dx
        if not self.disable_shear_heating:
            weak_form += (
                self.test
                * fd.inner(self.stress, self.apx.strain_rate(self.u))
                * self.dx
            )
        weak_form += (
            -self.test * self.rho * self.cp * fd.dot(self.u, self.n) * trial
            # + self.test * fd.dot(fd.dot(self.k, fd.grad(self.T + trial)), self.n)
        ) * self.ds

        return weak_form

    def mass(self, trial: fd.Function) -> fd.Form:
        return self.test * self.rho * self.cp * trial * self.dx

    def solve(self) -> None:
        self.timestepper.advance()


class AdvectionSolver:
    def __init__(
        self,
        solution: fd.Function,
        u: fd.Function,
        time_step: fd.Function | float,
        time_stepper,
        bcs=None,
        solver_parameters=None,
    ) -> None:
        self.solution = solution
        self.solution_old = fd.Function(solution)

        solver_parameters = solver_parameters or {
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }

        self.solver = GenericTransportSolver(
            "advection",
            solution,
            time_step,
            time_stepper,
            eq_attrs={"u": u},
            bcs=bcs,
            solver_parameters=solver_parameters,
        )

    def solve(self) -> None:
        self.solver.solve()
        self.solution_old.assign(self.solution)
