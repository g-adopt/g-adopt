r"""The two history families of the coupled GIA solver, with one time integrator.

The viscoelastic history (the internal variables, or the creep strain and
backstress of `gadopt.BackstressApproximation`) is stored in one of two ways:

- `DGHistory`: a discontinuous tensor field that is an unknown of the mixed
  system, with its weak history equation next to the momentum equation. The
  block-0 preconditioners eliminate it cell by cell.
- `QuadratureHistory`: values at the quadrature points of the mechanics mesh,
  eliminated by a local Newton solve at every point
  (`gadopt.constitutive_integrators`). The mixed system then holds the
  displacement and the potential only.

Both read their time coefficients from one `BDF2History`, whose accepted
level is the one public history Function of the solver: there is no second
copy to keep in step. Backward Euler is order 1 of the same integrator, which
then stores no previous level.

The formulations extend `gadopt.stokes_integrators.HistoryFormulation` and
are driven by `InternalVariableMechanics`. They do not write
`approximation.mu`: `shear_coefficient` returns the penalty coefficient and
the solver passes it to the momentum terms as the equation attribute `mu`.
"""

import firedrake as fd

from .equations import Equation
from .internal_variable_equation import history_slices, internal_variable_space
from .stokes_integrators import HistoryFormulation, MixedHistoryFormulation
from .viscoelastic_timestepping import BDF2History

__all__ = ["DGHistory", "QuadratureHistory", "history_formulation"]


def _make_integrator(history, dt, order):
    """One `BDF2History` whose accepted level is `history` itself."""
    return BDF2History([history], dt, order=order, own_current=True,
                       keep_previous=(order == 2))


class DGHistory(MixedHistoryFormulation):
    r"""The history as a discontinuous unknown of the mixed space (field 1).

    The weak history equation of the approximation (`history_terms`) at the
    time level of the integrator: the old value is the BDF2 combination
    `beta m_n + gamma m_{n-1}` (or `m_n` on a backward-Euler step), and the
    new value carries `alpha`. The equation is negated through the scaling
    factor so that the `(u, M)` and `(M, u)` blocks are transposes. At fixed
    history the displacement block is elastic, so the penalty coefficient is
    `mu0`.

    Args:
      solver: the coupled solver, which owns `dt` and the mesh.
      degree: the polynomial degree of the DG history.
      order: 1 for backward Euler, 2 for variable-step BDF2.
    """

    name = "DG"

    def __init__(self, solver, degree, order):
        super().__init__()
        approximation = solver.approximation
        #: The DG degree, or the quadrature degree.
        self.degree = degree
        self.space = internal_variable_space(
            solver.mesh, approximation.n_history_tensors, degree, family="DG")
        self.history = fd.Function(self.space, name="internal_variables")
        self.time_integrator = _make_integrator(self.history, solver.dt, order)

    def bind(self, solver) -> None:
        n = len(history_slices(solver.solution_split[self.history_index]))
        if n != solver.approximation.n_history_tensors:
            raise ValueError(
                f"The history field holds {n} tensor(s); the approximation "
                f"needs {solver.approximation.n_history_tensors}.")
        HistoryFormulation.bind(self, solver)

    def shear_coefficient(self, approximation, dt):
        return approximation.mu0

    def add_history_equations(self, mechanics, solver) -> None:
        index = self.history_index
        integrator = self.time_integrator
        solver.equations.append(
            Equation(
                solver.tests[index],
                solver.solution_space[index],
                solver.approximation.history_terms(),
                eq_attrs={
                    "maxwell_times": self.maxwell_times(mechanics, solver),
                    "displacement": mechanics.displacement,
                    "dt": solver.dt,
                    "trial_old": integrator.combinations()[0],
                    "time_alpha": integrator.alpha,
                },
                approximation=solver.approximation,
                bcs=solver.weak_bcs,
                quad_degree=solver.quad_degree,
                scaling_factor=-solver.scaling_factor,
                intersect_measures=mechanics.intersect_measures,
            )
        )

    def advance(self, solver) -> None:
        # The step that solved knows where the new history is.
        self.time_integrator.accept([solver.solved_history()])


class QuadratureHistory(HistoryFormulation):
    r"""The history at quadrature points, eliminated by a local Newton solve.

    The mixed space holds the displacement (field 0) and the potential. The
    stress sees the updated history as an expression of the displacement and
    of one scalar fluidity per point (`material.updates`), which the
    approximation's `quadrature_update` defines. The step is solved and
    recorded by `gadopt.constitutive_integrators.DirectConstitutiveBackend`.

    The displacement tangent with the history substituted has the effective
    shear coefficient `sum_i mu_i alpha / (alpha + dt / tau_i)`, which is the
    penalty coefficient here.

    Args:
      solver: the coupled solver.
      degree: the quadrature degree, which is the solver's `quad_degree`.
      order: 1 for backward Euler, 2 for variable-step BDF2.
    """

    name = "Quadrature"

    def __init__(self, solver, degree, order):
        super().__init__()
        approximation = solver.approximation
        #: The DG degree, or the quadrature degree.
        self.degree = degree
        self.space = internal_variable_space(
            solver.mesh, approximation.n_history_tensors, degree,
            family="Quadrature")
        self.history = fd.Function(self.space, name="internal_variables")
        self.time_integrator = _make_integrator(self.history, solver.dt, order)
        #: The local constitutive update, built in `history_values`.
        self.material = None

    def shear_coefficient(self, approximation, dt):
        alpha = self.time_integrator.alpha
        return sum(mu * alpha / (alpha + dt / tau)
                   for mu, tau in zip(approximation.shear_modulus,
                                      approximation.maxwell_times))

    def displacement(self, solver):
        return solver.solution_split[0]

    def displacement_test(self, solver):
        return solver.tests[0]

    def displacement_space(self, solver):
        return solver.solution_space[0]

    def history_values(self, mechanics, solver):
        self.material = solver.approximation.quadrature_update(solver)
        return [self.material.on_mixed_state(g) for g in self.material.updates]

    def add_history_equations(self, mechanics, solver) -> None:
        # The history is eliminated at quadrature points; no equation.
        return None

    def advance(self, solver) -> None:
        # The step that solved knows where the new history is.
        self.time_integrator.accept([solver.solved_history()])


def history_formulation(family, solver, degree, order):
    """The formulation of `family`, `"DG"` or `"Quadrature"`."""
    if family == "DG":
        return DGHistory(solver, degree, order)
    if family == "Quadrature":
        return QuadratureHistory(solver, degree, order)
    raise ValueError(
        f"internal_variable_family must be 'DG' or 'Quadrature', got {family!r}.")
