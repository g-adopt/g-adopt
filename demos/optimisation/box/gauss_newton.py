"""Gauss-Newton Hessian product for the TAO bounded Newton types, read from the tape.

The mathematics
---------------

Both objectives of this campaign (the box of ``problem.py`` and the cylinder of
``cylindrical/cylindrical_problem.py``) are a weighted sum of terms, each of which is
the output of one ``AssembleBlock`` of a form that is quadratic in one state:

    J(m) = sum_k w_k f_k(s_k(m)),        f_k(s) = int q_k(s - d_k) dmu_k,

with ``s_k`` a control-dependent state (the final temperature, the projected control
``T0``, the mixed Stokes state of one timestep), ``d_k`` data, ``q_k`` a quadratic
and ``w_k`` a constant weight. The Hessian is

    H v = sum_k w_k [ J_k^T f_k''(J_k v)  +  f_k'(s_k) . s_k''(m)[v, .] ],

where ``J_k = ds_k/dm`` is the tangent-linear map from the control to the state. The
second part carries the residual ``f_k'(s_k)``. It is the only part that can make the
Hessian indefinite. The Gauss-Newton product keeps the first part alone:

    H_GN v = sum_k w_k J_k^T f_k''(J_k v).

It is positive semi-definite for every ``v``, because ``<v, H_GN v> = sum_k w_k
f_k''[J_k v, J_k v]`` and every ``f_k''`` is positive semi-definite (a mass matrix, a
stiffness matrix or a boundary mass matrix with a positive coefficient). For a term
whose state is affine in the control (the regularisation terms, whose state is a
projection of the control with constant boundary values) ``s_k'' = 0``, so the
Gauss-Newton product is exact for that term.

The pyadjoint route
-------------------

pyadjoint has no Gauss-Newton product, and a reduced functional per residual would
cost one adjoint sweep per term (127 per product on the cylinder). This module builds
the product from three tape sweeps:

1. One tangent-linear sweep in the direction ``v``. Afterwards the block variable of
   every state carries ``ds_k = J_k v`` as its ``tlm_value``.
2. One "seed" per term on the adjoint value of the block variable of its state:
   ``seed_k = w_k d/ds [ d f_k / ds ] [ds_k]``, the second variation of the replaced
   form of the term block contracted with ``ds_k``. This is the second term of
   ``AssembleBlock.evaluate_hessian_component`` with ``adj_input = w_k``.
3. One adjoint sweep. The output of every term block has no adjoint value, so the term
   blocks and the arithmetic blocks that sum them are skipped (``Block.evaluate_adj``
   returns when no output has an adjoint value), and the first variation never enters.
   Every other block applies the transposed linearised model, ``J_k^T``, to the seeds.
   The adjoint value of the control is then ``H_GN v``, a Cofunction in the dual of the
   control space, the same space as pyadjoint's Hessian action under TAO.

After the product the release step clears the tangent-linear values, the adjoint
values and the adjoint state that every solve block keeps for a later Hessian
(``CachedSolverBlock.adj_state``, which ``tape.reset_variables`` does not clear). A
full Hessian product after a Gauss-Newton product then raises in pyadjoint instead of
reading the Gauss-Newton-seeded adjoint state. The next gradient writes a new state.

Pieces of pyadjoint that are used and are not public API: ``Block.evaluate_adj``,
``BlockVariable.add_adj_output``, ``Block.reset_adjoint_state``, the tape markings and
``AdjFloatExprBlock._operator``. The checks of ``gn_checks.py`` test every one of them
against pyadjoint's own gradient and Hessian.
"""

import gc
import time
from dataclasses import dataclass

from firedrake import Cofunction, Function, TestFunction, assemble, derivative
from firedrake.adjoint_utils.blocks.assembly import AssembleBlock
from petsc4py import PETSc
from pyadjoint import stop_annotating
from pyadjoint.adjfloat import AdjFloatExprBlock
from pyadjoint.block_variable import BlockVariable
from pyadjoint.drivers import marked_controls
from pyadjoint.optimization.tao_solver import PETScVecInterface
from ufl import ZeroBaseForm
from ufl.algorithms import expand_derivatives
from ufl.form import Form


HESSIAN_CHOICES = ("full", "gn", "gn-then-full")

# The relative defect of ``sum_k w_k f_k`` against ``J`` that the operator accepts at
# construction. No solver is on this path, only the arithmetic of the tape, so the
# defect is round-off of a sum of a few floats.
LINEAR_DEFECT_TOLERANCE = 1e-12


@dataclass
class QuadraticTerm:
    """One quadratic term of the taped objective.

    Attributes:
        label: a name for the record: the tape index of the term block, the name of
            the state Function and, for a state Function that several terms read
            (the Stokes state of each timestep on the cylinder), the position of the
            term among them in tape order.
        index: the position of the term block on the tape.
        block: the ``AssembleBlock`` whose output is ``f_k``.
        state: the block variable of the state ``s_k`` in ``block.get_dependencies()``.
            It is the version of the state Function that this term read, which is not
            ``function.block_variable`` when the Function was written again later on
            the tape (the mixed state ``z`` of every timestep, ``T_0`` on the cylinder).
        weight: ``w_k = dJ / df_k``, a constant number read from the arithmetic of the
            tape.
    """

    label: str
    index: int
    block: AssembleBlock
    state: BlockVariable
    weight: float

    @property
    def output(self):
        """The block variable of ``f_k``."""
        return self.block.get_outputs()[0]

    def replaced_form(self):
        """The form of the term with every coefficient replaced by its saved value.

        This is the form that ``AssembleBlock`` differentiates in its own adjoint
        and Hessian methods, so the seed below uses the same object. It is rebuilt at
        every call, because the saved values of a later replay can be new objects.
        """
        return self.block.prepare_evaluate_adj(None, None, None)


def _is_empty_form(form):
    """True for a form that ``expand_derivatives`` reduced to zero."""
    return isinstance(form, ZeroBaseForm) or (isinstance(form, Form) and form.empty())


def find_quadratic_terms(rf, expected_count=None):
    """Find the quadratic terms of the taped objective, their states and their weights.

    The walk starts at the block variable of ``J`` and goes back through the tape
    inside ``marked_control_dependents``:

    - a block variable that does not depend on the control ends the walk: it is data
      or a weight;
    - one made by an ``AdjFloatExprBlock`` (a sum, a product, a quotient of numbers)
      is arithmetic, and the walk continues with its dependencies;
    - one made by an ``AssembleBlock`` is a term;
    - any other producer raises, because the objective would then depend on the
      control through something that is not a sum of assembled terms.

    Each term must pass three tests:

    1. J is linear in the term outputs: for each arithmetic block and each pair
       ``(i, j)`` of its control-dependent dependencies, including ``i == j``, the
       second derivative of the operator, ``codegen(diff=(i, j))``, is exactly 0.0
       at the saved inputs. Sums, differences, negations, products with data and
       quotients by data pass. A product of two terms, a quotient by a term and a
       power fail.
    2. One state per term: exactly one control-dependent dependency of the term
       block, and it is a Function.
    3. The term is quadratic in that state: the third variation of the replaced form
       in the state expands to an empty form.

    The weights come from one reverse pass through the arithmetic blocks only: the
    adjoint value of ``J`` is set to ``rf.scale`` (1.0 here), each arithmetic block
    propagates it with its exact derivative, and ``w_k`` is the adjoint value that
    arrives at the output of term block ``k``. A float operand of the arithmetic is an
    ``AdjFloat`` with no producer block, so the walk stops there.

    Args:
        rf: the ``ReducedFunctional``. Its tape must hold a replay at the current
            control (the saved outputs are read).
        expected_count: the number of terms that the objective must have, or None.
            A mismatch raises, so that a term that the walk misses, or one that it
            finds twice, cannot pass unnoticed.

    Returns:
        list of ``QuadraticTerm`` in tape order.

    Raises:
        RuntimeError: when a test fails or the count does not match.
    """
    tape = rf.tape
    control = rf.controls[0]
    functional_bv = rf.functional.block_variable
    blocks = tape.get_blocks()
    # Every block variable to the block that made it, and that block's tape index.
    producer = {}
    for index, block in enumerate(blocks):
        for output in block.get_outputs():
            producer[output] = (index, block)

    terms = {}
    arithmetic = {}
    with tape.marked_control_dependents([control]):
        if not functional_bv.is_control_dependent:
            raise RuntimeError("the functional does not depend on the control")
        stack, seen = [functional_bv], set()
        while stack:
            bv = stack.pop()
            if bv in seen or not bv.is_control_dependent:
                continue
            seen.add(bv)
            if bv not in producer:
                # A control-dependent variable without a producer is the control
                # itself: J would read the control outside every assembled term.
                raise RuntimeError("the functional reads the control directly, not "
                                   "through an assembled term")
            index, block = producer[bv]
            if isinstance(block, AdjFloatExprBlock):
                arithmetic[index] = block
                stack.extend(block.get_dependencies())
            elif isinstance(block, AssembleBlock):
                terms[index] = block
            else:
                raise RuntimeError(f"the functional depends on the control through a "
                                   f"block that is neither arithmetic nor an assembly: "
                                   f"{block}")

        # Test 1: J is linear in the term outputs.
        for index, block in arithmetic.items():
            deps = block.get_dependencies()
            inputs = [dep.saved_output for dep in deps]
            dependent = [i for i, dep in enumerate(deps) if dep.is_control_dependent]
            for a in dependent:
                for b in dependent:
                    if b < a:
                        continue
                    value = block._operator.codegen(diff=(a, b))(*inputs)
                    if float(value) != 0.0:
                        raise RuntimeError(
                            f"J is not linear in the term outputs: the second "
                            f"derivative ({a}, {b}) of arithmetic block {index} "
                            f"({block}) is {float(value)}")

        # Tests 2 and 3: one state per term, quadratic in it.
        states = {}
        for index, block in terms.items():
            dependent = [dep for dep in block.get_dependencies()
                         if dep.is_control_dependent]
            if len(dependent) != 1 or not isinstance(dependent[0].output, Function):
                raise RuntimeError(
                    f"term block {index} ({block}) has {len(dependent)} "
                    f"control-dependent dependencies; it needs exactly one Function")
            state = dependent[0]
            form = block.prepare_evaluate_adj(None, None, None)
            s = state.saved_output
            # Three variations in the direction of a fresh Function. A quadratic form
            # in s has a constant second variation, so its third is zero after
            # expansion. A Function direction keeps every form a 0-form.
            e = Function(s.function_space())
            third = expand_derivatives(
                derivative(derivative(derivative(form, s, e), s, e), s, e))
            if not _is_empty_form(third):
                raise RuntimeError(f"term block {index} ({block}) is not quadratic in "
                                   f"its state")
            states[index] = state

        # The weights: one reverse pass through the arithmetic alone.
        tape.reset_variables()
        functional_bv.adj_value = rf.scale
        for index in sorted(arithmetic, reverse=True):
            arithmetic[index].evaluate_adj(markings=True)
        weights = {}
        for index, block in terms.items():
            value = block.get_outputs()[0].adj_value
            if value is None:
                raise RuntimeError(f"term block {index} received no weight")
            weights[index] = float(value)
        tape.reset_variables()
        if not arithmetic:
            # J is the output of the one term block itself.
            weights = {index: float(rf.scale) for index in terms}

    # Labels. A state Function that more than two terms read (the Stokes state of
    # every timestep) gets the position of the term among them in tape order.
    names = {index: states[index].output.name() for index in terms}
    counts = {}
    for name in names.values():
        counts[name] = counts.get(name, 0) + 1
    position = {}
    out = []
    for index in sorted(terms):
        name = names[index]
        label = f"tape{index}:{name}"
        if counts[name] > 2:
            k = position.get(name, 0)
            position[name] = k + 1
            label += f"#{k}"
        out.append(QuadraticTerm(label=label, index=index, block=terms[index],
                                 state=states[index], weight=weights[index]))
    if expected_count is not None and len(out) != expected_count:
        raise RuntimeError(f"found {len(out)} quadratic terms on the tape, expected "
                           f"{expected_count}: {[t.label for t in out]}")
    return out


def expected_term_count_box(weights):
    """The number of quadratic terms of the box objective of ``problem.py``.

    The temperature misfit is always on the tape. The damping and the smoothing term
    are on the tape only when their weight is positive.

    Args:
        weights: ``problem.weights`` with ``alpha_d`` and ``alpha_s``.

    Returns:
        int
    """
    return 1 + int(weights["alpha_d"] > 0) + int(weights["alpha_s"] > 0)


def expected_term_count_cylinder(weights, timesteps):
    """The number of quadratic terms of the cylindrical objective.

    ``generate_inverse_problem`` adds one temperature term, one velocity term per
    timestep (each an assembly on the mixed state of that step), and one damping and
    one smoothing term, each only when its weight is positive. 128 at 125 steps with
    every weight positive.

    Args:
        weights: ``problem.weights`` with ``alpha_T``, ``alpha_u``, ``alpha_d`` and
            ``alpha_s``.
        timesteps: the number of timesteps on the tape.

    Returns:
        int
    """
    return (int(weights["alpha_T"] > 0) + timesteps * int(weights["alpha_u"] > 0)
            + int(weights["alpha_d"] > 0) + int(weights["alpha_s"] > 0))


def add_gauss_newton_arguments(parser):
    """Add the options of the Hessian product and of the Krylov tolerance.

    The box driver (``inverse.py``) and the cylindrical driver (``compare.py``) add
    the same options through this function, so that the two cannot drift apart.

    Args:
        parser: an ``argparse.ArgumentParser``.
    """
    parser.add_argument("--hessian", default="full", choices=HESSIAN_CHOICES,
                        help="the Hessian product of the BNK Krylov solve: 'full' "
                             "(pyadjoint's Hessian, default), 'gn' (Gauss-Newton for "
                             "the whole run) or 'gn-then-full' (Gauss-Newton for the "
                             "first --gn-switch-it iterations, then the full Hessian, "
                             "with a return to Gauss-Newton for one iteration after a "
                             "full solve that stops on negative curvature). Anything "
                             "but 'full' needs a tao-*-riesz optimiser and the zero "
                             "shift (tao_bnk_imax=0)")
    parser.add_argument("--gn-switch-it", type=int, default=5,
                        help="gn-then-full: the solves of the first K outer iterations "
                             "use Gauss-Newton (default 5, hIPPYlib's GN_iter)")
    parser.add_argument("--gn-switch-gtol", type=float, default=0.0,
                        help="gn-then-full: switch also when TAO's gradient norm, which "
                             "for BNK is the Fischer-Burmeister residual in nodal "
                             "coordinates, falls to this fraction of its first value; "
                             "0 turns the test off (default)")
    parser.add_argument("--krylov-rtol", type=float, default=None,
                        help="fixed relative tolerance of the BNK Krylov solve "
                             "(default 1e-2); excludes --krylov-ew")
    parser.add_argument("--krylov-ew", action="store_true",
                        help="Eisenstat-Walker tolerance of the BNK Krylov solve "
                             "(tao_ksp_ew, version 2, rtol0 = rtolmax = 0.5); "
                             "excludes --krylov-rtol")
    parser.add_argument("--tr-norm", default="nodal", choices=("nodal", "metric"),
                        help="norm of the trust radius of STCG: 'nodal' (Euclidean "
                             "norm of the nodal values, default) or 'metric' (the norm "
                             "of the preconditioner, tao_bnk_ksp_cg_dtype "
                             "preconditioned)")


def gauss_newton_record(settings):
    """The settings of ``gauss_newton_settings`` under the keys of the run record.

    The record of a run is the metadata plus the recorder's summary, and the summary
    holds ``hessian``, the count of full products. So the choice of ``--hessian`` goes
    into the record as ``hessian_choice``; every other key keeps its name.

    Args:
        settings: the dict of ``gauss_newton_settings``.

    Returns:
        dict
    """
    return {("hessian_choice" if key == "hessian" else key): value
            for key, value in settings.items()}


def gauss_newton_settings(args):
    """The keyword arguments of ``run_tao`` for the options of the Hessian product.

    Args:
        args: the parsed command line.

    Returns:
        dict with ``hessian``, ``gn_switch_it``, ``gn_switch_gtol``,
        ``krylov_rtol``, ``krylov_ew`` and ``tr_norm``.

    Raises:
        SystemExit: when ``--krylov-ew`` and ``--krylov-rtol`` are both given.
    """
    if args.krylov_ew and args.krylov_rtol is not None:
        raise SystemExit("--krylov-ew and --krylov-rtol exclude each other")
    return {"hessian": args.hessian, "gn_switch_it": args.gn_switch_it,
            "gn_switch_gtol": args.gn_switch_gtol,
            "krylov_rtol": 1e-2 if args.krylov_rtol is None else args.krylov_rtol,
            "krylov_ew": bool(args.krylov_ew), "tr_norm": args.tr_norm}


class GaussNewtonOperator:
    """The Gauss-Newton product ``v -> H_GN v`` on the tape of a reduced functional.

    Args:
        rf: the ``ReducedFunctional``. One control, ``scale == 1``, and no checkpoint
            schedule, because the tangent-linear and adjoint sweeps here need every
            state in memory (pyadjoint has no checkpoint-manager path for the
            tangent-linear sweep).
        terms: the list of ``find_quadratic_terms``, or None to find them here.
        expected_count: passed to ``find_quadratic_terms`` when ``terms`` is None.
        recorder: the ``Recorder`` of ``optimisers.py``, or None. When it is set,
            every product calls ``recorder.end_gn(elapsed)``, which counts it and
            stores its time.

    Attributes:
        terms: the quadratic terms.
        adjoint_current: whether the adjoint state of the solve blocks belongs to a
            gradient at the current tape point. A product sets it to False in its
            release step, because the release clears that state. The full-Hessian
            mode of ``GaussNewtonHessianMat`` sets it to True after its update, which
            runs ``rf.derivative()``.
        linear_defects: the defect of ``linear_combination_defect`` at construction
            and after each call of ``check_linear_combination``.
    """

    def __init__(self, rf, terms=None, expected_count=None, recorder=None):
        if len(rf.controls) != 1:
            raise RuntimeError("the Gauss-Newton product supports one control")
        if rf.scale != 1.0:
            raise RuntimeError(f"the Gauss-Newton product needs rf.scale == 1, got "
                               f"{rf.scale}")
        if rf.tape._checkpoint_manager is not None:
            raise RuntimeError("the Gauss-Newton product needs every state in memory: "
                               "the tape has a checkpoint schedule")
        self.rf = rf
        self.tape = rf.tape
        self.control = rf.controls[0]
        self.space = self.control.control.function_space()
        self.terms = (find_quadratic_terms(rf, expected_count) if terms is None
                      else terms)
        self.recorder = recorder
        self.adjoint_current = False
        self.linear_defects = []
        self.check_linear_combination(raise_above=LINEAR_DEFECT_TOLERANCE)

    # -- consistency ---------------------------------------------------------------

    def linear_combination_defect(self):
        """``|sum_k w_k f_k - J| / |J|`` at the saved values of the last replay.

        The objective is the weighted sum of its terms, so the defect is round-off
        unless a term is missing, a weight is wrong, or J holds a constant that is
        not a term. No solve is on this path.

        Returns:
            float
        """
        total = sum(term.weight * float(term.output.saved_output) for term in self.terms)
        J = float(self.rf.functional.block_variable.saved_output)
        return abs(total - J) / abs(J) if J != 0.0 else abs(total)

    def check_linear_combination(self, raise_above=None):
        """Compute the defect, keep it, and raise above a threshold if one is given.

        Args:
            raise_above: a threshold, or None to record the value only. A TAO
                callback must not raise (petsc4py turns the exception into PETSc error
                101, which ends the run), so the Hessian update records only.

        Returns:
            float, the defect.
        """
        defect = self.linear_combination_defect()
        self.linear_defects.append(defect)
        if raise_above is not None and defect > raise_above:
            raise RuntimeError(f"sum_k w_k f_k differs from J by {defect:.3e} "
                               f"(relative); a term or a weight is wrong")
        return defect

    # -- the sweeps ----------------------------------------------------------------

    def tlm_states(self, v):
        """Tangent-linear sweep in the direction ``v``: ``ds_k = J_k v`` for every term.

        The same calls as pyadjoint's ``compute_tlm``: the tangent-linear values of
        the tape are cleared, the control receives ``v``, and the sweep runs inside
        the control marking, so that only control-dependent blocks are evaluated.

        Args:
            v: a Function in the control space.

        Returns:
            list of Functions, the tangent-linear value of each state, in the order
            of ``terms``. The objects stay valid after the next reset, which only
            drops the references on the tape.
        """
        self.tape.reset_tlm_values()
        self.control.tlm_value = v
        with stop_annotating(), self.tape.marked_control_dependents([self.control]):
            self.tape.evaluate_tlm(markings=True)
        states = [term.state.tlm_value for term in self.terms]
        for term, ds in zip(self.terms, states):
            if ds is None:
                raise RuntimeError(f"the state of term {term.label} has no "
                                   f"tangent-linear value")
        return states

    def _adjoint_sweep(self):
        """Reverse sweep from the seeds on the states; returns a copy of the result.

        The calls of pyadjoint's ``compute_derivative`` without its first step: the
        adjoint value of J stays None, so the first variation of every term never
        enters, and the sweep starts from the seeds alone.

        Returns:
            Cofunction in the dual of the control space.
        """
        with stop_annotating(), \
                self.tape.marked_control_dependents([self.control]), \
                self.tape.marked_functional_dependencies(self.rf.functional), \
                marked_controls([self.control]):
            self.tape.evaluate_adj(markings=True)
        derivative_value = self.control.get_derivative(apply_riesz=False)
        out = Cofunction(self.space.dual())
        out.assign(derivative_value)
        return out

    def _release(self):
        """Clear every value that the sweeps left on the tape.

        - the tangent-linear and the adjoint values of every block variable;
        - the adjoint state of every block. A solve block keeps its adjoint solution
          in ``adj_state`` for a later second-order sweep, and the sweep here wrote
          the Gauss-Newton-seeded solution there. ``Block.reset_adjoint_state`` sets
          it to None, so a full Hessian product without a new gradient raises in
          pyadjoint instead of using it. The next gradient writes a new state.
        """
        self.tape.reset_tlm_values()
        self.tape.reset_variables()
        for block in self.tape.get_blocks():
            block.reset_adjoint_state()
        self.adjoint_current = False

    def seeds(self, directions, terms=None):
        """The Gauss-Newton seed of each term for given state directions.

        ``seed_k = assemble(w_k * d/ds (d f_k/ds [phi]) [ds_k])``, with ``phi`` the
        test function of the state space: the second variation of the replaced form
        in its state, contracted with the tangent-linear value ``ds_k``. A Cofunction
        in the dual of the state space.

        Args:
            directions: list of state directions, one per term of ``terms``.
            terms: the terms, default all.

        Returns:
            list of Cofunction or None (None for a term whose second variation
            vanishes, which none of the terms of this campaign does).
        """
        terms = self.terms if terms is None else terms
        out = []
        for term, ds in zip(terms, directions):
            form = term.replaced_form()
            s = term.state.saved_output
            second = expand_derivatives(
                derivative(derivative(form, s, TestFunction(s.function_space())), s, ds))
            out.append(None if _is_empty_form(second)
                       else assemble(term.weight * second))
        return out

    def action(self, v, terms=None):
        """``H_GN v``, a Cofunction in the dual of the control space.

        Args:
            v: a Function in the control space.
            terms: a subset of ``self.terms`` to restrict the sum to, or None for all.
                A check uses the regularisation terms alone, whose product must equal
                the metric ``A``.

        Returns:
            Cofunction.
        """
        start = time.perf_counter()
        selected = self.terms if terms is None else terms
        try:
            all_states = self.tlm_states(v)
            by_term = {id(t): ds for t, ds in zip(self.terms, all_states)}
            directions = [by_term[id(t)] for t in selected]
            seeds = self.seeds(directions, selected)
            # The adjoint values must be empty before the seeds go in.
            self.tape.reset_variables()
            for term, seed in zip(selected, seeds):
                if seed is not None:
                    term.state.add_adj_output(seed)
            out = self._adjoint_sweep()
        finally:
            self._release()
        elapsed = time.perf_counter() - start
        if self.recorder is not None:
            self.recorder.end_gn(elapsed)
        # A product leaves reference cycles that CPython frees only when its cyclic
        # collector runs. The full product collects in ``hessian_cb_post``, which this
        # route never calls, so the collection is here, after the timing stops.
        gc.collect()
        return out

    def first_variation_gradient(self):
        """The gradient from first-variation seeds and the same adjoint sweep.

        ``seed_k = assemble(w_k * d f_k/ds [phi])`` on the state of each term, then
        the adjoint sweep of ``action``. The result must equal ``rf.derivative()``:
        it tests the seed placement, the weights and the sweep, apart from the
        tangent-linear half. Not used by a run.

        Returns:
            Cofunction in the dual of the control space.
        """
        try:
            self.tape.reset_variables()
            for term in self.terms:
                form = term.replaced_form()
                s = term.state.saved_output
                first = expand_derivatives(
                    derivative(form, s, TestFunction(s.function_space())))
                term.state.add_adj_output(assemble(term.weight * first))
            out = self._adjoint_sweep()
        finally:
            self._release()
        return out

    def term_curvatures(self, directions_a, directions_b):
        """``w_k f_k''[a_k, b_k]`` for each term, from two sets of state directions.

        With ``a_k = ds_k(v)`` and ``b_k = ds_k(w)`` the sum over ``k`` is
        ``<w, H_GN v>`` computed from two tangent-linear sweeps alone, without the
        adjoint half. Not used by a run.

        Returns:
            list of float.
        """
        out = []
        for term, a, b in zip(self.terms, directions_a, directions_b):
            form = term.replaced_form()
            s = term.state.saved_output
            out.append(float(assemble(term.weight * expand_derivatives(
                derivative(derivative(form, s, a), s, b)))))
        return out

    def describe(self, aggregate_above=3):
        """The term list for the run record.

        Args:
            aggregate_above: a state Function that at least this many terms read is
                written as one entry with the count and the range of the weights (the
                velocity terms of the cylinder, one per timestep).

        Returns:
            list of dict.
        """
        groups = {}
        for term in self.terms:
            groups.setdefault(term.state.output.name(), []).append(term)
        out = []
        for name, group in groups.items():
            space = group[0].state.output.function_space()
            element = space.ufl_element()
            common = {"state": name, "state_dim": int(space.dim()),
                      "state_element": str(element)}
            if len(group) >= aggregate_above:
                weights = [t.weight for t in group]
                out.append(dict(common, count=len(group), first=group[0].label,
                                last=group[-1].label, weight_min=min(weights),
                                weight_max=max(weights)))
            else:
                for term in group:
                    out.append(dict(common, label=term.label, weight=term.weight))
        return out


class GaussNewtonHessianMat:
    """Python ``Mat`` context that replaces pyadjoint's Hessian matrix in TAO.

    It is installed as the Hessian of the TAO object in the way that
    ``riesz_hessian.ShiftedHessianMat`` is, and it applies one of two products:

    - mode ``"gn"``: ``H_GN x`` from ``GaussNewtonOperator``;
    - mode ``"full"``: pyadjoint's Hessian product, through pyadjoint's own matrix.

    The choice ``"gn"`` keeps mode ``"gn"`` for the whole run. The choice
    ``"gn-then-full"`` starts in ``"gn"`` and switches to ``"full"`` at the update of
    TAO iteration ``switch_it`` (so the solves of iterations 0 to ``switch_it - 1``
    use Gauss-Newton, as hIPPYlib's ``GN_iter``), or earlier when ``switch_gtol > 0``
    and TAO's gradient norm falls to ``switch_gtol`` of its first value. For the
    bounded Newton types that norm is the Fischer-Burmeister residual in nodal
    coordinates. In mode ``"full"`` a fall-back applies: when the last Krylov solve
    ended with ``CONVERGED_NEG_CURVE``, the next iteration uses Gauss-Newton, and the
    switch test applies again at the update after it. Every change goes into
    ``switches``.

    The matrix defines ``mult``, ``multHermitian`` and ``shift`` and nothing else (no
    ``createSubMatrix``, no ``getDiagonal``), so BNK builds its virtual sub-matrix on
    the inactive set and takes the same branches as with pyadjoint's matrix.

    Args:
        hessian_mat: pyadjoint's Hessian ``Mat`` from ``TAOSolver``.
        operator: the ``GaussNewtonOperator``.
        choice: ``"gn"`` or ``"gn-then-full"`` (``"full"`` keeps mode full, for a
            check).
        switch_it: see above.
        switch_gtol: see above; zero turns the gradient test off.

    Attributes:
        mode: ``"gn"`` or ``"full"``, the product of the next Krylov solve.
        switches: list of dict, one per change of mode.
        _shift: the sum of the ``MatShift`` values since the last update. With the
            zero shift that every Gauss-Newton run must use it stays zero.
    """

    def __init__(self, hessian_mat, operator, choice="gn", switch_it=5, switch_gtol=0.0):
        if choice not in HESSIAN_CHOICES:
            raise ValueError(f"choice must be one of {HESSIAN_CHOICES}, got {choice!r}")
        self.hessian_mat = hessian_mat
        self.hessian_context = hessian_mat.getPythonContext()
        self.operator = operator
        self.choice = choice
        self.switch_it = int(switch_it)
        self.switch_gtol = float(switch_gtol)
        self.mode = "full" if choice == "full" else "gn"
        self.switches = []
        self.gnorm0 = None
        self._shift = 0.0
        control = operator.control.control
        self.interface = PETScVecInterface((control,))
        self._m = Function(control.function_space())
        self._v = Function(control.function_space())
        # The number of Hessian updates in each mode, for the record.
        self.updates = {"gn": 0, "full": 0}

    @property
    def adjoint_current(self):
        """Whether a full product may run: the flag of the operator."""
        return self.operator.adjoint_current

    @adjoint_current.setter
    def adjoint_current(self, value):
        self.operator.adjoint_current = bool(value)

    @property
    def sizes(self):
        """The sizes of pyadjoint's Hessian matrix."""
        return self.hessian_mat.getSizes()

    def create_mat(self):
        """Build the PETSc ``Mat`` that carries this context, set up as pyadjoint does."""
        mat = PETSc.Mat().createPython(self.sizes, self, comm=self.hessian_mat.getComm())
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        mat.setUp()
        mat.assemble()
        return mat

    def _switch(self, new_mode, iteration, reason, gnorm, recorder_work=None):
        """Change the mode and append the change to ``switches``."""
        self.switches.append({"iteration": int(iteration), "from": self.mode,
                              "to": new_mode, "reason": reason,
                              "gradient_norm": float(gnorm), "work": recorder_work})
        self.mode = new_mode

    def choose_mode(self, tao):
        """Apply the switch rule of ``gn-then-full`` at a Hessian update.

        Args:
            tao: the TAO object, or any object with ``getIterationNumber``,
                ``getSolutionStatus`` and ``getKSP`` (a check passes a stub).
        """
        iteration = tao.getIterationNumber()
        gnorm = float(tao.getSolutionStatus()[2])
        if self.gnorm0 is None:
            self.gnorm0 = gnorm
        if self.choice != "gn-then-full":
            return
        work = None
        recorder = self.operator.recorder
        if recorder is not None:
            work = float(recorder.work())
        if self.mode == "gn":
            reason = None
            if iteration >= self.switch_it:
                reason = f"iteration {iteration} >= switch_it {self.switch_it}"
            elif (self.switch_gtol > 0.0 and self.gnorm0
                  and gnorm <= self.switch_gtol * self.gnorm0):
                reason = (f"gradient norm {gnorm:.3e} <= {self.switch_gtol:g} x "
                          f"{self.gnorm0:.3e}")
            if reason is not None:
                self._switch("full", iteration, reason, gnorm, work)
        else:
            last = tao.getKSP().getConvergedReason()
            if last == PETSc.KSP.ConvergedReason.CONVERGED_NEG_CURVE:
                self._switch("gn", iteration,
                             "fall-back: the last full solve ended with "
                             "CONVERGED_NEG_CURVE", gnorm, work)

    def update(self, tao, x, H, P):
        """The Hessian callback of TAO.

        Mode ``"full"``: pyadjoint's own update, which replays the tape at ``x`` and
        runs ``rf.derivative()``, on pyadjoint's matrix. Mode ``"gn"``: a replay at
        ``x`` alone, because the Gauss-Newton product reads no adjoint state. The
        replay fires ``eval_cb_post``, so the recorder counts one forward run, and it
        adds no gradient row (only the derivative callback adds rows). Then the
        linear-combination defect of the operator is recorded.
        """
        self.choose_mode(tao)
        if self.mode == "full":
            type(self.hessian_context).update(tao, x, self.hessian_mat, P)
            self.adjoint_current = True
        else:
            self.interface.from_petsc(x, (self._m,))
            self.operator.rf(self._m)
            self.operator.check_linear_combination()
        self.updates[self.mode] += 1
        self._shift = 0.0

    def shift(self, A, alpha):
        """``MatShift``: add ``alpha`` to the shift, as pyadjoint's matrix does."""
        self._shift += alpha

    def mult(self, A, x, y):
        """``y = H x + shift x`` with ``H`` the product of the current mode."""
        if self.mode == "full":
            if not self.adjoint_current:
                raise RuntimeError("full Hessian product without a current adjoint state")
            self.hessian_context.mult(self.hessian_mat, x, y)
        else:
            self.interface.from_petsc(x, (self._v,))
            out = self.operator.action(self._v)
            self.interface.to_petsc(y, (out,))
        if self._shift != 0.0:
            y.axpy(self._shift, x)

    def multHermitian(self, A, x, y):
        """Both products are symmetric, so the Hermitian action is the same action."""
        self.mult(A, x, y)

    def record(self):
        """The top-level entries of the run record."""
        defects = self.operator.linear_defects
        return {"hessian_switches": list(self.switches),
                "gn_terms": self.operator.describe(),
                "gn_linear_defect_max": float(max(defects)) if defects else None,
                "hessian_updates": dict(self.updates)}


def build_gauss_newton_hessian(tao, problem, recorder=None, choice="gn", switch_it=5,
                               switch_gtol=0.0, expected_count=None):
    """Install ``GaussNewtonHessianMat`` as the Hessian of a TAO object of ``TAOSolver``.

    Args:
        tao: the TAO object of ``TAOSolver``, after its ``setUp``.
        problem: the inverse problem (box or cylinder), for its reduced functional.
        recorder: the ``Recorder`` of the run, or None.
        choice: ``"gn"`` or ``"gn-then-full"``.
        switch_it, switch_gtol: the switch settings of ``gn-then-full``.
        expected_count: the number of quadratic terms the objective must have.

    Returns:
        ``(mat, context)``. The caller must keep both alive for the whole solve,
        because a petsc4py TAO object cannot carry Python attributes.
    """
    hessian_mat, pmat = tao.getHessian()[:2]
    operator = GaussNewtonOperator(problem.reduced_functional,
                                   expected_count=expected_count, recorder=recorder)
    context = GaussNewtonHessianMat(hessian_mat, operator, choice=choice,
                                    switch_it=switch_it, switch_gtol=switch_gtol)
    mat = context.create_mat()
    tao.setHessian(context.update, H=mat, P=pmat)
    return mat, context
