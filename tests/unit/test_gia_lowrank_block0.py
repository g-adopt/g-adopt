r"""The low-rank DtN update in the block-0 preconditioner: fast checks without a solve.

What is under test is the construction around `gadopt.LowRankPotentialPC`:
the capacitance form of the Woodbury identity with zero weights, the outer
stopping options of `selfgrav_dtn_iterative_solver_parameters`, and the
refusals of a preset that does not match the solver's DtN representation,
of a preconditioning matrix the class cannot read, and of an installation
on a Jacobian that is not matrix-free.

`build` here is also the annulus builder of
`test_gia_nullspace_projection_adjoint.py`, and `block0_context` and
`potential_pc` are read by `test_sea_level.py`.
"""
import numpy as np
import pytest

import gadopt  # noqa: F401  - before firedrake, as the drivers do
import firedrake as fd
from firedrake.adjoint import (
    get_working_tape,
    pause_annotation,
)
from pyadjoint.tape import annotate_tape

from gadopt import CylindricalDtN
from gadopt.approximations import CompressibleInternalVariableApproximation
from gadopt.gia_couplings import FluidCore, Rotation, SurfaceLoad
from gadopt.gia_presets import (
    OMEGA_SQ_EARTH,
    selfgrav_dtn_iterative_solver_parameters,
)
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.preconditioners import LowRankPotentialPC

# The annulus of the GIA suite (dr 0.2, 32 cells around) and its constants.
import test_gia_gravity  # noqa: E402
from gia_models.annulus import (
    B_MU,
    CURVE_INNER,
    CURVE_OUTER,
    CURVE_RC,
    CURVE_RE,
    LAMBDA,
    SIGMA_HAT,
)
# The module fixture of `test_gia_gravity`, bound by assignment so that pytest finds
# it here and flake8 does not read the test arguments of the same name as
# redefinitions of an import.
meshes = test_gia_gravity.meshes


# The tolerances of the preset that `build` hands to the solver.
OUTER_RTOL = 1e-10
BLOCK0_RTOL = 1e-4

#: The reference gravity of the annulus, units of g_bar.
G0 = 1.0


@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


def preset(representation, **kwargs):
    """The two-block iterative preset of the production configuration."""
    options = dict(condensed=False, block0="condensed",
                   block0_rtol=BLOCK0_RTOL, outer_rtol=OUTER_RTOL,
                   block0_max_it=200, snes_type="ksponly",
                   multiplier_pc="gadopt.DtNMultiplierDenseSchurPC",
                   dtn_representation=representation)
    options.update(kwargs)
    return selfgrav_dtn_iterative_solver_parameters(**options)


def build(meshes, representation, *, truncation=3, fluid_core=True,
          rotation=True, solver_parameters=None, dt=1.0,
          approximation_kwargs=None, nullspace=None,
          initial_displacement=None):
    """One coupled solver on the annulus, in either representation.

    Mirrors `NOTES/measurements/lowrank-iterative/lowrank_block0.py`, which is
    where the measurements in the finding come from, so that a count asserted
    here is the count recorded there.

    The surface load is `SIGMA_HAT (cos 2 theta + m)` on Re, with the mean
    `m = 0.25` on a fluid core and 0 on a rigid one. With a fluid core the
    mean is the part of the load that the core pressure balances, so the
    core-pressure row carries a nonzero value. The load is one `SurfaceLoad`:
    the traction `B_mu g sigma` on the mechanics, the mass sheet in the
    potential row, the rotation row and the 2-D monopole datum.

    Args:
      meshes: the `(parent, sub)` pair.
      representation: `"multiplier"` or `"lowrank"`.
      truncation: the DtN truncation of both boundaries.
      fluid_core: a `FluidCore` of density 2 at the CMB, or a rigid core
        (`un = 0`).
      rotation: the rotational closure with the polar moment of the mantle.
      solver_parameters: a complete dictionary or a preset name; `None` is
        `preset(representation)`.
      dt: the time step, held in a `Constant`.
      approximation_kwargs: approximation settings on top of the defaults;
        a control (`shear_modulus`, `viscosity`) goes in here.
      nullspace: the `nullspace` argument of the solver.
      initial_displacement: UFL interpolated into the displacement before
        construction, so it is the initial state of the first solve.

    Returns:
      `(solver, u, psi)`.
    """
    parent, sub = meshes
    Xm = fd.SpatialCoordinate(sub)
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    if initial_displacement is not None:
        u.interpolate(initial_displacement)
    gravity_bcs = {
        CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    mean = fd.Constant(0.25) if fluid_core else fd.Constant(0.0)
    couplings = [SurfaceLoad(
        boundary=CURVE_RE, g_surface=G0,
        sigma=SIGMA_HAT * (fd.cos(2 * fd.atan2(Xm[1], Xm[0])) + mean))]
    mechanics_bcs = {}
    if fluid_core:
        couplings.append(FluidCore(boundary=CURVE_RC, rho_core=2.0))
    else:
        mechanics_bcs[CURVE_RC] = {"un": 0.0}
    if rotation:
        dx_m = fd.Measure("dx", domain=sub,
                          intersect_measures=(fd.Measure("dx", domain=parent),))
        couplings.append(Rotation(C=fd.assemble(fd.dot(Xm, Xm) * dx_m),
                                  Omega_sq=OMEGA_SQ_EARTH))

    settings = dict(
        bulk_modulus=1.0, density=fd.Constant(1.0),
        shear_modulus=fd.Constant(1.0), viscosity=fd.Constant(1.0),
        g=fd.Constant(G0), B_mu=fd.Constant(B_MU),
        self_gravity_number=fd.Constant(LAMBDA))
    settings.update(approximation_kwargs or {})
    approx = CompressibleInternalVariableApproximation(**settings)
    if solver_parameters is None:
        solver_parameters = preset(representation)
    solver = SelfGravitatingGIASolver(
        u, psi, approx, dt=fd.Constant(dt), gravity_bcs=gravity_bcs,
        bcs=mechanics_bcs, couplings=couplings,
        dtn_representation=representation, nullspace=nullspace,
        solver_parameters=solver_parameters)
    return solver, u, psi


def block0_context(solver):
    """The `CondensedBlockPC` instance block 0 runs, after a solve."""
    outer = solver.solver.snes.ksp.pc.getPythonContext()
    block0, _ = outer.pc.getFieldSplitSchurGetSubKSP()
    return block0.getPC().getPythonContext()


def potential_pc(condensed_block):
    """The preconditioner of the potential split of the block-0 nest."""
    split = condensed_block.condensed_ksp.getPC().getFieldSplitSubKSP()[1]
    return split.getPC().getPythonContext()


# ---------------------------------------------------------------------------
# 1. The identity, on its own terms
# ---------------------------------------------------------------------------
def test_the_capacitance_form_survives_zero_weights():
    """`Cap = I + theta W (U^T Z)` reproduces the dense inverse exactly.

    The form of the Woodbury identity in every textbook carries `W^-1/theta`
    in the capacitance, and it does not exist for this problem: a weight is
    `(lam_k - alpha/R) / (scale_k * A_h)`, and `gadopt.dtn_form` builds
    `lam = (l+1)/R` on an exterior boundary and `l/R` on an interior one with
    `alpha = 1`, so the weight is EXACTLY zero for the exterior `l = 0` mode
    and the interior `l = 1` modes. The class carries `theta W` instead.

    This is the algebra alone, on a dense random system with two exact zeros
    and two negatives in `W`, so that a failure here is a failure of the
    identity and not of any solver around it.
    """
    rng = np.random.default_rng(0)
    n, k = 40, 6
    M = rng.normal(size=(n, n))
    A = M @ M.T + n * np.eye(n)
    U = rng.normal(size=(n, k))
    w = np.array([0.0, -0.7, 1.3, 0.0, 2.1, -0.2])
    theta = 3.7

    exact = np.linalg.inv(A + theta * U @ np.diag(w) @ U.T)
    A_inv = np.linalg.inv(A)
    Z = A_inv @ U
    cap = np.eye(k) + (theta * w)[:, np.newaxis] * (U.T @ Z)
    applied = A_inv - Z @ np.linalg.solve(
        cap, (theta * w)[:, np.newaxis] * (U.T @ A_inv))

    assert np.abs(exact - applied).max() / np.abs(exact).max() < 1e-13


def test_zero_weights_are_actually_present(meshes):
    """The zero weights above are this configuration's, not a hypothetical.

    If this ever returns zero, the test above is guarding an unreachable case
    and the simpler `W^-1` form would do. It returns four.
    """
    solver, _, _ = build(meshes, "lowrank")
    weights = np.concatenate(
        [rows.weights for rows in solver.dtn_operator.mode_rows])
    assert int((weights == 0.0).sum()) == 4


# ---------------------------------------------------------------------------
# 2. The stopping rule of the outer FGMRES
# ---------------------------------------------------------------------------
def test_the_outer_stopping_options_reach_the_dictionary():
    """`outer_atol`, `outer_max_it` and `outer_accept_capped` set the outer KSP.

    The defaults leave the dictionary as callers had it before: no
    `ksp_atol` (PETSc's own, effectively off), a cap of 200 and no
    `ksp_converged_maxits`.

    CATCHES: an option that never reaches the outer KSP, or a default that
    changes the dictionary of every existing caller.
    """
    default = preset("lowrank")
    assert "ksp_atol" not in default
    assert "ksp_converged_maxits" not in default
    assert default["ksp_max_it"] == 200
    chosen = preset("lowrank", outer_atol=1e-11, outer_max_it=30,
                    outer_accept_capped=True)
    assert chosen["ksp_atol"] == 1e-11
    assert chosen["ksp_max_it"] == 30
    assert "ksp_converged_maxits" in chosen


@pytest.mark.parametrize("kwargs", [{"outer_atol": -1.0},
                                    {"outer_max_it": 0}])
def test_the_preset_refuses_a_wrong_outer_stopping_option(kwargs):
    with pytest.raises(ValueError, match="outer_atol must be >= 0"):
        preset("lowrank", **kwargs)


# ---------------------------------------------------------------------------
# 3. What the configuration refuses
# ---------------------------------------------------------------------------
def test_the_preset_supports_the_eliminated_layout_on_the_low_rank_path():
    """The retained u/psi nest carries the low-rank potential correction."""
    options = selfgrav_dtn_iterative_solver_parameters(condensed=True, dtn_representation="lowrank")
    assert options["dtn_fieldsplit_0_pc_python_type"] == "gadopt.ReducedGIABlockPC"
    assert options["dtn_fieldsplit_0_condensed_fieldsplit_1_pc_python_type"] == "gadopt.LowRankPotentialPC"


def test_the_preset_refuses_an_unknown_representation():
    with pytest.raises(ValueError, match="must be 'multiplier' or 'lowrank'"):
        selfgrav_dtn_iterative_solver_parameters(
            condensed=False, dtn_representation="low_rank")


def test_a_low_rank_preset_on_a_multiplier_solver_is_refused(meshes):
    """This direction raises late and unhelpfully, so it is caught early.

    Without the check the failure is inside `PCSetUp`, reported as an
    unhandled Python exception under a PETSc error code.
    """
    with pytest.raises(ValueError,
                       match="LowRankPotentialPC needs dtn_representation='lowrank'"):
        build(meshes, "multiplier", solver_parameters=preset("lowrank"))


def test_a_multiplier_preset_on_a_low_rank_solver_is_refused(meshes):
    """This direction also raises late: plain GAMG refuses the Python block.

    `CondensedBlockPC.initialize` installs the Python potential block whenever
    `dtn_operator` is in the context, and `PCSetUp_GAMG` then fails with `No
    method getinfo for Mat of type python`, an unhandled exception under a
    PETSc error code. The early check gives a message that names the fix.
    """
    with pytest.raises(ValueError,
                       match="low-rank solver needs gadopt.LowRankPotentialPC"):
        build(meshes, "lowrank", solver_parameters=preset("multiplier"))


@pytest.mark.parametrize("kind", ["assembled", "wrong_context"])
def test_the_class_refuses_a_preconditioning_matrix_it_cannot_read(kind):
    """Selected anywhere but on the Python potential block, it says so.

    Two ways to get there: on the multiplier path the potential block is an
    assembled matrix, and outside `CondensedBlockPC` altogether it may be a
    Python matrix carrying something else. The assembled case is why the class
    tests the matrix type before asking for a context: `getPythonContext` on an
    assembled matrix raises inside PETSc, and this runs in a PETSc callback,
    where that surfaces as an error code far from its cause.
    """
    from firedrake.petsc import PETSc

    if kind == "assembled":
        matrix = PETSc.Mat().createAIJ([2, 2], comm=fd.COMM_WORLD)
        matrix.setUp()
        matrix.assemble()
    else:
        matrix = PETSc.Mat().createPython([2, 2], object(),
                                          comm=fd.COMM_WORLD)
        matrix.setUp()
    pc = PETSc.PC().create(comm=fd.COMM_WORLD)
    pc.setOperators(matrix, matrix)
    with pytest.raises(ValueError, match="_LowRankPotentialOperator"):
        LowRankPotentialPC().initialize(pc)


# ---------------------------------------------------------------------------
# 4. The installation
# ---------------------------------------------------------------------------
def test_install_refuses_a_jacobian_that_is_not_matrix_free():
    """A non-Python Jacobian is refused with a message, not a crash.

    `install_augmented_context` swaps the Python context of a matrix-free
    Jacobian. On any other matrix type `getPythonContext` reads invalid
    memory, and the process dies. The pyop3 Firedrake produces such a matrix
    (a nest matrix in place of the requested matrix-free one, for every mixed
    form with a Real subspace), so the guard must turn it into an error that
    names the cause.
    """
    from firedrake.petsc import PETSc
    from gadopt.dtn_coupled import install_augmented_context

    # A small assembled matrix stands for the wrong Jacobian. The DtN operator
    # is never reached, so None is enough.
    matrix = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
    matrix.setUp()
    matrix.assemble()
    try:
        with pytest.raises(RuntimeError, match="matrix-free Jacobian"):
            install_augmented_context(matrix, None)
    finally:
        matrix.destroy()
