"""Tolerances for the weak boundary condition diagnostics of the viscoelastic solvers.

The entrypoints in this directory do the assembly and the solves under
`doit run_case` and write their numbers to `.dat` files. This file reads those
files and applies the tolerances, so a failure names the case and prints the
number that missed.

Running these tests without running `doit run_case` first fails on the missing
file, which is how every case directory in `tests/` behaves.
"""

from math import log2
from pathlib import Path

import numpy as np
import pytest

from .gia_helpers import (
    BLOCK_SYMMETRY_CASES,
    REFINEMENT_RESOLUTIONS,
    WEAK_U_HISTORIES,
)
from .meta import MESHES, REFINEMENT_DT_OVER_TAU

CASE_DIR = Path(__file__).parent

# Symmetry is measured relative to the norm of the matrix itself, so this is a
# dimensionless distance from exact symmetry.
SYMMETRY_RTOL = 1e-13
# The variational structure identity is exact in exact arithmetic.
STRUCTURE_RTOL = 1e-12
# The displacement space is P2, so both the gap between the two formulations and
# the normal jump on the weak boundary must fall at least this fast.
REFINEMENT_ORDER = 2.0
# Below this fraction of the displacement norm the coarse gap is round-off, and
# a rate computed from it would be measuring noise instead of a difference
# between the two formulations.
GAP_FLOOR = 1e-8


def load(name):
    """Read one diagnostics file as rows and columns.

    `loadtxt` collapses a file holding a single row, or a single column, to one
    dimension, and the two cases are then indistinguishable. `ndmin=2` keeps the
    shape the file was written with, so every caller can index it as
    `[row][column]`.
    """
    return np.loadtxt(CASE_DIR / name, ndmin=2)


@pytest.mark.parametrize("mesh_key", MESHES)
def test_coupled_displacement_block_symmetry(mesh_key):
    """The displacement block of the coupled Jacobian is symmetric.

    At fixed history the coupled momentum stress is elastic, so a symmetrising
    term built with the effective viscosity instead of the elastic modulus
    leaves this block asymmetric by an amount that grows with dt/tau. The
    default coupled preset preconditions the block with CG, which assumes a
    symmetric operator, so this is a solver-level requirement.
    """
    rows = load(f"block_symmetry-{mesh_key}.dat")
    for (dt_over_tau, exponent), (ratio,) in zip(BLOCK_SYMMETRY_CASES, rows):
        assert ratio <= SYMMETRY_RTOL, (
            f"dt/tau={dt_over_tau}, exponent={exponent}: "
            f"asymmetry {ratio:.3e}")


@pytest.mark.parametrize("mesh_key", MESHES)
def test_pointwise_residual_is_first_variation(mesh_key):
    """The pointwise weak "un" residual is the first variation of its functional.

    `InternalVariableSolver` substitutes the backward-Euler update into the
    stress, so the variation carries the effective viscosity, which is what
    makes its symmetrising term and its penalty share a coefficient.
    """
    (ratio,), = load(f"pointwise_structure-{mesh_key}.dat")
    assert ratio <= STRUCTURE_RTOL, f"structure error {ratio:.3e}"


@pytest.mark.parametrize("mesh_key", MESHES)
def test_coupled_residual_is_first_variation(mesh_key):
    """The coupled weak "un" residual is the first variation at fixed history.

    The coupled stress is elastic in the displacement, so both its variation and
    its penalty carry the elastic modulus. That is what distinguishes this
    reference from the pointwise one.
    """
    (ratio,), = load(f"coupled_structure-{mesh_key}.dat")
    assert ratio <= STRUCTURE_RTOL, f"structure error {ratio:.3e}"


@pytest.mark.parametrize("mesh_key", MESHES)
def test_weak_u_jacobian_symmetric(mesh_key):
    """The weak "u" branch has a symmetric Jacobian for the viscoelastic stress.

    The branch has to work for a stress with a bulk part: the tangent and the
    penalty both pick that part up, and the residual is the first variation of a
    boundary functional, so its Jacobian is a Hessian.
    """
    rows = load(f"weak_u_symmetry-{mesh_key}.dat")
    for history, (ratio,) in zip(WEAK_U_HISTORIES, rows):
        assert ratio <= SYMMETRY_RTOL, f"{history}: asymmetry {ratio:.3e}"


@pytest.mark.parametrize("dt_over_tau", REFINEMENT_DT_OVER_TAU)
def test_coupled_matches_pointwise_under_refinement(dt_over_tau):
    """The coupled and pointwise solutions converge together under refinement.

    The coupled displacement rows differ from the pointwise ones by terms
    proportional to the normal jump on the weak boundary, and those terms vanish
    when the exact solution satisfies the boundary condition. Both the gap and
    the jump must therefore fall at the P2 order.

    This bounds the size of the difference between the two formulations. It does
    not detect an asymmetric displacement block;
    `test_coupled_displacement_block_symmetry` does that.
    """
    rows = load(f"refinement-{dt_over_tau}.dat")
    assert len(rows) == len(REFINEMENT_RESOLUTIONS)
    gaps, normal_jumps, displacement_norms = rows[:, 0], rows[:, 1], rows[:, 2]

    # The coarse gap must be a real difference between the two formulations and
    # not round-off, or the rates below are computed from noise.
    assert gaps[0] >= GAP_FLOOR * displacement_norms[0], (
        f"gap {gaps[0]:.3e} is at round-off relative to the displacement norm "
        f"{displacement_norms[0]:.3e}")

    gap_rate = log2(gaps[0] / gaps[1])
    jump_rate = log2(normal_jumps[0] / normal_jumps[1])
    assert gap_rate >= REFINEMENT_ORDER, f"gaps {gaps}, rate {gap_rate:.2f}"
    assert jump_rate >= REFINEMENT_ORDER, (
        f"normal jumps {normal_jumps}, rate {jump_rate:.2f}")


def test_coupled_iterative_preset_converges():
    """The coupled iterative preset converges with weak "un" boundaries.

    The preset eliminates the internal variables by static condensation and
    solves the condensed displacement operator with CG, which is defined only
    for a symmetric operator. A PETSc converged reason is positive when the
    solve converged.
    """
    (snes_reason, ksp_reason), = load("iterative_preset.dat")
    assert snes_reason > 0, f"SNES diverged, reason {int(snes_reason)}"
    assert ksp_reason > 0, (
        f"condensed displacement CG diverged, reason {int(ksp_reason)}")
