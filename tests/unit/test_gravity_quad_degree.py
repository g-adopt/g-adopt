"""The mesh-aware boundary quadrature default, and the check behind it.

Two things are under test and they are separable.

The **default** (`GravitySolver.default_quad_degree`) replaces
`2 (max_mode + element_degree)`, which has no mesh dependence at all, with a
rule in `L h_max / R` - how much of one oscillation falls inside one facet.
What must hold is that it follows the mesh, that it stays odd (a degree-`q`
rule uses `(q + 2) // 2` points, so `q` even and `q + 1` are the same rule and
the odd one is exact to one degree more), that it never costs more than the
default it replaces, and above all that the degree it picks **actually
integrates the modes**.

The **check** (`check_boundary_quadrature`) differences two degrees on one
mesh, which cancels the geometry error that an analytic normalisation would
measure instead of the quadrature.

The fast tests here check the cap, the facet scale, an explicit degree, the
fallback without a DtN boundary, CG2, and the agreement of the low-rank
path. The rest (the degree follows the mesh and stays odd, never costs more
than the incumbent, every mode is integrated, the warnings fire and stay
quiet, silence implies adequacy on warped shells, the check is an
instrument, rank independence) are rows of the `quadrature-*` entries of
the `gravity` case in `tests/gia` (driver `gia_gravity.py`). The mesh
builders are in `tests/gia/gia_models/gravity_quadrature.py`, which that
driver and this module share.

Calibration and the reason the earlier constants were wrong:
`NOTES/FINDING-QUADRATURE-DEGREE-FORM.md`,
`NOTES/FINDING-QUADRATURE-CANCELLATION.md`.
"""

import firedrake as fd
import numpy as np
import pytest

from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gia_models.gravity_quadrature import (
    annulus,
    build,
    incumbent,
    shell,
)


#: The target the rule was calibrated to reach.
TARGET = 1e-8


class TestDefaultDegree:
    def test_stops_growing_once_the_mesh_resolves_the_mode(self):
        """The `O(L^4) -> O(L^2)` property the low-rank build needs.

        Doubling the truncation on a mesh fine enough to resolve it must not
        double the degree. The incumbent does exactly that, which is what makes
        the low-rank build - carrying `(q/2 + 1)^2` per mode - quartic in `L`.
        """
        fine = annulus(384)
        low = build(fine, CylindricalDtN(M=8)).quad_degree
        high = build(fine, CylindricalDtN(M=16)).quad_degree
        assert high - low <= 2
        assert incumbent(CylindricalDtN(M=16)) - incumbent(CylindricalDtN(M=8)) == 16

    def test_the_cap_can_bind_and_the_constructor_says_so(self):
        """The known limitation, pinned rather than left to be rediscovered.

        The cap keeps the new default from ever costing more than the old one,
        which on a coarse enough boundary means shipping a degree the rule
        itself would have raised - a level-1 cubed sphere has facets spanning
        so much of the sphere that `L h_max / R` asks for more than
        `2 (L + 1)`. The result is no worse than the default it replaces, which
        is the guarantee, but it is not adequate either.

        The constructor says so without measuring anything: it already knows
        what the rule asked for and what the cap allowed, and comparing two
        integers is free. See `warn_on_quadrature_rule_limits`.
        """
        descriptor = SphericalDtN(L=2)
        psi = fd.Function(fd.FunctionSpace(shell(1), "CG", 1))
        with pytest.warns(UserWarning, match="capped"):
            solver = GravitySolver(
                psi, 0.0, bcs={"top": {"dtn": descriptor}},
                solver_parameters="iterative")
        assert solver.quad_degree == incumbent(descriptor)
        assert solver.quad_rule_report.requested > solver.quad_rule_report.incumbent
        # and the measurement, when asked for, agrees that it is inadequate
        assert solver.check_boundary_quadrature(action="warn") > 1e-8

    def test_facet_scale_matches_the_calibration_definition(self):
        """`h_mean` is mean(sqrt(area)), never sqrt(mean(area)).

        Pinned directly, because the two agree in 2-D, differ by only a few
        per cent in 3-D, and the difference is invisible in every other
        assertion in this file while being enough to break the bounds.
        """
        mesh = shell(2, warp=0.20)
        solver = build(mesh, SphericalDtN(L=3))
        probe = solver.geometry_probe_measure()
        h_max, h_mean, radius = solver.boundary_facet_scale(probe, "top")

        areas = np.asarray(fd.assemble(
            fd.TestFunction(fd.FunctionSpace(mesh, "DG", 0))
            * probe("top")).dat.data_ro, dtype=float)
        areas = areas[areas > 0]
        assert np.isclose(h_mean, np.sqrt(areas).mean(), rtol=1e-12)
        assert np.isclose(h_max, np.sqrt(areas).max(), rtol=1e-12)
        assert h_mean < np.sqrt(areas.mean())      # Jensen, strictly

    def test_explicit_degree_is_honoured(self):
        solver = build(annulus(96), CylindricalDtN(M=4), quad_degree=17)
        assert solver.quad_degree == 17

    def test_no_dtn_boundary_falls_back(self):
        """Nothing to size the rule against; behaviour must not change."""
        mesh = annulus(96)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(psi, 0.0, bcs={"top": {"flux": 0.0}},
                               solver_parameters="direct")
        assert solver.quad_degree == 2 * (1 + 1)


class TestDefaultIsAdequate:
    """The property the whole change exists for: the chosen degree integrates.

    Self-convergence over *every* treated mode, at the degree the rule picks,
    against the 1e-8 the constants were calibrated to.
    """

    def test_holds_at_cg2(self):
        solver = build(annulus(192, irregular=True), CylindricalDtN(M=16),
                       element_degree=2)
        assert solver.check_boundary_quadrature(sample="all") < TARGET


class TestLowRankPathAgrees:
    """The default reaches both representations, and must not break either."""

    def test_both_paths_take_the_same_degree(self):
        mesh = annulus(192)
        descriptor = CylindricalDtN(M=8)
        multiplier = build(mesh, descriptor)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        lowrank = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=8)}},
            solver_parameters="direct", dtn_representation="lowrank")
        assert lowrank.quad_degree == multiplier.quad_degree

    def test_trace_build_self_assertion_passes_at_the_default(self):
        """The low-rank build asserts its own node coincidence unconditionally.

        The default is now odd, where it was always even before, and the
        coincidence `(q + 2) // 2 - 1 == q // 2` has to hold at both parities
        for that assertion to keep passing. Constructing the solver is the
        test: the assertion raises on failure.
        """
        mesh = annulus(192)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(
            psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=12)}},
            solver_parameters="direct", dtn_representation="lowrank")
        assert solver.quad_degree % 2 == 1
        assert all(rows.used_trace_build for rows in solver.mode_rows)
        assert all(rows.trace_degree == solver.quad_degree // 2
                   for rows in solver.mode_rows)
