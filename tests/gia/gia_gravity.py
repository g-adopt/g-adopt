r"""Driver of the `gravity` case: `GravitySolver` alone, against closed forms.

Usage: `python gia_gravity.py --entry <name>` (under `mpiexec -n 2` for an
entry whose name ends in `-np2`), with the entry names of
`gia_entries.GRAVITY`.

Every entry solves the gravitational Poisson problem

    -lap psi = 4 pi G rho   in the domain,

with DtN maps, `psi` (Dirichlet), `flux` and `sigma` (sheet) conditions on
its boundaries, and writes measured numbers as `name value` lines to
`out/gravity-<entry>.dat`. The test file `test_gia_gravity_case.py` holds
the tolerances of the unit tests that the rows come from and asserts; this
script asserts nothing. The only recorded values of the case are the CG
counts of the two serial 2-D closed-form entries, which these entries also
write to `out/gravity-<entry>-counts.dat`.

The groups of entries:

- `closed-*`: solves against closed forms on two or three meshes of one
  family, each mesh half the size of the one before, so that the observed
  order of the relative L2 error `log2(e_coarse / e_fine)` can be compared
  with the order 3 of a CG2 solution with degree-2 coordinates. The 2-D
  entries run the boundary sheets, the volumetric shell, the uniform disc
  (the 2-D monopole datum), the Dirichlet and flux conditions, every
  channel of enclosed mass and the gauge of the monopole datum on one DtN
  representation each; the low-rank entry also solves the parity problems
  of `test_dtn_lowrank_parity.py` on both representations in one process,
  and the multiplier entry the setup refusals and the field-number split of
  `test_gravity_solver.py`. The 3-D entries run the `Y_lm` sheet and the
  interior monopole on the cubed-sphere shell. The CG counts come from the
  same sheet problem with the iterative preset;
- `submesh`: a density on a `Submesh` against the same-mesh solution, the
  iterative preset against the direct one, and the low-rank against the
  multiplier representation on that configuration;
- `adjoint-*`: the Taylor, replay, tangent and closed-form gradient checks
  of `test_gravity_adjoint.py` (multiplier) and
  `test_gravity_adjoint_lowrank.py` (low-rank);
- `quadrature-*`: the boundary quadrature rule of `DtNGravityForm` and its
  check (`test_gravity_quad_degree.py`). These build solvers and integrate
  the DtN modes; they do not solve, so their 3-D meshes run anywhere.

The closed forms used below, with `G = 1` throughout:

- a sheet `sigma = cos(m phi)` on a circle of radius `a` has the potential
  `(2 pi a / m) (r_< / r_>)^m cos(m phi)` on both sides;
- a sheet `sigma = Y_lm` on a sphere of radius `a` has the interior
  potential `(4 pi a / (2l + 1)) (r / a)^l Y_lm`;
- a uniform sheet `sigma` on the inner sphere `r = a` of a 3-D shell has
  the domain-side potential `4 pi sigma a^2 / r`;
- a uniform disc of unit density has `psi = -pi (r^2 - R^2)` in the gauge
  of the solver, which is quadratic and so exact in CG2: the error left is
  the polygon of `UnitDiskMesh`, which converges at order 2;
- in the annulus with a unit density, a sheet `sigma` and a cavity mass
  `M_c` on the inner boundary, `psi = -pi r^2 + B ln r` with
  `B = 2 pi a^2 - 4 pi sigma a - 2 M_c`, up to the constant that the gauge
  (mean of `psi` on the outer boundary is zero) fixes;
- the shell density `cos(m phi)` on `[r1, r2]` has the mode
  `(2 pi / m) [r^-m I_in(r) + r^m I_out(r)]`, `I_in = int r'^(m+1) dr'`
  from `r1` to `min(max(r, r1), r2)`, `I_out = int r'^(1-m) dr'` from there
  to `r2`.

The rows reuse the meshes and helpers of the unit tests, which this script
imports from `tests/unit`:

- `test_gravity_solver.py`: `annulus_mesh`, `shell_mesh_3d`,
  `relative_l2_error`, `annulus_source`, `solve_annulus`,
  `field_number_parameters`, `max_coefficient_difference`, `RMIN`, `RMAX`,
  `R1_SHELL`, `R2_SHELL`;
- `test_dtn_lowrank_parity.py`: `solve_both`, `shell_source`, `tight`,
  `multiplier_potential_iterations`;
- `test_gravity_adjoint.py`: the control and direction builders, the
  forward maps, `tape_forward`, `trace_coeff_integral`,
  `trace_coeff_rdirect`, `real_scalar`, the tape inspection helpers;
- `test_gravity_adjoint_lowrank.py`: the low-rank forward maps;
- `test_gravity_quad_degree.py`: `annulus`, `shell`, `icosahedral_shell`,
  `build`, `incumbent`, `TARGET`.

When those unit files are deleted, these helpers must move to `tests/gia`
first.
"""

import argparse
import sys
import warnings
from collections import Counter
from contextlib import contextmanager, nullcontext

import gadopt  # imported before firedrake
import firedrake as fd
import numpy as np
from firedrake.adjoint import (Control, ReducedFunctional, continue_annotation,
                               get_working_tape, pause_annotation, stop_annotating,
                               taylor_test)
from firedrake.petsc import PETSc
from gadopt import CylindricalDtN, GravitySolver, SphericalDtN
from gadopt.gravity_solver import (QUADRATURE_ELEMENT_DEGREE_COST,
                                   QUADRATURE_RULE_COEFFICIENTS)
from gadopt.spherical_harmonics import real_spherical_harmonic
from pyadjoint import AdjFloat
from pyadjoint.tape import annotate_tape

import gia_march as M
from gia_entries import (CASE_DIR, GRAVITY, GRAVITY_DISC_LEVELS,
                         GRAVITY_LEVELS_2D, GRAVITY_LEVELS_3D, GRAVITY_MASS_CHANNELS,
                         GRAVITY_MONOPOLE_CONTROLS, GRAVITY_QUADRATURE_SWEEP,
                         GRAVITY_SHEETS, GRAVITY_TOLERANCE_SWEEP, gravity_counts,
                         gravity_quadrature_entry, gravity_values)

sys.path.insert(0, str(CASE_DIR.parent / "unit"))
import test_dtn_lowrank_parity as TL  # noqa: E402  (after the path of tests/unit)
import test_gravity_adjoint as TA  # noqa: E402
import test_gravity_adjoint_lowrank as TAL  # noqa: E402
import test_gravity_quad_degree as TQ  # noqa: E402
import test_gravity_solver as TG  # noqa: E402

RMIN, RMAX = TG.RMIN, TG.RMAX


# -- Meshes -------------------------------------------------------------------------

#: The 2-D annulus family of the closed-form rows: the azimuthal count and
#: the radial layer height halve together, so each level has half the facet
#: size of the one before. The finest level is finer than the unit meshes
#: (192 facets, `dr = 0.1`), so the unit bounds hold there with margin; the
#: order between the two finest levels is the measured quantity.
ANNULUS_LEVELS = ((48, 0.25), (96, 0.125), (192, 0.0625))
#: The same family for the problems that the unit tests solved on 20 radial
#: layers (`dr = 0.05`): every channel of enclosed mass.
MASS_LEVELS = ((48, 0.2), (96, 0.1), (192, 0.05))
#: The meshes of the volumetric shell, which conform to the shell
#: boundaries `R1_SHELL` and `R2_SHELL`. The finest level is the unit mesh.
SHELL_LEVELS = ((64, 0.2), (128, 0.1), (256, 0.05))
#: The 3-D cubed-sphere shells: refinement level and radial layer height.
SPHERE_LEVELS = ((2, 0.25), (3, 0.125))
assert len(ANNULUS_LEVELS) == len(MASS_LEVELS) == len(SHELL_LEVELS) == GRAVITY_LEVELS_2D
assert len(SPHERE_LEVELS) == GRAVITY_LEVELS_3D
assert len(GRAVITY_DISC_LEVELS) == GRAVITY_LEVELS_2D


def polar(mesh):
    """The radius and the azimuth of the coordinates of `mesh`, as UFL."""
    X = fd.SpatialCoordinate(mesh)
    return X, fd.sqrt(fd.dot(X, X)), fd.atan2(X[1], X[0])


def put_coefficients(out, prefix, solver):
    """Write every trace coefficient of `solver` as `<prefix>_<bc>_<key>`.

    The low-rank and the multiplier entries write the same names, so the test
    compares the two representations coefficient by coefficient.
    """
    for bc_id, values in solver.coefficients().items():
        for key, value in values.items():
            out[f"{prefix}_{bc_id}_{key}"] = float(value)


def inactive_coefficient(solver, active):
    """The largest trace coefficient whose key is not `active`, on every boundary."""
    return max(abs(float(value)) for values in solver.coefficients().values()
               for key, value in values.items() if key != active)


# -- closed-2d ----------------------------------------------------------------------

def sheet_rows(out, representation):
    """Boundary sheets `cos(m phi)` on a DtN boundary, three meshes each.

    `TestCylindrical::test_boundary_sheet`: the domain-side potential of a
    sheet at radius `a` is the single harmonic `(2 pi a / m) (r_< /
    r_>)^m cos(m phi)`, with `r / a` inside the outer boundary and `a / r`
    outside the inner one. Each level writes the relative L2 error, the
    trace coefficient of the active mode (the amplitude times
    `(a / a)^m = 1`), the largest inactive coefficient (zero by the
    orthogonality of the modes) and every coefficient. On the low-rank
    path, CG runs on `A + B` with an LU factorisation of `A` as its
    preconditioner, so its count measures the low-rank update and not the
    mesh.
    """
    for side, m_mode in GRAVITY_SHEETS:
        name = f"sheet_{side}{m_mode}"
        a = RMAX if side == "top" else RMIN
        amplitude = 2 * np.pi * a / m_mode
        out[f"{name}_amplitude"] = amplitude
        for k, (n_azimuthal, dr) in enumerate(ANNULUS_LEVELS):
            mesh = TG.annulus_mesh(n_azimuthal=n_azimuthal, dr=dr)
            _, r, phi = polar(mesh)
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
            solver = GravitySolver(
                psi, 0.0,
                bcs={"top": {"dtn": CylindricalDtN(M=4)},
                     "bottom": {"dtn": CylindricalDtN(M=4)},
                     side: {"dtn": CylindricalDtN(M=4), "sigma": fd.cos(m_mode * phi)}},
                dtn_representation=representation)
            solver.solve()
            radial = (r / a) ** m_mode if side == "top" else (a / r) ** m_mode
            reference = amplitude * radial * fd.cos(m_mode * phi)
            out[f"{name}_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)
            out[f"{name}_coefficient_{k}"] = float(solver.coefficients()[side][f"cos{m_mode}"])
            out[f"{name}_inactive_{k}"] = inactive_coefficient(solver, f"cos{m_mode}")
            put_coefficients(out, f"coefficient_{name}_{k}", solver)
            if representation == "lowrank":
                out[f"{name}_cg_direct_{k}"] = solver.ksp.getIterationNumber()


def count_rows(out, counts, representation, levels, mesh_of, bcs_of, degree):
    """CG counts of the potential block under the iterative preset, per level.

    CG preconditioned by algebraic multigrid has a count that does not grow
    with the mesh. On the multiplier path the count is that of the potential
    block inside the Schur split (the outer FGMRES count is 1, because a full
    Schur factorisation is exact); on the low-rank path it is the count of
    CG on `A + B` with multigrid built from `A` alone
    (`test_dtn_lowrank_parity.py::test_cg_iteration_counts`).

    Args:
      out: the values of the entry.
      counts: the counts of the entry, written also to the counts file.
      representation: the DtN representation.
      levels: the mesh parameters of each level.
      mesh_of: builds the mesh of one level.
      bcs_of: builds the boundary conditions on a mesh.
      degree: the degree of the potential space.
    """
    for k, level in enumerate(levels):
        mesh = mesh_of(*level)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", degree))
        solver = GravitySolver(psi, 0.0, bcs=bcs_of(mesh), solver_parameters="iterative",
                               dtn_representation=representation)
        solver.solve()
        if representation == "lowrank":
            count = solver.ksp.getIterationNumber()
        else:
            count = TL.multiplier_potential_iterations(solver)
        counts[f"cg_{k}"] = count
        out[f"cg_{k}"] = count


def shell_rows(out, representation):
    """The volumetric shell `cos(3 phi)` on `[R1_SHELL, R2_SHELL]`, three meshes.

    `TestCylindrical::test_volumetric_shell`. The meshes conform to the shell
    boundaries and the source is integrated at degree `2 m + 8`, so the
    remaining error is that of the potential. The closed form is evaluated
    at degree 10.
    """
    m_mode = 3

    def inner_integral(lower, upper):
        # int r^(m+1) dr: the interior moment of the shell below r.
        return (upper ** (m_mode + 2) - lower ** (m_mode + 2)) / (m_mode + 2)

    def outer_integral(lower, upper):
        # int r^(1-m) dr: the exterior moment of the shell above r (m != 2).
        return (upper ** (2 - m_mode) - lower ** (2 - m_mode)) / (2 - m_mode)

    for k, (n_azimuthal, dr) in enumerate(SHELL_LEVELS):
        mesh = TG.annulus_mesh(n_azimuthal=n_azimuthal, dr=dr,
                               interfaces=(TG.R1_SHELL, TG.R2_SHELL))
        _, r, phi = polar(mesh)
        shell = fd.conditional(fd.And(r >= TG.R1_SHELL, r <= TG.R2_SHELL), 1.0, 0.0)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, fd.cos(m_mode * phi) * shell,
            bcs={"top": {"dtn": CylindricalDtN(M=5)},
                 "bottom": {"dtn": CylindricalDtN(M=5)}},
            source_quad_degree=2 * m_mode + 8, dtn_representation=representation)
        solver.solve()
        r_in = fd.min_value(fd.max_value(r, TG.R1_SHELL), TG.R2_SHELL)
        radial = (r ** -m_mode * inner_integral(TG.R1_SHELL, r_in)
                  + r ** m_mode * outer_integral(r_in, TG.R2_SHELL))
        reference = (2 * np.pi / m_mode) * radial * fd.cos(m_mode * phi)
        out[f"shell_error_{k}"] = TG.relative_l2_error(psi, reference, mesh, degree=10)
        put_coefficients(out, f"coefficient_shell_{k}", solver)


def disc_rows(out, representation):
    """The uniform disc, the sharpest check of the 2-D monopole datum.

    `TestCylindrical::test_monopole_uniform_disc`: `psi = -pi (r^2 - 1)` in
    the gauge of the solver, exact in CG2, so the error is that of the
    polygonal boundary of `UnitDiskMesh` (order 2). The mean of `psi` on the
    boundary is the gauge itself and follows the same geometry error down.
    """
    for k, refinement in enumerate(GRAVITY_DISC_LEVELS):
        mesh = fd.UnitDiskMesh(refinement_level=refinement)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(psi, 1.0, bcs={1: {"dtn": CylindricalDtN(M=3)}},
                               solver_parameters="direct",
                               dtn_representation=representation)
        solver.solve()
        X = fd.SpatialCoordinate(mesh)
        reference = -np.pi * (fd.dot(X, X) - 1.0)
        out[f"disc_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)
        perimeter = fd.assemble(1 * solver.ds(1))
        out[f"disc_mean_{k}"] = abs(fd.assemble(psi * solver.ds(1)) / perimeter)


def dirichlet_flux_rows(out, representation):
    """The `psi` and `flux` conditions reproduce a harmonic, three meshes.

    `TestCylindrical::test_dirichlet_and_flux`: `psi = (r / RMAX)^2 cos(2
    phi)` is prescribed on the outer boundary and its normal derivative
    `-(m / r) psi` (outward normal `-r_hat`) on the inner one. No DtN
    boundary: on the low-rank path the mode set is empty.
    """
    m_mode = 2
    for k, (n_azimuthal, dr) in enumerate(ANNULUS_LEVELS):
        mesh = TG.annulus_mesh(n_azimuthal=n_azimuthal, dr=dr)
        _, r, phi = polar(mesh)
        reference = (r / RMAX) ** m_mode * fd.cos(m_mode * phi)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(
            psi, 0.0,
            bcs={"top": {"psi": reference},
                 "bottom": {"flux": -(m_mode / r) * reference}},
            quad_degree=12, dtn_representation=representation)
        solver.solve()
        out[f"dirichlet_flux_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)


def mass_rows(out, representation):
    """Density, sheets and prescribed fluxes all enclose mass, three meshes.

    `TestCylindrical::test_monopole_counts_every_mass_channel`. A unit
    density fills the annulus; the `sheet` channel adds `sigma = 0.3` on the
    inner DtN boundary, the `flux` channel replaces the inner DtN map by a
    flux `2 M_c / RMIN` that implies a cavity mass `M_c = 2`. The enclosed
    mass is `pi (RMAX^2 - RMIN^2) + 2 pi RMIN sigma + M_c`, and the potential
    is `-pi r^2 + B ln r`, `B = 2 pi RMIN^2 - 4 pi sigma RMIN - 2 M_c`,
    re-centred so that its mean on the outer boundary is zero, which is the
    gauge of the solver. The mean of `psi` on the outer boundary is written
    too; it is not zero on the coarse meshes, where the discrete boundary
    length differs from `2 pi R`.
    """
    for channel in GRAVITY_MASS_CHANNELS:
        sigma = 0.3 if channel == "sheet" else 0.0
        cavity = 2.0 if channel == "flux" else 0.0
        expected = np.pi * (RMAX ** 2 - RMIN ** 2) + 2 * np.pi * RMIN * sigma + cavity
        out[f"mass_{channel}_expected"] = expected
        for k, (n_azimuthal, dr) in enumerate(MASS_LEVELS):
            mesh = TG.annulus_mesh(n_azimuthal=n_azimuthal, dr=dr)
            inner = ({"flux": fd.Constant(2 * cavity / RMIN)} if channel == "flux"
                     else {"dtn": CylindricalDtN(M=3)})
            if sigma:
                inner["sigma"] = fd.Constant(sigma)
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
            solver = GravitySolver(psi, 1.0,
                                   bcs={"top": {"dtn": CylindricalDtN(M=3)}, "bottom": inner},
                                   solver_parameters="direct",
                                   dtn_representation=representation)
            solver.solve()
            out[f"mass_{channel}_mass_{k}"] = solver.total_enclosed_mass()
            _, r, _ = polar(mesh)
            log_coefficient = 2 * np.pi * RMIN ** 2 - 4 * np.pi * sigma * RMIN - 2 * cavity
            reference = -np.pi * r ** 2 + log_coefficient * fd.ln(r)
            perimeter = fd.assemble(1 * solver.ds("top"))
            reference -= fd.assemble(reference * solver.ds("top")) / perimeter
            out[f"mass_{channel}_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)
            out[f"mass_{channel}_mean_{k}"] = abs(
                fd.assemble(psi * solver.ds("top")) / perimeter)
            # The coefficients include the interior `mean` mode, which only a
            # problem with net mass excites; the representation comparison
            # of the test reads them.
            put_coefficients(out, f"coefficient_mass_{channel}_{k}", solver)


def gauge_rows(out, representation):
    """The Robin shift `alpha` fixes the constant without changing the answer.

    `TestCylindrical::test_monopole_gauge_does_not_depend_on_the_robin_shift`.
    The cancellation is exact only where the discrete boundary length equals
    `2 pi R`, which the degree-2 coordinates of the annulus give; that
    precondition is written as `gauge_length_ratio`. For each `alpha` the
    mean of `psi` on the outer boundary and the L2 difference to the first
    solution, relative to its norm, are written.
    """
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.1)
    length = fd.assemble(1 * fd.ds_t(domain=mesh))
    _, r, _ = polar(mesh)
    radius = fd.assemble(r * fd.ds_t(domain=mesh)) / length
    out["gauge_length_ratio"] = length / (2 * np.pi * radius)
    solutions = []
    for k, alpha in enumerate((0.5, 1.0, 3.0)):
        # A subclass per value, because `alpha` is a class attribute of the
        # solver that the boundary form reads at construction.
        shifted = type("ShiftedGravitySolver", (GravitySolver,), {"alpha": alpha})
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = shifted(psi, 1.0,
                         bcs={"top": {"dtn": CylindricalDtN(M=3)},
                              "bottom": {"dtn": CylindricalDtN(M=3)}},
                         solver_parameters="direct", dtn_representation=representation)
        solver.solve()
        perimeter = fd.assemble(1 * solver.ds("top"))
        out[f"gauge_mean_{k}"] = abs(fd.assemble(psi * solver.ds("top")) / perimeter)
        solutions.append(psi.copy(deepcopy=True))
    scale = np.sqrt(fd.assemble(solutions[0] ** 2 * fd.dx))
    for k, other in enumerate(solutions[1:], start=1):
        out[f"gauge_difference_{k}"] = np.sqrt(
            fd.assemble((other - solutions[0]) ** 2 * fd.dx)) / scale


def setup_rows(out):
    """The net-mass refusals and warnings of `TestSetup` (multiplier path).

    - `test_net_mass_is_carried_not_refused`: a unit density with one DtN
      boundary solves, and the enclosed mass is the area;
    - `test_dirichlet_with_net_mass_refused`: a net mass with a strong `psi`
      condition is refused, and a zero-mean density with the same condition
      solves to a nonzero potential;
    - `test_leakage_threshold_is_scale_invariant`: the refusal does not
      depend on the scale of `rho` (1, 1e-3, 1e-10);
    - `test_leakage_warning_band`: a mass of leakage size warns, a zero-mean
      and a deliberate density do not.

    Each refusal and warning is written as 1 (it happened) or 0.
    """
    mesh = TG.annulus_mesh(n_azimuthal=32, dr=0.5)
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    solver = GravitySolver(psi, 1.0, bcs={"top": {"dtn": CylindricalDtN(M=1)}})
    solver.solve()
    out["setup_net_mass"] = solver.total_enclosed_mass()
    out["setup_net_mass_area"] = fd.assemble(1 * solver.dx)

    def refused(rho):
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(psi, rho, bcs={"top": {"dtn": CylindricalDtN(M=1)},
                                              "bottom": {"psi": fd.Constant(0.0)}})
        try:
            solver.solve()
        except ValueError as error:
            return int("over-constrained" in str(error))
        return 0

    out["setup_dirichlet_net_mass_refused"] = refused(1.0)
    for k, magnitude in enumerate((1.0, 1e-3, 1e-10)):
        out[f"setup_leakage_refused_{k}"] = refused(fd.Constant(magnitude))

    _, _, phi = polar(mesh)
    Q = fd.FunctionSpace(mesh, "CG", 2)
    zero_mean = fd.Function(Q).interpolate(fd.cos(2 * phi))
    psi_zero = fd.Function(Q)
    GravitySolver(psi_zero, zero_mean,
                  bcs={"top": {"dtn": CylindricalDtN(M=3)},
                       "bottom": {"psi": fd.Constant(0.0)}}).solve()
    out["setup_zero_mean_dirichlet_norm"] = np.sqrt(fd.assemble(psi_zero ** 2 * fd.dx))

    def warned(rho_expr):
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        solver = GravitySolver(psi, fd.Function(Q).interpolate(rho_expr),
                               bcs={"top": {"dtn": CylindricalDtN(M=1)}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver.solve()
        return int(any("Net mass" in str(w.message) for w in caught))

    clean = fd.cos(2 * phi)
    out["setup_leakage_warns"] = warned(clean + 1e-6)
    out["setup_zero_mean_warns"] = warned(clean)
    out["setup_deliberate_warns"] = warned(fd.Constant(1.0))


def split_rows(out):
    """The split by index sets against the split by field numbers, one rank.

    `TestIndexSetSplit::test_matches_the_field_number_split`: on both presets
    the two descriptions run the same Krylov method on the same operator, so
    the potentials and the coefficients agree to round-off and the outer
    counts are equal.
    """
    mesh = TG.annulus_mesh(n_azimuthal=64, dr=0.25)
    rho = TG.annulus_source(mesh)
    for preset in ("iterative", "direct"):
        psi_new, solver_new = TG.solve_annulus(mesh, rho, 5, preset)
        psi_old, solver_old = TG.solve_annulus(
            mesh, rho, 5, TG.field_number_parameters(preset, solver_new._multiplier_offset,
                                                     solver_new.n_multipliers))
        out[f"split_{preset}_psi"] = TG.relative_l2_error(psi_new, psi_old, mesh)
        out[f"split_{preset}_coefficients"] = TG.max_coefficient_difference(
            solver_old.coefficients(), solver_new.coefficients())
        out[f"split_{preset}_count_new"] = solver_new.solver.snes.ksp.getIterationNumber()
        out[f"split_{preset}_count_old"] = solver_old.solver.snes.ksp.getIterationNumber()


def parity_rows(out):
    """Low-rank against multiplier in one process (`test_dtn_lowrank_parity.py`).

    The two representations discretise the same operator, so the potentials
    agree to round-off (relative L2) and so do the trace coefficients
    (absolute):

    - `two_boundaries`: both boundaries, so the interior `mean` mode, whose
      low-rank weight is negative, is present
      (`test_parity_annulus_two_boundaries`);
    - `sheet`: a `sigma` sheet on a DtN boundary (`test_parity_annulus_sheet`);
    - `strong_bc`: a prescribed `psi` beside a DtN boundary, where the
      columns of `C` must be zero at constrained degrees of freedom
      (`test_parity_with_strong_bc`).
    """
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.25)
    _, r, phi = polar(mesh)
    rho = TL.annulus_source(mesh)
    configurations = {
        "two_boundaries": (rho, lambda: {"top": {"dtn": CylindricalDtN(M=5)},
                                         "bottom": {"dtn": CylindricalDtN(M=5)}}, 1),
        "sheet": (0.0, lambda: {"top": {"dtn": CylindricalDtN(M=4),
                                        "sigma": fd.cos(3 * phi)},
                                "bottom": {"dtn": CylindricalDtN(M=4)}}, 2),
        "strong_bc": (0.0, lambda: {"top": {"psi": (r / RMAX) ** 2 * fd.cos(2 * phi)},
                                    "bottom": {"dtn": CylindricalDtN(M=4)}}, 2),
    }
    for name, (density, bcs, degree) in configurations.items():
        results = TL.solve_both(mesh, density, bcs, degree=degree)
        (psi_m, solver_m), (psi_l, solver_l) = results["multiplier"], results["lowrank"]
        out[f"parity_{name}_psi"] = TG.relative_l2_error(psi_l, psi_m, mesh)
        out[f"parity_{name}_coefficients"] = TG.max_coefficient_difference(
            solver_m.coefficients(), solver_l.coefficients())


def run_closed_2d(representation):
    """The rows of `closed-2d-<representation>`.

    Returns:
      The values, and the CG counts for `gravity_counts`.
    """
    out, counts = {"ranks": fd.COMM_WORLD.size}, {}
    sheet_rows(out, representation)
    count_rows(out, counts, representation, ANNULUS_LEVELS,
               lambda n, dr: TG.annulus_mesh(n_azimuthal=n, dr=dr),
               lambda mesh: {"top": {"dtn": CylindricalDtN(M=4),
                                     "sigma": fd.cos(2 * polar(mesh)[2])},
                             "bottom": {"dtn": CylindricalDtN(M=4)}}, 2)
    if representation == "lowrank":
        for k in range(GRAVITY_LEVELS_2D):
            counts[f"sheet_top2_cg_direct_{k}"] = out[f"sheet_top2_cg_direct_{k}"]
    shell_rows(out, representation)
    disc_rows(out, representation)
    dirichlet_flux_rows(out, representation)
    mass_rows(out, representation)
    gauge_rows(out, representation)
    if representation == "multiplier":
        setup_rows(out)
        split_rows(out)
    else:
        parity_rows(out)
    return out, counts


def run_closed_2d_two_ranks():
    """The rows of `closed-2d-lowrank-np2`: the sheets and the mass channels.

    On two ranks the boundary quadrature degree, the measured boundary
    radius (which sets the DtN eigenvalues `m / R`), the boundary area and
    the enclosed mass are all reductions over the ranks. The low-rank and
    the multiplier representations share these, so their parity on two ranks
    (`parallel-gravity-np2`) cannot see a reduction that is wrong for both;
    the comparison of these rows with the serial entry can. The other rows of
    the serial entry add compilation time and no reduction of their own.
    """
    out = {"ranks": fd.COMM_WORLD.size}
    sheet_rows(out, "lowrank")
    mass_rows(out, "lowrank")
    return out


# -- closed-3d ----------------------------------------------------------------------

def sphere_sheet_rows(out, counts, representation):
    """The sheet `Y_21` on the outer DtN boundary of the shell, two meshes.

    `TestSpherical::test_boundary_sheet`: the interior potential is
    `(4 pi a / (2l + 1)) (r / a)^l Y_lm` with `a = RMAX`, and the trace
    coefficient of `Y2,1` on the outer boundary is the amplitude. The
    default preset in 3-D is iterative, so the CG count of the potential
    block is written per level.
    """
    l_mode, m_order = 2, 1
    key = f"Y{l_mode},{m_order}"
    amplitude = 4 * np.pi * RMAX / (2 * l_mode + 1)
    out["sphere_sheet_amplitude"] = amplitude
    for k, (refinement, dr) in enumerate(SPHERE_LEVELS):
        mesh = TG.shell_mesh_3d(refinement_level=refinement, dr=dr)
        X, r, _ = polar(mesh)
        Y = real_spherical_harmonic(l_mode, m_order, X)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(psi, 0.0,
                               bcs={"top": {"dtn": SphericalDtN(L=3), "sigma": Y},
                                    "bottom": {"dtn": SphericalDtN(L=3)}},
                               dtn_representation=representation)
        solver.solve()
        reference = amplitude * (r / RMAX) ** l_mode * Y
        out[f"sphere_sheet_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)
        out[f"sphere_sheet_coefficient_{k}"] = float(solver.coefficients()["top"][key])
        out[f"sphere_sheet_inactive_{k}"] = inactive_coefficient(solver, key)
        put_coefficients(out, f"coefficient_sphere_sheet_{k}", solver)
        if representation == "lowrank":
            count = solver.ksp.getIterationNumber()
        else:
            count = TL.multiplier_potential_iterations(solver)
        counts[f"cg_{k}"] = count
        out[f"cg_{k}"] = count


def sphere_monopole_rows(out, representation):
    """A uniform sheet on the inner boundary: the `l = 0` machinery, two meshes.

    `TestSpherical::test_interior_monopole`: `psi = 4 pi sigma RMIN^2 / r`;
    the `Y0,0` trace coefficient on a boundary of radius `R` is
    `sqrt(4 pi) psi(R)`, because `Y_00 = 1 / sqrt(4 pi)`.
    """
    sigma = 1.3
    for k, (refinement, dr) in enumerate(SPHERE_LEVELS):
        mesh = TG.shell_mesh_3d(refinement_level=refinement, dr=dr)
        _, r, _ = polar(mesh)
        psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
        solver = GravitySolver(psi, 0.0,
                               bcs={"top": {"dtn": SphericalDtN(L=1)},
                                    "bottom": {"dtn": SphericalDtN(L=1), "sigma": sigma}},
                               dtn_representation=representation)
        solver.solve()
        reference = 4 * np.pi * sigma * RMIN ** 2 / r
        out[f"sphere_monopole_error_{k}"] = TG.relative_l2_error(psi, reference, mesh)
        for marker, radius in (("top", RMAX), ("bottom", RMIN)):
            trace = 4 * np.pi * sigma * RMIN ** 2 / radius
            out[f"sphere_monopole_{marker}_coefficient_{k}"] = float(
                solver.coefficients()[marker]["Y0,0"])
            out[f"sphere_monopole_{marker}_expected"] = np.sqrt(4 * np.pi) * trace
        put_coefficients(out, f"coefficient_sphere_monopole_{k}", solver)


def sphere_parity_rows(out):
    """3-D low-rank against multiplier, and the evidence that the gap is the tolerance.

    `test_parity_shell`: solved to `ksp_rtol` 1e-14 on both paths (on the
    multiplier path the potential block's own tolerance is set too, since
    the outer FGMRES converges in one iteration). `test_shell_parity_is_
    tolerance_limited`: the relative L2 gap at `rtol` 1e-9, 1e-11, 1e-13,
    which must follow `rtol` down if it is the Krylov tolerance and not the
    representation.
    """
    mesh = TG.shell_mesh_3d(refinement_level=2, dr=0.25)
    rho = TL.shell_source(mesh)

    def pair(rtol):
        result = {}
        for representation in ("multiplier", "lowrank"):
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, rho,
                                   bcs={"top": {"dtn": SphericalDtN(L=3)},
                                        "bottom": {"dtn": SphericalDtN(L=3)}},
                                   dtn_representation=representation,
                                   solver_parameters="iterative",
                                   solver_parameters_extra=TL.tight(representation, rtol))
            solver.mixed_solution.assign(0)
            solver.solve()
            result[representation] = (psi, solver)
        return result

    result = pair(1e-14)
    (psi_m, solver_m), (psi_l, solver_l) = result["multiplier"], result["lowrank"]
    out["parity_shell_psi"] = TG.relative_l2_error(psi_l, psi_m, mesh)
    out["parity_shell_coefficients"] = TG.max_coefficient_difference(
        solver_m.coefficients(), solver_l.coefficients())
    for k, rtol in enumerate(GRAVITY_TOLERANCE_SWEEP):
        result = pair(rtol)
        out[f"parity_shell_gap_{k}"] = TG.relative_l2_error(
            result["lowrank"][0], result["multiplier"][0], mesh)


def monopole_absent_rows(out):
    """3-D has no undetermined monopole (`TestSetup::test_monopole_absent_in_3d`).

    No boundary carries the 2-D monopole datum, and the enclosed mass that
    the datum would read stays at zero after a solve with net mass.
    """
    mesh = TG.shell_mesh_3d(refinement_level=1, dr=0.5)
    psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    solver = GravitySolver(psi, 1.0, bcs={"top": {"dtn": SphericalDtN(L=1)}})
    out["absent_monopole_boundaries"] = len(solver.monopole_boundaries)
    solver.solve()
    out["absent_enclosed_mass"] = solver.total_enclosed_mass()


def run_closed_3d(representation):
    """The rows of `closed-3d-<representation>`."""
    out, counts = {"ranks": fd.COMM_WORLD.size}, {}
    sphere_sheet_rows(out, counts, representation)
    sphere_monopole_rows(out, representation)
    if representation == "lowrank":
        sphere_parity_rows(out)
    else:
        monopole_absent_rows(out)
    return out, counts


# -- submesh ------------------------------------------------------------------------

def disc_submesh(refinement):
    """`UnitDiskMesh` relabelled with the cells `r < 0.55` as 99, and that `Submesh`."""
    mesh = fd.UnitDiskMesh(refinement_level=refinement)
    _, r, _ = polar(mesh)
    indicator = fd.Function(fd.FunctionSpace(mesh, "DG", 0)).interpolate(
        fd.conditional(r < 0.55, 1.0, 0.0))
    marked = fd.RelabeledMesh(mesh, [indicator], [99])
    return marked, fd.Submesh(marked, marked.topological_dimension, 99)


def relative_l2(first, second, mesh):
    """`|first - second| / |second|` in L2 on `mesh`; a reduction over all ranks."""
    return TG.relative_l2_error(first, second, mesh)


def run_submesh():
    """A density on a `Submesh` against the same density on the parent.

    - `TestCylindrical::test_cross_mesh_consistency`: `cos(2 phi)` on the
      cells `r < 0.55`, held in DG0 on the submesh and in DG0 times an
      indicator on the parent, gives the same potential. The unit test took
      the largest nodal difference; this row takes the relative L2
      difference, which does not depend on the partition;
    - `test_cross_mesh_iterative_preset`: the iterative preset against the
      direct one on the cross-mesh configuration, whose potential block is
      `blockdiag(A, DG0 dummy mass)` over two meshes;
    - `test_dtn_lowrank_parity.py::test_submesh_parity_and_the_mixed_space_disappearing`:
      the low-rank representation against the multiplier one on the same
      configuration (CG2), and the sizes of the two mixed spaces.
    """
    out = {"ranks": fd.COMM_WORLD.size}
    marked, submesh = disc_submesh(3)
    _, r, phi = polar(marked)
    _, _, phi_sub = polar(submesh)
    rho_sub = fd.Function(fd.FunctionSpace(submesh, "DG", 0)).interpolate(fd.cos(2 * phi_sub))
    rho_same = fd.Function(fd.FunctionSpace(marked, "DG", 0)).interpolate(
        fd.cos(2 * phi) * fd.conditional(r < 0.55, 1.0, 0.0))

    def solve(rho, **kwargs):
        psi = fd.Function(fd.FunctionSpace(marked, "CG", 2))
        solver = GravitySolver(psi, rho, bcs={1: {"dtn": CylindricalDtN(M=3)}}, **kwargs)
        solver.mixed_solution.assign(0)
        solver.solve()
        return psi, solver

    psi_same, solver_same = solve(rho_same)
    psi_sub, solver_sub = solve(rho_sub)
    out["same_mesh_cross_mesh"] = int(solver_same.cross_mesh)
    out["submesh_cross_mesh"] = int(solver_sub.cross_mesh)
    out["submesh_vs_same_mesh"] = relative_l2(psi_sub, psi_same, marked)

    psi_direct, _ = solve(rho_sub, solver_parameters="direct")
    psi_iterative, solver_iterative = solve(rho_sub, solver_parameters="iterative")
    out["iterative_offset"] = solver_iterative._multiplier_offset
    out["iterative_vs_direct"] = relative_l2(psi_iterative, psi_direct, marked)

    psi_m, solver_m = solve(rho_sub, dtn_representation="multiplier")
    psi_l, solver_l = solve(rho_sub, dtn_representation="lowrank")
    out["multiplier_fields"] = len(solver_m.mixed_space)
    out["multiplier_count"] = solver_m.n_multipliers
    out["multiplier_offset"] = solver_m._multiplier_offset
    out["lowrank_single_space"] = int(solver_l.mixed_space is solver_l.solution_space)
    out["lowrank_count"] = solver_l.n_multipliers
    out["lowrank_vs_multiplier_psi"] = relative_l2(psi_l, psi_m, marked)
    out["lowrank_vs_multiplier_coefficients"] = TG.max_coefficient_difference(
        solver_m.coefficients(), solver_l.coefficients())
    return out


# -- adjoint ------------------------------------------------------------------------

@contextmanager
def taping():
    """A fresh tape with annotation on inside the block, off afterwards."""
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()
    try:
        yield tape
    finally:
        if annotate_tape():
            pause_annotation()


def clear_tape():
    """Empty the tape and leave annotation off, between two rows."""
    if annotate_tape():
        pause_annotation()
    get_working_tape().clear_tape()


def guarded_taylor(out, prefix, rf, m, h, J):
    """The numbers of `test_gravity_adjoint.assert_taylor_with_guards`, unasserted.

    The four guards against a passing Taylor test that proves nothing:
    the directional derivative `h . dJ/dm` (`<prefix>_dJdm`), the response
    `|J(m + 0.01 h) - J(m)|` of the functional to the largest step
    (`<prefix>_response`), the Taylor rate with the gradient (`_rate`, 2 for
    a correct gradient) and without it (`_rate0`, 1 whatever the gradient),
    with `|J|` as `_J`. The replay is put back at `m` before and after.
    """
    absJ = abs(float(J))
    rf(m)
    dJdm = h._ad_dot(rf.derivative())
    with stop_annotating():
        m_pert = m._ad_add(h._ad_mul(0.01))
    J0 = float(rf(m))
    Jp = float(rf(m_pert))
    rf(m)
    out[f"{prefix}_J"] = absJ
    out[f"{prefix}_dJdm"] = float(dJdm)
    out[f"{prefix}_response"] = abs(Jp - J0)
    out[f"{prefix}_rate"] = taylor_test(rf, m, h)
    out[f"{prefix}_rate0"] = taylor_test(rf, m, h, dJdm=0.0)
    return float(dJdm)


def replay_rows(out, prefix, rf, m, J, forward=None, m2=None, args=()):
    """Replay at the recorded control (`b1`) and at a second one (`b2`).

    `<prefix>_recorded`, `_replay`: the recorded functional and the replay at
    `m`; `_fresh2`, `_replay2`: a fresh untaped solve at `m2` and the replay
    there (`test_gravity_adjoint.assert_replay_b1`, `assert_replay_b2`).
    """
    out[f"{prefix}_recorded"] = float(J)
    out[f"{prefix}_replay"] = float(rf(m))
    if forward is not None:
        with stop_annotating():
            J_fresh, _ = forward(m2, *args)
        out[f"{prefix}_fresh2"] = float(J_fresh)
        out[f"{prefix}_replay2"] = float(rf(m2))
        rf(m)


def taped(forward, m, *args):
    """Tape `forward(m, *args)` on a fresh tape; return `(J, solver, rf)`."""
    clear_tape()
    return TA.tape_forward(forward, m, *args)


def first_order_ladder(out, prefix, rf, m, h):
    """The R0 and R1 rates of `test_gravity_adjoint.taylor_first_order_ladder`.

    R0 (no gradient) converges at order 1, R1 (with the gradient) at order 2;
    the smallest of each over the four steps and the smallest R1 residual
    (above round-off) are written.
    """
    ladder = TA.taylor_first_order_ladder(rf, m, h)
    out[f"{prefix}_R0_rate"] = min(ladder["R0_rate"])
    out[f"{prefix}_R1_rate"] = min(ladder["R1_rate"])
    out[f"{prefix}_R1_min"] = min(ladder["R1"])


def rho_rows(out, prefix, forward, mesh, three_d=False):
    """Taylor on `rho` with replay `b1`, `b2` and the guards, for one forward map."""
    with stop_annotating():
        if three_d:
            m, h, m2 = TA.rho_control_3d(mesh), TA.rho_perturb_3d(mesh), TA.rho_control2_3d(mesh)
        else:
            m, h, m2 = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh), TA.rho_control2_2d(mesh)
    J, _, rf = taped(forward, m)
    replay_rows(out, prefix, rf, m, J, forward, m2)
    return J, rf, m, h


def no_multiplier_rows(out, forward, mesh):
    """`test_taylor_rho_no_multiplier`: plain Dirichlet and the `M = 0` Robin term."""
    for variant in ("dirichlet2d", "lu2d"):
        with stop_annotating():
            m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
            m2 = TA.rho_control2_2d(mesh)
        J, _, rf = taped(forward, m, variant)
        replay_rows(out, f"rho_{variant}", rf, m, J, forward, m2, (variant,))
        guarded_taylor(out, f"rho_{variant}", rf, m, h, J)


def sigma_rows(out, prefix, forward, mesh, three_d=False):
    """`test_taylor_sigma`: the sheet as a Function control, `rho = 0`."""
    with stop_annotating():
        V = fd.FunctionSpace(mesh, "CG", 1)
        X, _, phi = polar(mesh)
        if three_d:
            Y11 = real_spherical_harmonic(1, 1, X)
            Y10 = real_spherical_harmonic(1, 0, X)
            sigma = fd.Function(V).interpolate(Y11)
            h = fd.Function(V).interpolate(0.8 * Y11 + 0.2 * Y10)
            sigma2 = fd.Function(V).interpolate(Y11 + 0.5 * Y10)
        else:
            sigma = fd.Function(V).interpolate(fd.cos(2 * phi))
            h = fd.Function(V).interpolate(0.8 * fd.cos(2 * phi) + 0.2 * fd.cos(phi))
            sigma2 = fd.Function(V).interpolate(fd.cos(2 * phi) + 0.5 * fd.cos(3 * phi))
    geometry = "3d" if three_d else "2d"
    J, _, rf = taped(forward, sigma, geometry)
    replay_rows(out, prefix, rf, sigma, J, forward, sigma2, (geometry,))
    guarded_taylor(out, prefix, rf, sigma, h, J)


def G_rows(out, prefix, forward, mesh, three_d=False):
    """`test_gradient_G_constant`: `G` as a `Real` control, `dJ/dG = 2 J / G`.

    `psi` is linear in `G`, so `J = int psi^2` is quadratic in `G`: the
    derivative is `2 J / G` exactly, and a central difference is exact up to
    round-off. A gradient times 1.01 makes the first-order remainder
    `0.01 eps dJ`, and its Taylor rate falls to about 1 (`_sabotage`).
    """
    with stop_annotating():
        rho = TA.rho_control_3d(mesh) if three_d else TA.rho_control_2d(mesh)
        G = TA.real_scalar(mesh, 1.3)
        h = TA.real_scalar(mesh, 1.0)
    geometry = "3d" if three_d else "2d"
    J, _, rf = taped(forward, G, geometry, rho)
    J0, G0 = float(J), float(G)
    replay_rows(out, prefix, rf, G, J, forward, TA.real_scalar(mesh, 1.7), (geometry, rho))
    rf(G)
    dJdG = h._ad_dot(rf.derivative())
    out[f"{prefix}_dJdG"] = dJdG
    out[f"{prefix}_analytic"] = 2.0 * J0 / G0
    epsilon = 1e-3
    plus = float(rf(TA.real_scalar(mesh, G0 + epsilon)))
    minus = float(rf(TA.real_scalar(mesh, G0 - epsilon)))
    rf(G)
    out[f"{prefix}_central"] = (plus - minus) / (2 * epsilon)
    guarded_taylor(out, prefix, rf, G, h, J)
    out[f"{prefix}_sabotage"] = taylor_test(rf, G, h, dJdm=1.01 * dJdG)


def sabotage_rows(out, forward, mesh):
    """`test_taylor_detects_sabotaged_gradient`: a gradient wrong by 1 % is caught.

    Two corruptions: the directional derivative times 1.01, and 1 % of the
    gradient's Euclidean norm added along a second density that `h` sees.
    Both rates fall to about 1 and both first-order residuals at the
    smallest step are at least 10 times the clean one.
    """
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
        noise = TA.rho_control2_2d(mesh)
    _, _, rf = taped(forward, m)
    g = rf.derivative()
    true_value = h._ad_dot(g)
    R1_clean, out["sabotage_rate_clean"] = TA._residual_ladder(rf, m, h, true_value)
    R1_mul, out["sabotage_rate_multiplicative"] = TA._residual_ladder(
        rf, m, h, 1.01 * true_value)
    out["sabotage_overlap"] = fd.assemble(noise * h * fd.dx)
    corrupted = rf.derivative()
    scale = 0.01 * np.linalg.norm(g.dat.data_ro) / np.linalg.norm(noise.dat.data_ro)
    corrupted.dat.data[:] = g.dat.data_ro + scale * noise.dat.data_ro
    R1_noise, out["sabotage_rate_noise"] = TA._residual_ladder(rf, m, h, h._ad_dot(corrupted))
    out["sabotage_residual_multiplicative"] = R1_mul[-1] / R1_clean[-1]
    out["sabotage_residual_noise"] = R1_noise[-1] / R1_clean[-1]


def tangent_rows(out, forward, mesh, solve, native_forward=None, native_mesh=None,
                 native_args=()):
    """`test_tlm_adjoint_dot_product`: the adjoint is the transpose of the tangent.

    Native: pyadjoint's tangent sweep along `h` against `h . dJ/dm`
    (`tlm_adjoint`, `tlm_native`). Exact: `rho -> psi` is linear, so the
    tangent of `J = int psi^2` along `h` is `2 int psi(m) psi(h)`, computed
    from two independent untaped solves (`tlm_exact`, against `tlm_adjoint2`
    on the DtN configuration).
    """
    native_mesh = native_mesh or mesh
    native_forward = native_forward or forward
    with stop_annotating():
        m, h = TA.rho_control_2d(native_mesh), TA.rho_perturb_2d(native_mesh)
    J, _, rf = taped(native_forward, m, *native_args)
    out["tlm_adjoint"] = h._ad_dot(rf.derivative())
    m.block_variable.tlm_value = h
    get_working_tape().evaluate_tlm()
    out["tlm_native"] = float(J.block_variable.tlm_value)

    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
    _, _, rf = taped(forward, m)
    out["tlm_adjoint2"] = h._ad_dot(rf.derivative())
    with stop_annotating():
        psi_m = solve(m).solution.copy(deepcopy=True)
        psi_h = solve(h).solution.copy(deepcopy=True)
        out["tlm_exact"] = 2.0 * fd.assemble(psi_m * psi_h * fd.dx)


def replay_net_mass_rows(out, forward, mesh):
    """`test_reduced_functional_replay`: replay against fresh solves.

    At the recorded control, at a second zero-mean control, and at a control
    with net mass, whose enclosed mass is on the tape, so its replay agrees
    with a fresh solve.
    """
    with stop_annotating():
        m, m2 = TA.rho_control_2d(mesh), TA.rho_control2_2d(mesh)
        _, r, _ = polar(mesh)
        with_mass = fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(
            fd.exp(-(((r - 1.7) / 0.15) ** 2)))
    J, _, rf = taped(forward, m)
    replay_rows(out, "replay", rf, m, J, forward, m2)
    out["replay_mass"] = float(rf(with_mass))
    with stop_annotating():
        out["replay_mass_fresh"] = float(forward(with_mass)[0])


def assign_rows(out, mesh, representation):
    """`test_solution_assign_preserves_tape`: `solution` and `subfunctions[0]` agree.

    The functional on `solver.solution` (through the subfunction and assign
    blocks) and on `mixed_solution.subfunctions[0]` give the same value and
    gradient. On the multiplier path a raw `dat.data` copy of the same field
    is a negative control: it severs the tape, so its gradient is exactly
    zero.
    """
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
    clear_tape()
    with taping():
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, m,
                                   bcs={"top": {"dtn": CylindricalDtN(M=2)},
                                        "bottom": {"dtn": CylindricalDtN(M=2)}},
                                   dtn_representation=representation)
        solver.solve()
        J_sol = fd.assemble(solver.solution ** 2 * fd.dx)
        J_mix = fd.assemble(solver.mixed_solution.subfunctions[0] ** 2 * fd.dx)
        if representation == "multiplier":
            with stop_annotating():
                raw = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
                raw.dat.data[:] = solver.mixed_solution.subfunctions[0].dat.data_ro
            J_raw = fd.assemble(raw ** 2 * fd.dx)
    rf_sol = ReducedFunctional(J_sol, Control(m))
    rf_mix = ReducedFunctional(J_mix, Control(m))
    out["assign_J_solution"] = float(J_sol)
    out["assign_J_mixed"] = float(J_mix)
    out["assign_replay_solution"] = float(rf_sol(m))
    out["assign_replay_mixed"] = float(rf_mix(m))
    g_sol, g_mix = rf_sol.derivative(), rf_mix.derivative()
    denominator = np.max(np.abs(g_mix.dat.data_ro))
    out["assign_gradient_difference"] = np.max(
        np.abs(g_sol.dat.data_ro - g_mix.dat.data_ro)) / denominator
    if representation == "multiplier":
        g_raw = ReducedFunctional(J_raw, Control(m)).derivative()
        out["assign_raw_gradient_max"] = np.max(np.abs(g_raw.dat.data_ro))
        out["assign_raw_difference"] = np.max(
            np.abs(g_raw.dat.data_ro - g_mix.dat.data_ro)) / denominator
        guarded_taylor(out, "assign", rf_sol, m, h, J_sol)


def submesh_adjoint_rows(out, forward):
    """`test_taylor_rho_submesh`: the cross-mesh path, differentiated end to end.

    The density is DG0 on the submesh `r < 0.55` of `UnitDiskMesh(2)`, the
    potential CG1 on the parent, with one exterior DtN boundary, so the
    densities are zero-mean in azimuth.
    """
    with stop_annotating():
        marked, submesh = disc_submesh(2)
        m = TA._rho_sub(submesh, [(2, 1.0)])
        h = TA._rho_sub(submesh, [(2, 0.8), (1, 0.2)])
        m2 = TA._rho_sub(submesh, [(2, 1.0), (3, 0.5)])
    J, solver, rf = taped(forward, m, marked)
    out["submesh_single_space"] = int(solver.mixed_space is solver.solution_space)
    replay_rows(out, "submesh", rf, m, J, forward, m2, (marked,))
    out["submesh_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))
    guarded_taylor(out, "submesh", rf, m, h, J)


def sheet_closed_form_rows(out, representation):
    """The gradient of a trace coefficient against `dc/ds = 2 pi G a / m`.

    `test_gradient_matches_closed_form_sheet_2d`: a sheet `s cos(2 phi)` on
    the outer boundary `a = RMAX`, `G = 1`, CG2 on the unit annulus. The
    trace coefficient `c = int psi cos(m phi) ds / (pi R)` is linear in `s`,
    so its derivative is the sheet factor `2 pi a / m`; the nearest wrong
    closed form, the factor of mode `m + 1`, is written as `_wrong`.
    `J = c^2` gets the guarded Taylor test on the same tape.
    """
    m_mode = 2
    mesh = TG.annulus_mesh(n_azimuthal=192, dr=0.1)
    with stop_annotating():
        s = TA.real_scalar(mesh, 1.0)
        hs = TA.real_scalar(mesh, 1.0)
    clear_tape()
    with taping():
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
            _, _, phi = polar(mesh)
            solver = GravitySolver(
                psi, 0.0,
                bcs={"top": {"dtn": CylindricalDtN(M=4), "sigma": s * fd.cos(m_mode * phi)},
                     "bottom": {"dtn": CylindricalDtN(M=4)}},
                dtn_representation=representation)
        solver.solve()
        _, R = solver.boundary_geometry["top"]
        c = fd.assemble(solver.mixed_solution.subfunctions[0] * fd.cos(m_mode * phi)
                        * solver.ds("top")) / (np.pi * R)
        J = c ** 2
    out["sheet_gradient"] = hs._ad_dot(ReducedFunctional(c, Control(s)).derivative())
    out["sheet_gradient_analytic"] = 2 * np.pi * RMAX / m_mode
    out["sheet_gradient_wrong"] = 2 * np.pi * RMAX / (m_mode + 1)
    guarded_taylor(out, "sheet", ReducedFunctional(J, Control(s)), s, hs, J)


def sphere_sheet_closed_form_rows(out, representation):
    """The gradient of a trace coefficient against `dc/ds = 4 pi G a / (2l + 1)`.

    `test_gradient_matches_closed_form_sheet_3d`: a sheet `s Y_21` on the
    outer boundary of the unit shell (level 2, `dr = 0.15`), CG2, `G = 1`.
    The coefficient is `int psi Y_21 ds / R^2` on the multiplier path and the
    taped `coefficients()` value on the low-rank path, as in the two unit
    tests.
    """
    l_mode, m_order = 2, 1
    mesh = TG.shell_mesh_3d(refinement_level=2, dr=0.15)
    with stop_annotating():
        s = TA.real_scalar(mesh, 1.0)
        hs = TA.real_scalar(mesh, 1.0)
        Y = real_spherical_harmonic(l_mode, m_order, fd.SpatialCoordinate(mesh))
    clear_tape()
    with taping():
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
            extra = ({"solver_parameters": "iterative"} if representation == "lowrank"
                     else {})
            solver = GravitySolver(psi, 0.0,
                                   bcs={"top": {"dtn": SphericalDtN(L=3), "sigma": s * Y},
                                        "bottom": {"dtn": SphericalDtN(L=3)}},
                                   dtn_representation=representation, **extra)
        solver.solve()
        if representation == "lowrank":
            c = solver.coefficients()["top"][f"Y{l_mode},{m_order}"]
        else:
            _, R = solver.boundary_geometry["top"]
            Yc = real_spherical_harmonic(l_mode, m_order, solver.X)
            c = fd.assemble(solver.mixed_solution.subfunctions[0] * Yc
                            * solver.ds("top")) / R ** 2
        J = c ** 2
    out["sphere_sheet_gradient"] = hs._ad_dot(ReducedFunctional(c, Control(s)).derivative())
    out["sphere_sheet_gradient_analytic"] = 4 * np.pi * RMAX / (2 * l_mode + 1)
    guarded_taylor(out, "sphere_sheet", ReducedFunctional(J, Control(s)), s, hs, J)


def monopole_fresh_rows(out, representation):
    """The monopole channel against a central difference of fresh solves.

    `test_monopole_gradient_against_fresh_solves`: on `UnitDiskMesh(2)` with
    one DtN boundary, the 2-D monopole datum carries `rho` and `sigma`
    through the enclosed mass and `G` through its own prefactor. A Taylor
    test replays the tape, so a mass channel severed from the tape would
    pass it; a central difference (`eps = 1e-5`) of solves of new solvers
    cannot see the tape. Per control: `_adjoint` and `_central`.
    """
    mesh = fd.UnitDiskMesh(refinement_level=2)
    V = fd.FunctionSpace(mesh, "CG", 2)
    Q = fd.FunctionSpace(mesh, "CG", 1)
    X = fd.SpatialCoordinate(mesh)

    def build(rho_value, sigma_value, g_value):
        psi = fd.Function(V)
        rho = fd.Function(Q).assign(rho_value)
        sigma = fd.Function(Q).assign(sigma_value)
        gravity = TA.real_scalar(mesh, g_value)
        solver = GravitySolver(psi, rho, bcs={1: {"dtn": CylindricalDtN(M=3), "sigma": sigma}},
                               gravitational_constant=gravity, solver_parameters="direct",
                               dtn_representation=representation)
        return solver, psi, {"rho": rho, "sigma": sigma, "G": gravity}

    with stop_annotating():
        rho0 = fd.Function(Q).interpolate(1.0 + 0.3 * X[0])
        sigma0 = fd.Function(Q).interpolate(0.4 + 0.2 * X[1])
        direction = fd.Function(Q).interpolate(0.05 * (1.0 + X[1] ** 2))
        g_direction = TA.real_scalar(mesh, 0.1)

    for control in GRAVITY_MONOPOLE_CONTROLS:
        clear_tape()
        with taping():
            solver, psi, controls = build(rho0, sigma0, 1.0)
            solver.solve()
            J = fd.assemble(psi ** 2 * fd.dx)
        gradient = ReducedFunctional(J, Control(controls[control])).derivative()
        step = direction if control != "G" else g_direction
        out[f"monopole_{control}_adjoint"] = float(fd.assemble(fd.action(gradient, step)))
        clear_tape()

        def objective_at(shift):
            with stop_annotating():
                fresh, fresh_psi, _ = build(
                    rho0 + shift * direction if control == "rho" else rho0,
                    sigma0 + shift * direction if control == "sigma" else sigma0,
                    1.0 + shift * 0.1 if control == "G" else 1.0)
                fresh.solve()
                return float(fd.assemble(fresh_psi ** 2 * fd.dx))

        epsilon = 1e-5
        out[f"monopole_{control}_central"] = (
            objective_at(epsilon) - objective_at(-epsilon)) / (2 * epsilon)


def tape_structure_rows(out, mesh):
    """`test_tape_structure`: one annotated multiplier solve, block by block.

    The solve records one solve block on the mixed space and one on the
    `Real` space of the enclosed mass (the 2-D monopole datum), no assemble
    block (the net-mass check runs off the tape), two assign blocks (the
    reset of the mass and the copy of the solution) and at least one
    subfunction block; the mass solve, the mixed solve and the objective
    reach `J`. In the `dirty` variant the solver is built under annotation:
    the assemble blocks of the construction exist and none reaches `J`.
    Each flag is 1 or 0.
    """
    for variant in ("clean", "dirty"):
        clear_tape()
        tape = get_working_tape()
        with stop_annotating():
            m = TA.rho_control_2d(mesh)
        if variant == "dirty":
            continue_annotation()
        with (nullcontext() if variant == "dirty" else stop_annotating()):
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, m, bcs={"top": {"dtn": CylindricalDtN(M=2)},
                                                "bottom": {"dtn": CylindricalDtN(M=2)}})
        if variant == "clean":
            continue_annotation()
        n_before = len(tape.get_blocks())
        solver.solve()
        n_after = len(tape.get_blocks())
        blocks = tape.get_blocks()[n_before:n_after]
        J = fd.assemble(solver.solution ** 2 * fd.dx)
        pause_annotation()
        counts = Counter(type(b).__name__ for b in blocks)
        mass = TA._real_space_solves(blocks)
        mixed = TA._mixed_space_solves(blocks)
        objective = [b for b in tape.get_blocks()[n_after:]
                     if type(b).__name__ == "AssembleBlock"]
        p = f"tape_{variant}"
        out[f"{p}_solve_blocks"] = counts[TA.SOLVE_BLOCK_TYPE.__name__]
        out[f"{p}_mixed_solves"] = len(mixed)
        out[f"{p}_mass_solves"] = len(mass)
        out[f"{p}_mass_reaches"] = int(len(mass) == 1 and TA._reaches_objective(tape, mass[0], J))
        out[f"{p}_mixed_reaches"] = int(len(mixed) == 1
                                        and TA._reaches_objective(tape, mixed[0], J))
        out[f"{p}_solve_assembles"] = counts["AssembleBlock"]
        out[f"{p}_subfunction_blocks"] = counts["SubfunctionBlock"]
        out[f"{p}_assign_blocks"] = counts["FunctionAssignBlock"]
        out[f"{p}_objective_assembles"] = len(objective)
        out[f"{p}_objective_reaches"] = int(len(objective) == 1
                                            and TA._reaches_objective(tape, objective[0], J))
        out[f"{p}_J"] = float(J)
        if variant == "dirty":
            pre = [b for b in tape.get_blocks()[:n_before] if type(b).__name__ == "AssembleBlock"]
            out[f"{p}_construction_assembles"] = len(pre)
            out[f"{p}_construction_reaching"] = sum(
                int(TA._reaches_objective(tape, b, J)) for b in pre)
    clear_tape()


def float_severs_rows(out, mesh):
    """`test_coefficient_float_severs_tape`: `coefficients()` reads `float()`.

    On the multiplier path `coefficients()` reads the `Real` subfunction
    through `float()`, which the tape does not record: the gradient of a
    functional built from it is exactly zero, and its replay is the same
    number at every control. Written: the largest gradient entry and the
    number of four perturbed replays that return the recorded value.
    """
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
    clear_tape()
    with taping():
        solver = TA.solve_2d_mm(m)
        J = AdjFloat(solver.coefficients()["top"]["cos2"])
    rf = ReducedFunctional(J, Control(m))
    out["severed_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))
    J0 = float(rf(m))
    constant = 0
    for eps in [0.01 / 2 ** i for i in range(4)]:
        with stop_annotating():
            m_pert = m._ad_add(h._ad_mul(eps))
        constant += int(float(rf(m_pert)) == J0)
    out["severed_constant_replays"] = constant


def coefficient_integral_rows(out, mesh):
    """`test_coefficient_taped_integral_taylor` (multiplier path).

    The trace coefficient as a taped boundary integral, `c = int psi cos(2
    phi) ds / (pi R)` (`J_f = c^2`), restores the tape; it equals the value
    read off the multiplier subfunction (`J_c`) on the solution manifold, and
    the two gradients agree.
    """
    with stop_annotating():
        m, h, m2 = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh), TA.rho_control2_2d(mesh)
    clear_tape()
    with taping():
        solver = TA.solve_2d_mm(m)
        J_f = TA.trace_coeff_integral(solver, "top", 2) ** 2
        J_c = TA.trace_coeff_rdirect(solver, "top", "cos2") ** 2
    rf_f = ReducedFunctional(J_f, Control(m))
    rf_c = ReducedFunctional(J_c, Control(m))
    out["integral_recorded"] = float(J_f)
    out["integral_replay"] = float(rf_f(m))
    out["integral_read"] = float(J_c)
    with stop_annotating():
        out["integral_fresh2"] = float(
            TA.trace_coeff_integral(TA.solve_2d_mm(m2), "top", 2)) ** 2
    out["integral_replay2"] = float(rf_f(m2))
    rf_f(m)
    guarded_taylor(out, "integral", rf_f, m, h, J_f)
    g_f, g_c = rf_f.derivative(), rf_c.derivative()
    out["integral_gradient_difference"] = np.max(
        np.abs(g_f.dat.data_ro - g_c.dat.data_ro)) / np.max(np.abs(g_f.dat.data_ro))


def multiplier_read_rows(out, mesh):
    """`test_taylor_through_multiplier`: a functional read off a `Real` multiplier.

    `J = c^2` with `c` the integral of the `Real` subfunction: replay, the
    guarded Taylor test, agreement with the taped-integral gradient, and the
    blindness of the `cos 2` coefficient to a pure `cos 3` direction
    (`_orthogonal` against `_active`), which is why `h` must contain the
    active mode.
    """
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
        _, r, phi = polar(mesh)
        h_orth = fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(
            fd.cos(3 * phi) * fd.exp(-(((r - 1.7) / 0.15) ** 2)))
    clear_tape()
    with taping():
        solver = TA.solve_2d_mm(m)
        J_c = TA.trace_coeff_rdirect(solver, "top", "cos2") ** 2
        J_i = TA.trace_coeff_integral(solver, "top", 2) ** 2
    rf = ReducedFunctional(J_c, Control(m))
    rf_i = ReducedFunctional(J_i, Control(m))
    replay_rows(out, "read", rf, m, J_c)
    guarded_taylor(out, "read", rf, m, h, J_c)
    g, g_i = rf.derivative(), rf_i.derivative()
    out["read_gradient_difference"] = np.max(
        np.abs(g.dat.data_ro - g_i.dat.data_ro)) / np.max(np.abs(g.dat.data_ro))
    out["read_active"] = h._ad_dot(g)
    out["read_orthogonal"] = h_orth._ad_dot(g)


def canary_rows(out):
    """`test_firedrake_constant_is_not_differentiable_canary` (no gravity).

    A Firedrake `Constant` is not a differentiable control in this Firedrake:
    its gradient is exactly zero (the true value is `2 J / G = 41.6`) and a
    replay at a new `Constant` returns the recorded value. Written so that
    the day it changes is visible: the gradient and 1 if the replay is
    unchanged.
    """
    mesh = fd.UnitSquareMesh(4, 4)
    V = fd.FunctionSpace(mesh, "CG", 1)
    with stop_annotating():
        u = fd.Function(V).interpolate(fd.Constant(2.0))
        G = fd.Constant(1.3)
    clear_tape()
    with taping():
        J = fd.assemble((G * u * u) ** 2 * fd.dx)
    rf = ReducedFunctional(J, Control(G))
    out["canary_gradient"] = float(rf.derivative())
    out["canary_replay_unchanged"] = int(float(rf(fd.Constant(1.7))) == float(J))


def wall_rows(out):
    """`M = 35` on both boundaries: 142 fields, past PETSc's 128 (multiplier).

    `TestIndexSetSplit::test_solves_past_the_field_enumeration_wall`: the
    index-set split solves; the coefficients of modes up to 5 do not move
    from the `M = 5` solve (the annulus is symmetric, so the modes decouple);
    the field-number split raises a PETSc error at this size.
    `test_taylor_rho_past_field_enumeration_wall`: the derivative exists
    there (replay and guarded Taylor). Both rows are in this one process,
    because they share the same form, whose compilation costs about 40 s.
    """
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.5)
    rho = TG.annulus_source(mesh)
    _, solver_low = TG.solve_annulus(mesh, rho, 5, "iterative")
    _, solver_high = TG.solve_annulus(mesh, rho, 35, "iterative")
    out["wall_fields"] = len(solver_high.mixed_space)
    out["wall_multipliers"] = solver_high.n_multipliers
    out["wall_converged"] = solver_high.solver.snes.ksp.getConvergedReason()
    out["wall_scale"] = max(abs(value) for boundary in solver_low.coefficients().values()
                            for value in boundary.values())
    out["wall_low_mode_change"] = TG.max_coefficient_difference(
        solver_low.coefficients(), solver_high.coefficients())
    try:
        TG.solve_annulus(mesh, rho, 35, TG.field_number_parameters(
            "iterative", solver_high._multiplier_offset, solver_high.n_multipliers))
        out["wall_field_numbers_refused"] = 0
    except PETSc.Error:
        out["wall_field_numbers_refused"] = 1

    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
    J, solver, rf = taped(TA.forward_2d_past_wall, m)
    out["wall_taped_fields"] = len(solver.mixed_space)
    replay_rows(out, "wall", rf, m, J)
    guarded_taylor(out, "wall", rf, m, h, J)
    out["wall_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))


def lowrank_wall_rows(out):
    """`test_taylor_at_a_truncation_the_multiplier_path_cannot_reach` (low-rank).

    `M = 35` gives 141 modes and no field at all on the low-rank path; the
    derivative exists there (replay and guarded Taylor).
    """
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.5)
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)

    def forward(m_rho):
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, m_rho,
                                   bcs={"top": {"dtn": CylindricalDtN(M=35)},
                                        "bottom": {"dtn": CylindricalDtN(M=35)}},
                                   solver_parameters="iterative", dtn_representation="lowrank")
        solver.solve()
        return fd.assemble(solver.solution ** 2 * fd.dx), solver

    J, solver, rf = taped(forward, m)
    out["wall_modes"] = len(solver._multiplier_keys)
    out["wall_multipliers"] = solver.n_multipliers
    replay_rows(out, "wall", rf, m, J)
    guarded_taylor(out, "wall", rf, m, h, J)
    out["wall_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))


def taped_coefficients_rows(out, mesh):
    """`test_coefficient_taped_integral_taylor` (low-rank): `coefficients()` is taped.

    On the low-rank path `c_k = u_k . psi / (scale_k A_h)` is a taped linear
    functional of `psi`. It equals the taped boundary integral `int psi cos(2
    phi) ds / (pi R)`, and the two gradients agree; an untaped
    `coefficients()` would give an exactly zero gradient.
    """
    with stop_annotating():
        m, h = TA.rho_control_2d(mesh), TA.rho_perturb_2d(mesh)
    clear_tape()
    with taping():
        with stop_annotating():
            psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
            solver = GravitySolver(psi, m, bcs={"top": {"dtn": CylindricalDtN(M=2)},
                                                "bottom": {"dtn": CylindricalDtN(M=2)}},
                                   dtn_representation="lowrank")
        solver.solve()
        c_recovered = solver.coefficients()["top"]["cos2"]
        _, _, phi = polar(mesh)
        _, R = solver.boundary_geometry["top"]
        c_integral = fd.assemble(
            solver.mixed_solution * fd.cos(2 * phi) * solver.ds("top")) / (np.pi * R)
        J_recovered = c_recovered ** 2
        J_integral = c_integral ** 2
    out["coefficients_recovered"] = float(c_recovered)
    out["coefficients_integral"] = float(c_integral)
    rf_recovered = ReducedFunctional(J_recovered, Control(m))
    rf_integral = ReducedFunctional(J_integral, Control(m))
    g_recovered, g_integral = rf_recovered.derivative(), rf_integral.derivative()
    out["coefficients_gradient_max"] = np.max(np.abs(g_recovered.dat.data_ro))
    out["coefficients_integral_gradient_max"] = np.max(np.abs(g_integral.dat.data_ro))
    out["coefficients_gradient_difference"] = np.max(
        np.abs(g_recovered.dat.data_ro - g_integral.dat.data_ro)) / np.max(
            np.abs(g_integral.dat.data_ro))
    replay_rows(out, "coefficients", rf_recovered, m, J_recovered)
    guarded_taylor(out, "coefficients", rf_recovered, m, h, J_recovered)


def l0_corner_rows(out, mesh, representation):
    """The `L = 0` exterior corner of the 3-D shell (`test_taylor_monopole_constraint_row_3d`).

    With `alpha = 1` and `lam_0 = 1 / R` the weight `lam_0 - alpha / R` of
    the monopole is zero. On the multiplier path `J = c00^2` is read off the
    multiplier (a constraint row with a zero feedback row); on the low-rank
    path the monopole adds nothing to `B` and `J = (int psi)^2`. Replay and
    guarded Taylor; on the low-rank path the weight is written too.
    """
    with stop_annotating():
        m = TA._rho_radial_3d(mesh, 1.7, 0.15)
        h = TA._rho_radial_3d(mesh, 1.6, 0.2)
    if representation == "multiplier":
        forward = TA.forward_3d_l0
    else:
        def forward(m_rho):
            with stop_annotating():
                psi = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
                solver = GravitySolver(psi, m_rho, bcs={"top": {"dtn": SphericalDtN(L=0)}},
                                       dtn_representation="lowrank")
            solver.solve()
            out["l0_weight"] = float(solver.mode_rows[0].weights[0])
            return fd.assemble(solver.solution * fd.dx) ** 2, solver
    J, _, rf = taped(forward, m)
    replay_rows(out, "l0", rf, m, J)
    guarded_taylor(out, "l0", rf, m, h, J)


def run_adjoint_2d_multiplier():
    """The 2-D rows of `test_gravity_adjoint.py` (multiplier representation)."""
    out = {"ranks": fd.COMM_WORLD.size}
    mesh_lu = TG.annulus_mesh(n_azimuthal=64, dr=0.25)
    mesh_mm = TG.annulus_mesh(n_azimuthal=96, dr=0.2)
    tape_structure_rows(out, mesh_mm)
    no_multiplier_rows(out, TA.forward_2d_lu, mesh_lu)
    J, rf, m, h = rho_rows(out, "rho_2d", TA.forward_2d_mm, mesh_mm)
    first_order_ladder(out, "rho_2d", rf, m, h)
    guarded_taylor(out, "rho_2d", rf, m, h, J)
    out["rho_2d_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))
    float_severs_rows(out, mesh_mm)
    coefficient_integral_rows(out, mesh_mm)
    multiplier_read_rows(out, mesh_mm)
    assign_rows(out, mesh_mm, "multiplier")
    tangent_rows(out, TA.forward_2d_mm, mesh_mm, TA.solve_2d_mm,
                 native_forward=TA.forward_2d_lu, native_mesh=mesh_lu, native_args=("lu2d",))
    G_rows(out, "G_2d", TA.forward_G, mesh_mm)
    canary_rows(out)
    sigma_rows(out, "sigma_2d", TA.forward_sheet, mesh_mm)
    replay_net_mass_rows(out, TA.forward_2d_mm, mesh_mm)
    monopole_fresh_rows(out, "multiplier")
    sabotage_rows(out, TA.forward_2d_mm, mesh_mm)
    submesh_adjoint_rows(out, TA.forward_submesh)
    sheet_closed_form_rows(out, "multiplier")
    wall_rows(out)
    clear_tape()
    return out


def run_adjoint_2d_lowrank():
    """The 2-D rows of `test_gravity_adjoint_lowrank.py` (low-rank representation)."""
    out = {"ranks": fd.COMM_WORLD.size}
    mesh = TG.annulus_mesh(n_azimuthal=96, dr=0.2)
    no_multiplier_rows(out, TAL.forward_no_multiplier, mesh)
    J, rf, m, h = rho_rows(out, "rho_2d", TAL.forward_rho_2d, mesh)
    first_order_ladder(out, "rho_2d", rf, m, h)
    guarded_taylor(out, "rho_2d", rf, m, h, J)
    out["rho_2d_gradient_max"] = np.max(np.abs(rf.derivative().dat.data_ro))
    sigma_rows(out, "sigma_2d", TAL.forward_sheet, mesh)
    G_rows(out, "G_2d", TAL.forward_G, mesh)
    tangent_rows(out, TAL.forward_rho_2d, mesh, lambda rho: TAL.forward_rho_2d(rho)[1])
    sabotage_rows(out, TAL.forward_rho_2d, mesh)
    submesh_adjoint_rows(out, TAL.forward_submesh)
    replay_net_mass_rows(out, TAL.forward_rho_2d, mesh)
    assign_rows(out, mesh, "lowrank")
    sheet_closed_form_rows(out, "lowrank")
    lowrank_wall_rows(out)
    taped_coefficients_rows(out, mesh)
    monopole_fresh_rows(out, "lowrank")
    clear_tape()
    return out


def run_adjoint_3d(representation):
    """The 3-D rows of the two adjoint unit files, on one representation."""
    out = {"ranks": fd.COMM_WORLD.size}
    mesh = TG.shell_mesh_3d(refinement_level=1, dr=0.5)
    if representation == "multiplier":
        forward_rho, forward_sheet, forward_G = TA.forward_3d_mm, TA.forward_sheet, TA.forward_G
    else:
        forward_rho, forward_sheet, forward_G = (TAL.forward_rho_3d, TAL.forward_sheet,
                                                 TAL.forward_G)
    J, rf, m, h = rho_rows(out, "rho_3d", forward_rho, mesh, three_d=True)
    guarded_taylor(out, "rho_3d", rf, m, h, J)
    sigma_rows(out, "sigma_3d", forward_sheet, mesh, three_d=True)
    G_rows(out, "G_3d", forward_G, mesh, three_d=True)
    l0_corner_rows(out, mesh, representation)
    sphere_sheet_closed_form_rows(out, representation)
    clear_tape()
    return out


# -- quadrature ---------------------------------------------------------------------

def quadrature_warned(caught):
    """1 if a recorded warning is about the boundary quadrature, else 0.

    Only these count: the warped shells also warn about the deviation of the
    boundary radius, which says nothing about the quadrature degree
    (`test_silence_implies_adequacy`).
    """
    return int(any("boundary quadrature" in str(w.message) for w in caught))


def built(mesh, descriptor, element_degree=1, **kwargs):
    """`test_gravity_quad_degree.build` with its warnings recorded.

    Returns:
      The solver and 1 or 0 for a quadrature warning at construction.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        solver = TQ.build(mesh, descriptor, element_degree, **kwargs)
    return solver, quadrature_warned(caught)


def report_rows(out, prefix, solver, warned):
    """The chosen degree, the warning flag and the numbers of the rule's report."""
    out[f"{prefix}_degree"] = solver.quad_degree
    out[f"{prefix}_warned"] = warned
    report = solver.quad_rule_report
    if report is not None:
        out[f"{prefix}_requested"] = report.requested
        out[f"{prefix}_incumbent"] = report.incumbent
        out[f"{prefix}_distortion"] = report.distortion
        out[f"{prefix}_resolution"] = report.resolution
        out[f"{prefix}_max_distortion"] = report.calibration.max_distortion
        out[f"{prefix}_max_resolution"] = report.calibration.max_resolution


def check(solver, **kwargs):
    """`check_boundary_quadrature(sample="all")`, which warns instead of raising.

    The number is the largest relative change of the mode norms between the
    chosen degree and a richer one, over every treated mode: the integration
    error of the worst mode.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return solver.check_boundary_quadrature(sample="all", action="warn", **kwargs)


def run_quadrature_sweep(warp, L):
    """One mesh of `test_silence_implies_adequacy`: a warped level-2 shell.

    The constructor either warns about the boundary quadrature, or the degree
    it chose integrates every mode to the calibration target. Written: the
    degree and the report, the warning flag and the integration error.
    """
    solver, warned = built(TQ.shell(2, warp=warp), SphericalDtN(L=L))
    out = {"ranks": fd.COMM_WORLD.size, "warp": warp, "L": L}
    report_rows(out, "sweep", solver, warned)
    out["sweep_error"] = check(solver)
    return out


def rank_rows(out):
    """`test_degree_and_check_are_rank_independent`: the degree and the check.

    `h_max` is a maximum over owned facets and the mode norms are sums over
    owned entries; both are reductions, so the degree (9 in 2-D, 12 in 3-D)
    and the check must be the same on every rank count.
    """
    two_d, _ = built(TQ.annulus(96), CylindricalDtN(M=16))
    out["rank_2d_degree"] = two_d.quad_degree
    out["rank_2d_error"] = check(two_d)
    three_d, _ = built(TQ.shell(2), SphericalDtN(L=5))
    out["rank_3d_degree"] = three_d.quad_degree
    out["rank_3d_error"] = check(three_d)


def run_quadrature_rule():
    """The rows of `test_gravity_quad_degree.py` other than the warped sweep.

    Each group is named after its unit test:

    - `follows`: one truncation, three boundary resolutions; the degree
      falls as the mesh resolves the mode (`test_follows_the_mesh_not_the_truncation`);
    - `odd_<k>`: the degree is odd (`test_odd_because_odd_is_free`);
    - `cost_<k>`: never above the incumbent `2 (L + p)`
      (`test_never_costs_more_than_the_default_it_replaces`);
    - `extrapolating_warned`, `distortion_*`: the two free warnings
      (`test_an_extrapolating_mesh_is_flagged`,
      `test_distortion_alone_triggers_the_warning`);
    - `silent_warned`: no warning on three ordinary configurations
      (`test_a_well_covered_mesh_is_silent`);
    - `element_<k>_<p>`: the degree for CG1, CG2, CG3
      (`test_element_degree_costs_two_each`);
    - `rank_*`: `rank_rows`;
    - `triangle_*`: the triangular constants are distinct and taller; the
      icosahedral shell takes degree 15 and integrates
      (`test_triangular_facets_carry_their_own_constants`);
    - `adequate_<k>`: the chosen degree integrates every mode
      (`TestDefaultIsAdequate::test_every_mode_is_integrated`);
    - `instrument_*`: the check measures quadrature and not sphericity
      (`TestCheckIsAnInstrument`, five tests).
    """
    out = {"ranks": fd.COMM_WORLD.size}
    for k, n in enumerate((96, 192, 384)):
        out[f"follows_degree_{k}"] = built(TQ.annulus(n), CylindricalDtN(M=16))[0].quad_degree
    out["follows_incumbent"] = TQ.incumbent(CylindricalDtN(M=16))

    for k, (n, M_trunc) in enumerate(((96, 8), (192, 32), (384, 16))):
        out[f"odd_degree_{k}"] = built(TQ.annulus(n), CylindricalDtN(M=M_trunc))[0].quad_degree

    configurations = [(lambda: TQ.annulus(96), CylindricalDtN(M=2), 1),
                      (lambda: TQ.annulus(96), CylindricalDtN(M=32), 2),
                      (lambda: TQ.annulus(384), CylindricalDtN(M=16), 1),
                      (lambda: TQ.shell(1), SphericalDtN(L=2), 1),
                      (lambda: TQ.shell(2), SphericalDtN(L=5), 1)]
    for k, (mesh_of, descriptor, p) in enumerate(configurations):
        out[f"cost_degree_{k}"] = built(mesh_of(), descriptor, p)[0].quad_degree
        out[f"cost_incumbent_{k}"] = TQ.incumbent(descriptor, p)

    psi = fd.Function(fd.FunctionSpace(TQ.annulus(96, irregular=True), "CG", 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        GravitySolver(psi, 0.0, bcs={"top": {"dtn": CylindricalDtN(M=64)}},
                      solver_parameters="direct")
    out["extrapolating_warned"] = int(any("extrapolating" in str(w.message) for w in caught))

    mesh = TQ.shell(2, warp=0.30)
    solver, _ = built(mesh, SphericalDtN(L=8))
    report = solver.quad_rule_report
    out["distortion_value"] = report.distortion
    out["distortion_max"] = report.calibration.max_distortion
    out["distortion_resolution"] = report.resolution
    out["distortion_max_resolution"] = report.calibration.max_resolution
    out["distortion_same_rule"] = int((report.requested + 2) // 2
                                      == (report.incumbent + 2) // 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        TQ.build(mesh, SphericalDtN(L=8))
    out["distortion_warned"] = int(any("extrapolating" in str(w.message) for w in caught))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        TQ.build(TQ.annulus(192), CylindricalDtN(M=16))
        TQ.build(TQ.shell(2), SphericalDtN(L=5))
        TQ.build(TQ.shell(3), SphericalDtN(L=10))
    out["silent_warned"] = len([w for w in caught if issubclass(w.category, UserWarning)])

    for k, (mesh_of, descriptor_of) in enumerate(
            ((lambda: TQ.annulus(192), lambda: CylindricalDtN(M=16)),
             (lambda: TQ.shell(2), lambda: SphericalDtN(L=5)))):
        mesh = mesh_of()
        for p in (1, 2, 3):
            out[f"element_{k}_{p}"] = built(mesh, descriptor_of(), p)[0].quad_degree
    out["element_cost"] = QUADRATURE_ELEMENT_DEGREE_COST

    rank_rows(out)

    triangle = QUADRATURE_RULE_COEFFICIENTS["triangle"]
    quadrilateral = QUADRATURE_RULE_COEFFICIENTS["quadrilateral"]
    out["triangle_taller"] = int(triangle.intercept > quadrilateral.intercept)
    out["triangle_steeper"] = int(triangle.slope > quadrilateral.slope)
    descriptor = SphericalDtN(L=10)
    solver, _ = built(TQ.icosahedral_shell(2), descriptor)
    out["triangle_calibration"] = int(solver.quad_rule_report.calibration == triangle)
    out["triangle_degree"] = solver.quad_degree
    out["triangle_incumbent"] = TQ.incumbent(descriptor)
    out["triangle_error"] = check(solver)

    adequate = [(lambda: TQ.annulus(96), lambda: CylindricalDtN(M=8)),
                (lambda: TQ.annulus(96, irregular=True), lambda: CylindricalDtN(M=16)),
                (lambda: TQ.annulus(192, irregular=True), lambda: CylindricalDtN(M=32)),
                (lambda: TQ.shell(2), lambda: SphericalDtN(L=5)),
                (lambda: TQ.shell(2, warp=0.10), lambda: SphericalDtN(L=5)),
                (lambda: TQ.shell(3, warp=0.20), lambda: SphericalDtN(L=5))]
    for k, (mesh_of, descriptor_of) in enumerate(adequate):
        out[f"adequate_error_{k}"] = check(built(mesh_of(), descriptor_of())[0])

    mesh = TQ.shell(2)
    for k, q in enumerate((5, 9, 13, 21)):
        out[f"instrument_sweep_{k}"] = check(built(mesh, SphericalDtN(L=8), quad_degree=q)[0])
    out["instrument_rich_rule"] = check(built(mesh, SphericalDtN(L=5), quad_degree=25)[0])
    solver, _ = built(TQ.annulus(96), CylindricalDtN(M=6))
    bc_id, descriptor = solver.dtn_boundaries[0]
    side, radius = solver.boundary_geometry[bc_id]
    keys = [mode.key for mode in descriptor.mode_metadata(side, radius)]
    recovered = solver._mode_norms(bc_id, descriptor, side, radius, solver.quad_degree, keys)
    measure = solver.surface_measure(solver.quad_degree)(bc_id)
    assembled = np.array([fd.assemble(mode.expr ** 2 * measure)
                          for mode in descriptor.modes(side, radius, solver.X)])
    out["instrument_recovered_rule"] = float(np.max(np.abs(recovered - assembled)
                                                    / np.abs(assembled)))
    out["instrument_simplex_error"] = check(built(TQ.icosahedral_shell(2), SphericalDtN(L=2))[0])
    solver, _ = built(TQ.annulus(96, irregular=True), CylindricalDtN(M=20), quad_degree=5)
    out["instrument_explicit_report_none"] = int(solver.quad_rule_report is None)
    try:
        solver.check_boundary_quadrature(rtol=1e-6)
        out["instrument_inadequate_refused"] = 0
    except ValueError as error:
        out["instrument_inadequate_refused"] = int("does not resolve" in str(error))
    out["instrument_uniform_error"] = check(
        built(TQ.annulus(96), CylindricalDtN(M=20), quad_degree=5)[0])
    mesh = TQ.annulus(96, irregular=True)
    even = check(built(mesh, CylindricalDtN(M=20), quad_degree=4)[0], reference_degree=41)
    odd = check(built(mesh, CylindricalDtN(M=20), quad_degree=5)[0], reference_degree=41)
    out["instrument_even_odd_equal"] = int(even == odd)
    out["instrument_even"] = even
    return out


def run_quadrature_rank():
    """`rank_rows` alone, for the entry on two ranks."""
    out = {"ranks": fd.COMM_WORLD.size}
    rank_rows(out)
    return out


# -- Entry table --------------------------------------------------------------------

RUNS = {
    "closed-2d-multiplier": lambda: run_closed_2d("multiplier"),
    "closed-2d-lowrank": lambda: run_closed_2d("lowrank"),
    "closed-2d-lowrank-np2": run_closed_2d_two_ranks,
    "closed-3d-multiplier": lambda: run_closed_3d("multiplier"),
    "closed-3d-lowrank": lambda: run_closed_3d("lowrank"),
    "submesh": run_submesh,
    "adjoint-2d-multiplier": run_adjoint_2d_multiplier,
    "adjoint-2d-lowrank": run_adjoint_2d_lowrank,
    "adjoint-3d-multiplier": lambda: run_adjoint_3d("multiplier"),
    "adjoint-3d-lowrank": lambda: run_adjoint_3d("lowrank"),
    "quadrature-rule": run_quadrature_rule,
    "quadrature-rank-np2": run_quadrature_rank,
}
for _warp, _L in GRAVITY_QUADRATURE_SWEEP:
    RUNS[gravity_quadrature_entry(_warp, _L)] = (
        lambda warp=_warp, L=_L: run_quadrature_sweep(warp, L))
assert set(RUNS) == set(GRAVITY), set(RUNS) ^ set(GRAVITY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entry", required=True, choices=sorted(GRAVITY))
    name = parser.parse_args().entry
    print(f"gadopt from {gadopt.__file__}", flush=True)
    result = RUNS[name]()
    if isinstance(result, tuple):
        # A closed-form entry: its CG counts also go to their own file, with
        # the entry in each name, so that the counts files of the two serial
        # 2-D entries concatenate into one expected file without a clash.
        values, counts = result
        M.write_values(CASE_DIR / gravity_counts(name),
                       {f"{name}_{key}": value for key, value in counts.items()})
    else:
        values = result
    M.write_values(CASE_DIR / gravity_values(name), values)
