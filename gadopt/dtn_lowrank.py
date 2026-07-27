r"""The DtN boundary operator as a low-rank update, built once.

The multiplier formulation promotes every trace coefficient to a scalar `Real`
unknown and lets a Schur complement eliminate it. That keeps every term in
plain UFL - so pyadjoint tapes it and the trace coefficients come out as solved
unknowns - and it costs one boundary form per mode in the residual, re-assembled
on every matrix-free Krylov application.

This module builds the same discrete operator in the form Yu, Myhill & Al-Attar
(2026) use: eliminate the multipliers by hand and keep what is left, which is a
rank-`n` update to the Robin-shifted stiffness.

## The elimination

For mode `k` with eigenfunction `e_k`, write

    u_k = assemble(e_k * v * ds)          the boundary functional, one vector

`mu_k` is a *Real* test function, i.e. globally constant, so the constraint row
`integral (psi e_k - scale_k c_k) ds = 0` assembles to

    u_k^T psi - scale_k * A_h * c_k = 0,   A_h = assemble(1 * ds)

and the feedback term `(lam_k - alpha/R) c_k integral e_k v ds` becomes
`(lam_k - alpha/R) c_k u_k`. Hence

    c_k = u_k^T psi / (scale_k * A_h)
    B   = sum_k (lam_k - alpha/R) / (scale_k * A_h) * u_k u_k^T

**The denominator is `scale_k * A_h` and never `norm_k`.** `scale_k` is
`norm_k / |boundary|_analytic`, so the two agree only when the discrete boundary
measure equals the analytic one. Measured, substituting `norm_k` misses
`solver.coefficients()` by 3.6e-14 on a curved annulus but 2.7e-07 on a
cubed-sphere shell - above the parity gate, mode-independent, and shrinking
under refinement, so it looks like a discretisation effect and invites a
tolerance edit. It is not one. The correct expression reproduces
`coefficients()` to 1.5e-15.

## Building `u_k` without one compiled kernel per mode

The reference build is `assemble(e_k * v * ds)` per mode. It is the ground
truth, and it is too slow to ship: measured warm, 66-125 ms per mode, which at
the truncations this exists to reach is most of the time budget on its own.

The fast build uses one compiled kernel for all of them. Put a coefficient `g`
in a facet trace space and assemble `g * v * ds` once; then for each mode
overwrite `g`'s dofs from `dtn_tabulate` and re-execute. Two things make this
exact rather than approximate:

* the dofs are set **numerically** from the tabulation, never by interpolating
  the symbolic mode - `g.interpolate(e_k)` is itself a compiled kernel per mode
  and measured *slower* than the reference build it replaces; and
* `HDiv Trace` of degree `p` places its nodes at the `(p+1)`-point
  Gauss-Legendre points, and a measure of degree `q` uses a Gauss rule with
  `ceil((q+1)/2)` points. At `p = q // 2` those coincide, so `g` takes exactly
  the mode's values at exactly the points the rule evaluates and the assembled
  functional is the reference functional to round-off.

Measured against the reference: 2.3e-16 relative on the extruded cubed sphere,
1.1e-15 on the extruded annulus. The coincidence is non-monotone in `p` -
richer spaces give *worse* answers away from `p = q // 2` - which is the
signature that distinguishes it from ordinary interpolation accuracy.

On cost, two numbers, because they extrapolate differently. The reference build
is flat at 63-69 ms/mode *regardless of boundary size*, since it is dominated by
form compilation rather than by the mesh. The trace build is a fixed setup plus
a marginal per-mode assembly, so `setup_time` is recorded separately: at
`L = 10` on a level-2 shell the total is 0.96 s against the reference's 8.4 s,
an 8.7x whole-build ratio, but the marginal per-mode cost is what governs at the
truncations this exists for. Quoting the whole-build ratio at a small `n`
understates the gain, and quoting the marginal cost as if it were the whole
build overstates it; `NOTES/bench/build_cost.py` reports both on both axes.

That node placement is a variant choice inside FInAT rather than a documented
guarantee, and the same element at its default variant has equispaced nodes,
which would silently degrade the build to about `1e-8`. So the build **asserts**
itself against the reference on the highest-degree modes, unconditionally. It
must be the highest-degree ones: a constant is reproduced exactly by any nodal
interpolation whatever its node placement, so checking `Y_00` passes at 1.2e-16
on a build that is wrong by eight orders elsewhere.

Triangular facets are the known exception - a collapsed Gauss-Jacobi rule on a
triangle is not a tensor Gauss rule, and the coincidence genuinely fails there
(measured 1.0e-05 at `p = 3` falling monotonically to 4.3e-08 at `p = 6`, i.e.
ordinary interpolation error). Those meshes are detected up front and use the
reference build.
"""

import time

import numpy as np
from firedrake import (
    Function, FunctionSpace, SpatialCoordinate, TestFunction,
    VectorFunctionSpace, assemble)
from ufl.cell import TensorProductCell

from .dtn_tabulate import (
    tabulate_azimuthal_modes, tabulate_real_spherical_harmonics)

__all__ = ["BoundaryModeRows", "build_boundary_mode_rows",
           "boundary_facet_cellname", "supports_trace_build"]


def boundary_facet_cellname(mesh) -> str:
    """Reference cell of the facets a DtN boundary integral runs over.

    On an extruded mesh the DtN boundaries are `top`/`bottom`, whose facets are
    *base cells* rather than genuine codimension-1 sub-entities - which is also
    why a `Boundary Quadrature` space cannot be built on such a mesh at all
    (`make_quadrature_element` asks a `TensorProductCell` for a codim-1
    subelement, which does not exist: its facets come in two kinds).
    """
    cell = mesh.ufl_cell()
    if isinstance(cell, TensorProductCell):
        # The horizontal facets of an extruded cell are its base cell.
        return cell.sub_cells[0].cellname
    return {"triangle": "interval", "quadrilateral": "interval",
            "tetrahedron": "triangle", "hexahedron": "quadrilateral"}.get(
                cell.cellname, cell.cellname)


def supports_trace_build(mesh) -> bool:
    """Whether trace-space nodes coincide with the boundary quadrature points.

    True for interval and quadrilateral facets, where the rule is a tensor
    product of Gauss-Legendre points and `HDiv Trace` places its nodes on
    exactly those. False for triangular facets, whose collapsed Gauss-Jacobi
    rule is not a tensor rule - measured there, the build is an ordinary
    interpolation with error 1.0e-05 at `p = 3` and no exact `p`.
    """
    return boundary_facet_cellname(mesh) in ("interval", "quadrilateral")


class BoundaryModeRows:
    """The assembled boundary functionals of one DtN boundary.

    Attributes:
      keys: Mode labels, in `descriptor.modes` order.
      rows: `(n_modes, n_boundary_dofs)` array of the owned entries of
        `assemble(e_k * v * ds)`, restricted to the dofs any mode touches.
        Zero columns are dropped, which is what makes this the sparse `k x N`
        of the method rather than a dense one: a mode is supported on the whole
        boundary but on nothing else.
      dofs: Owned local indices of those columns.
      weights: `(lam_k - alpha/R) / (scale_k * A_h)`, the diagonal of the
        low-rank update.
      recovery: `1 / (scale_k * A_h)`, so `c_k = recovery_k * (u_k . psi)`.
      area: The discrete boundary measure `A_h = assemble(1 * ds)`.
      build_time: Seconds spent building, excluding the validation assembles.
      used_trace_build: Whether the fast path was used.
    """

    def __init__(self, keys, rows, dofs, weights, recovery, area,
                 build_time, setup_time, used_trace_build, trace_degree):
        self.keys = keys
        self.rows = rows
        self.dofs = dofs
        self.weights = weights
        self.recovery = recovery
        self.area = area
        self.build_time = build_time
        #: Of `build_time`, the part independent of the mode count: the trace
        #: space, the coordinate interpolation, the tabulation and the one form
        #: compilation. Split out because it dominates at the small truncations
        #: that run locally and vanishes at the ones this exists for, so a
        #: per-mode figure taken from `build_time` alone would extrapolate
        #: badly in the flattering direction.
        self.setup_time = setup_time
        self.used_trace_build = used_trace_build
        self.trace_degree = trace_degree

    @property
    def marginal_seconds_per_mode(self) -> float:
        """Build cost per mode with the one-off setup removed."""
        return (self.build_time - self.setup_time) / max(len(self.keys), 1)

    def coefficients(self, psi_local: np.ndarray) -> np.ndarray:
        """Rank-local contribution to `c_k`; the caller sums over ranks.

        `assemble` returns a Cofunction whose owned entries are already
        halo-reduced, so summing the owned dot products over ranks is the exact
        global dot with nothing double counted.
        """
        return self.rows @ psi_local[self.dofs] * self.recovery


def _tabulate(descriptor, side, points):
    """Mode values at `points`, in `descriptor.modes` order.

    The 2-D interior `mean` mode is the constant 1 and is not in the numerical
    tabulation, so it is prepended here - which is also the one place the two
    orderings could drift apart, and `tests/unit/test_dtn_tabulate.py` pins
    both against the descriptors' own keys.
    """
    if descriptor.dim == 3:
        return tabulate_real_spherical_harmonics(descriptor.L, points)
    table = tabulate_azimuthal_modes(descriptor.M, points[:, :2])
    if side == "interior":
        table = np.vstack([np.ones((1, points.shape[0])), table])
    return table


def _validation_modes(descriptor, side, table_keys):
    """Indices of the modes the build asserts itself on.

    The highest degree, both azimuthal branches. Never the constant: any nodal
    interpolation reproduces a constant exactly whatever its node placement, so
    `Y_00` cannot see a node/quadrature mismatch at all - measured, it passes at
    1.2e-16 on a build wrong by 8.3e-08 elsewhere.
    """
    if descriptor.dim == 3:
        L = descriptor.L
        wanted = [f"Y{L},{0}", f"Y{L},{L}", f"Y{L},{-L}"]
    else:
        M = descriptor.M
        wanted = [f"cos{M}", f"sin{M}"]
    return [table_keys.index(key) for key in wanted if key in table_keys]


def build_boundary_mode_rows(
    solution_space, measure, descriptor, side, radius, alpha,
    quad_degree, force_reference=False, rtol=1e-13,
) -> BoundaryModeRows:
    """Assemble `u_k = integral e_k v ds` for every mode of one DtN boundary.

    Args:
      solution_space: The potential's function space.
      measure: The boundary measure, already carrying `quad_degree`.
      descriptor: `CylindricalDtN` or `SphericalDtN`.
      side: "exterior" or "interior".
      radius: Measured boundary radius.
      alpha: The Robin shift.
      quad_degree: Degree of `measure`; sets the trace degree `quad_degree//2`.
      force_reference: Use the per-mode symbolic build even where the fast one
        would work. Ground truth for the tests.
      rtol: Tolerance of the self-assertion against the reference.

    Returns:
      A `BoundaryModeRows`.
    """
    mesh = solution_space.mesh()
    v = TestFunction(solution_space)
    X = SpatialCoordinate(mesh)
    modes = descriptor.modes(side, radius, X)
    keys = [mode.key for mode in modes]
    area = assemble(1 * measure)

    use_trace = supports_trace_build(mesh) and not force_reference
    trace_degree = quad_degree // 2 if use_trace else None

    start = time.perf_counter()
    if use_trace:
        rows, setup_time = _trace_build(mesh, v, measure, descriptor, side,
                                        trace_degree, len(modes))
    else:
        rows = np.array([np.asarray(assemble(mode.expr * v * measure).dat.data_ro,
                                    dtype=float) for mode in modes])
        setup_time = 0.0
    build_time = time.perf_counter() - start

    if use_trace:
        _assert_against_reference(rows, modes, keys, descriptor, side, v,
                                  measure, mesh, trace_degree, rtol)

    # Drop the columns no mode touches: a mode is supported on the whole
    # boundary and on nothing else, so this is the sparsity of the method.
    dofs = np.flatnonzero(np.any(rows != 0.0, axis=0))
    rows = np.ascontiguousarray(rows[:, dofs])

    denominator = np.array([mode.scale * area for mode in modes])
    weights = np.array([mode.lam - alpha / radius for mode in modes]) / denominator
    return BoundaryModeRows(
        keys, rows, dofs, weights, 1.0 / denominator, area, build_time,
        setup_time, use_trace, trace_degree)


def _trace_build(mesh, v, measure, descriptor, side, trace_degree, n_modes):
    """One compiled kernel, `n` re-executions with the coefficient overwritten."""
    setup_start = time.perf_counter()
    W = FunctionSpace(mesh, "HDiv Trace", trace_degree)
    Wv = VectorFunctionSpace(mesh, "HDiv Trace", trace_degree)
    nodes = Function(Wv).interpolate(SpatialCoordinate(mesh))
    table = _tabulate(descriptor, side, np.asarray(nodes.dat.data_ro, dtype=float))
    if table.shape[0] != n_modes:
        raise ValueError(
            f"tabulation produced {table.shape[0]} modes but the descriptor "
            f"table has {n_modes}; the two orderings have drifted apart")

    g = Function(W)
    form = g * v * measure
    assemble(form)  # compile once; part of setup, not of the per-mode cost
    setup_time = time.perf_counter() - setup_start
    rows = []
    for values in table:
        # Writing the owned dofs invalidates the halo; the assembly reads g,
        # which triggers global_to_local before the parloop, so ghost entries
        # are filled from their owners. That guarantee lapses only under a
        # frozen halo, so this loop must not run inside one.
        g.dat.data[:] = values
        rows.append(np.asarray(assemble(form).dat.data_ro, dtype=float))
    return np.array(rows), setup_time


def _assert_against_reference(rows, modes, keys, descriptor, side, v, measure,
                              mesh, trace_degree, rtol):
    """Check the fast build against the symbolic one on the sensitive modes."""
    indices = _validation_modes(descriptor, side, keys)
    if not indices:
        return
    scale = max(np.max(np.abs(rows[i])) for i in indices)
    if scale == 0.0:
        return
    worst, worst_key = 0.0, None
    for i in indices:
        reference = np.asarray(
            assemble(modes[i].expr * v * measure).dat.data_ro, dtype=float)
        deviation = np.max(np.abs(rows[i] - reference)) / scale
        if deviation > worst:
            worst, worst_key = deviation, keys[i]
    if worst > rtol:
        raise RuntimeError(
            f"The tabulated DtN boundary build disagrees with the symbolic one "
            f"by {worst:.3e} (> {rtol:.1e}) on mode {worst_key}.\n"
            f"Cause: the build requires the trace-space nodes to coincide with "
            f"the boundary quadrature points, which holds when 'HDiv Trace' of "
            f"degree {trace_degree} places its nodes at the "
            f"{trace_degree + 1}-point Gauss-Legendre points and the measure "
            f"uses that same rule. Boundary facets here are "
            f"'{boundary_facet_cellname(mesh)}'. If that is a simplex facet the "
            f"rule is a collapsed Gauss-Jacobi one and no degree coincides; "
            f"otherwise the element's node variant has changed. Either way the "
            f"representation is no longer exact - rebuild with "
            f"force_reference=True, or fall back to the multiplier path.")


class LowRankDtNOperator:
    """`A + B` as a PETSc python `Mat`, with `B` applied in factored form.

    `B = sum_k w_k u_k u_k^T` is dense on the boundary trace and is never
    formed. One application is, per boundary,

        y += C^T (W (C x))

    with `C` the rank-local rows and `W` the diagonal of weights: two small
    dense products against the boundary entries, and **one `Allreduce` of the
    `k`-vector** in between. That is the paper's three-stage parallel algorithm
    and the reason the rows are stored rank-locally rather than as a
    distributed `k x N` matrix - a distributed `C` would put each row on one
    rank, and since every row needs the whole boundary trace, `MatMult`'s
    scatter would move `O(N_boundary)` words per row-owning rank instead of
    `O(k)` words in total.

    `A + B` is SPD even though individual weights are negative: `w_k` is
    negative for any mode with `lam_k R < alpha`, which the interior `l = 0`
    mean mode (`lam = 0`) always is. Writing

        A + B = stiffness
              + sum_k lam_k/(scale_k A_h) u_k u_k^T
              + (alpha/R) ( M_boundary - sum_k u_k u_k^T/(scale_k A_h) )

    the last bracket is PSD by **Bessel's inequality** applied to the
    L^2(boundary)-orthogonal family `{e_k}`: it is the boundary mass form minus
    the projection onto the treated modes, i.e. the energy of the untreated
    tail. The middle sum is PSD because every `lam_k >= 0`. So the whole is the
    exact DtN energy plus the untreated Robin, and CG is legitimate. The
    orthogonality that argument needs is the same property
    `check_boundary_quadrature` verifies mode by mode.

    (With an interior boundary and no exterior one, constants sit in the
    nullspace. That is the physics of a source-free core, not a defect.)
    """

    def __init__(self, assembled, mode_rows, comm):
        self.assembled = assembled
        self.mode_rows = list(mode_rows)
        self.comm = comm
        #: Counts applications, so a test can confirm the operator was
        #: exercised rather than bypassed.
        self.applications = 0

    def mult(self, mat, x, y):
        self.assembled.mult(x, y)
        self._add_low_rank(x, y)
        self.applications += 1

    # Symmetric, so the transpose action is the same action. Stated rather than
    # inherited: PETSc will use this for a transposed solve, and getting it
    # wrong would show up only in the adjoint.
    multTranspose = mult

    def _add_low_rank(self, x, y):
        xa = x.array_r
        ya = y.array_w
        for rows in self.mode_rows:
            if rows.rows.size:
                local = rows.rows @ xa[rows.dofs]
            else:
                local = np.zeros(len(rows.keys))
            total = np.empty_like(local)
            self.comm.Allreduce(local, total)
            if rows.rows.size:
                ya[rows.dofs] += rows.rows.T @ (rows.weights * total)

    def coefficients(self, psi_local):
        """Trace coefficients of every boundary, summed across ranks."""
        out = []
        for rows in self.mode_rows:
            local = rows.coefficients(psi_local)
            total = np.empty_like(local)
            self.comm.Allreduce(local, total)
            out.append(total)
        return out


def apply_dirichlet_to_rows(mode_rows, constrained_dofs):
    """Zero the columns of `C` at strongly constrained degrees of freedom.

    Not optional. `A` has its constrained rows and columns eliminated, so a `B`
    that still couples to those dofs makes `A + B` inconsistent with `A` - the
    operator would apply a boundary feedback through a degree of freedom whose
    value is prescribed, and the result is not symmetric with the eliminated
    `A` either.
    """
    constrained = np.asarray(sorted(constrained_dofs), dtype=np.int64)
    if constrained.size == 0:
        return
    for rows in mode_rows:
        if not rows.rows.size:
            continue
        mask = np.isin(rows.dofs, constrained)
        if mask.any():
            rows.rows[:, mask] = 0.0
