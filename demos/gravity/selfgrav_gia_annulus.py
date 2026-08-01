"""The monolithic self-gravitating GIA prototype, on the Phase 2 annulus.

Driver for `gadopt.gia_gravity.SelfGravitatingGIASolver`: builds the four-region
parent annulus of `generate_selfgrav_annulus.py`, extracts the mantle as a
`Submesh`, couples the viscoelastic mechanics to the gravitational Poisson
equation and the rotational closure in one mixed space, and solves.

    PYTHONPATH=<worktree> \\
      ~/Workplace/firedrake-2026-03-03/venv-firedrake/bin/python3 \\
      demos/gravity/selfgrav_gia_annulus.py [--dr 0.1] [--nazim 64] [--gates]

## Time stepping

`--steps 1`, the default, is the single backward-Euler step every gate so far
has used. Anything larger runs `run_to_steady_state`, whose docstring explains
why the loop is here and not on the solver class. The **production
configuration has no fluid limit** - the deflection grows exponentially, and so
does the uncoupled mechanics on the same mesh, so it is the uniform-density
reference state and not the coupling - which is what `--fluid-limit` and
`--lam-factor` are for; `build_solver`'s docstring has the measurements.
Verified steady state on the coarse mesh:

    --dr 0.2 --nazim 32 --truncation 3 --dt 0.5 --steps 300 \\
      --fluid-limit --lam-factor 0.25

reaches Airy isostasy `zeta = -sigma_hat/rho_0` in 187 steps, `-9.9955e-04`
against `-1.0e-03`.

## The load

A `sigma_hat cos(2 phi)` mass sheet on the Earth's surface at Re. It appears
twice, and the two must stay one object: as a `normal_stress` on the mechanics
boundary, pressing the surface down, and as an `interior_sigma` sheet in the
potential, attracting through its own mass. Re is an **interior** facet of the
parent - that is the whole point of the stand-off geometry - so the sheet is a
`dS` integral, and the shipped exterior-facet spelling would integrate over
nothing and only warn.

Degree two, and not by accident: it is the degree the rotational feedback reads,
the degree the fluid-limit gates use, and a degree whose net mass over a circle
is zero, which keeps the 2-D monopole datum out of the way until something needs
it.

## Curve BOTH meshes

`Submesh` does **not** inherit the parent's P2 coordinates, so a submesh of a
curved parent is straight-sided and reports the polygon error, 4e-04 relative on
the circles. `curve_mesh` on the submesh recovers the parent's accuracy to the
same digits, because it is a pointwise function of vertex coordinates and the
two meshes share them, and cross-mesh assembly survives the re-curving (Phase 2
measured 1.2e-08 against 4.02e-04 for a coupling-shaped integral). The error the
un-recurved version makes is concentrated exactly at Rc and Re, which is where
the interface mass sheets carrying the entire source live.

## The gates this script runs

`--gates` runs the two that are preconditions for everything else. It does not
run V1, V3', V7, V8, V9 or V9b; those come next and are not in this file.

**G0 - the mechanics asymmetry baseline.** The coupled Jacobian is *not*
symmetric, because the mechanics Jacobian already is not: `viscosity_term`
differentiates to a full DG1 tensor in the internal variable while the
internal-variable source uses `deviatoric_strain`, so the two off-diagonal
blocks differ by the symmetric-deviatoric projector, and no row scaling repairs
it - the nonzero counts differ by a factor of two. On a 4x4 Cartesian square
that is `max|M - M^T|/max|M| = 3.2e-02` with the `(u, m)` pair at 1.9e-01
against 2.1e-01. A symmetry test written globally would therefore fail at
percent level on its first run before a single coupling term was wrong, and the
natural reaction - hunting for a sign error in the new code - wastes a day. So
the baseline is measured *here*, on the mesh the coupled solver actually uses,
with the coupling absent, and recorded - and measured twice, because `un = 0`
adds a second, larger source of asymmetry that the review did not see.

**V4 - discrete mass conservation.** The source is written as a divergence,
`Lambda int rho_0 u . grad(v)`, so with `v = 1` it is identically zero and the
discrete perturbation carries exactly zero net mass - not `O(h^p)`, exactly.
This is what keeps the 2-D solver inside the regime its monopole and log-gauge
treatment was built for.
"""
import argparse
import os
import sys

# BEFORE firedrake. PETSc imports gadopt lazily when it reads
# `pc_python_type: gadopt.DtNTwoBlockSchurPC`, long after a UFL multifunction
# has run, and Irksome's import-order guard then fires as a bare
# `petsc4py.PETSc.Error: error code 101` with nothing in the traceback naming
# either package.
import gadopt  # noqa: F401
import numpy as np  # noqa: E402
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

#: Road-map §2.2 with the production constants: rho_bar = 4500 kg/m^3,
#: D = 2891 km, g_bar = 9.815 m/s^2, mu_bar = 1e11 Pa.
B_MU = 1.2769         # rho_bar g_bar D / mu_bar
LAMBDA = 1.1116       # 4 pi G rho_bar D / g_bar, and Lambda ~ 1 is the point:
#                       self-gravitation is leading order here, not a correction.
SIGMA_HAT = 1.0e-3    # surface load amplitude, non-dimensional
LOAD_DEGREE = 2

HERE = os.path.dirname(os.path.abspath(__file__))


def build_meshes(dr_mantle, n_azimuthal, path=None):
    """The parent annulus and its mantle submesh, both P2-curved.

    gmsh runs on rank 0 only and every rank then reads the file: gmsh is not
    collective, and all ranks writing one path is a race that produces a
    truncated mesh rather than an error.
    """
    path = path or os.path.join(HERE, "selfgrav_gia_annulus.msh")
    if COMM_WORLD.rank == 0:
        gen.generate(path, dr_mantle=dr_mantle, n_azimuthal=n_azimuthal)
    COMM_WORLD.barrier()

    parent = curve_mesh(Mesh(path))
    parent.cartesian = False
    # Curve the submesh too. `Mesh(X_p2)` is a new geometry over the same
    # topology, and the entity maps to the parent survive it.
    sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
    sub.cartesian = False
    return parent, sub


def approximation(density=1.0, g=1.0, self_gravity_number=None):
    """A fresh approximation object, and it must be fresh.

    `CoupledInternalVariableSolver.__init__` does
    `approximation.mu = approximation.effective_viscosity(dt)`, mutating the
    object. Two solvers built from one approximation at different time steps
    silently share the later `mu`; G0 builds a second solver, so each gets its
    own.

    `g` and `self_gravity_number` are keywords with the production defaults, so
    that the gates which monkey-patch this function with a one-argument factory
    keep working; only the fluid-limit configuration passes anything else, and
    `build_solver` passes each one only when it differs. See `build_solver`.
    """
    return CompressibleInternalVariableApproximation(
        bulk_modulus=1.0, density=density, shear_modulus=1.0, viscosity=1.0,
        g=g, B_mu=B_MU,
        self_gravity_number=LAMBDA if self_gravity_number is None
        else self_gravity_number)


def build_solver(parent, sub, *, dt=1.0, truncation=5, rotation=True,
                 solver_parameters=None, solver_parameters_extra=None,
                 declare_nullspace=True, fluid_limit=False, lam_factor=1.0,
                 fluid_core=None, **solver_kwargs):
    """The coupled solver, its mixed function, and the pieces a gate needs.

    `declare_nullspace` is on by default and is defect D-2's fix. With free slip
    `un = 0` at the CMB and traction at the surface there is no condition on the
    *tangential* displacement anywhere, so the rigid rotation `(-y, x)` is a
    zero-energy mode of the mechanics - and of everything the coupling adds,
    since it is divergence free, normal to neither circle, and contributes
    nothing to `dI_33`. Undeclared, the mode is annihilated only to
    `||J u_rot||/||u_rot|| ~ 2e-06` by facet geometry error, and two runs of the
    same system land on different multiples of it. See
    `gadopt.gia_gravity.rigid_rotation_nullspace`.

    `fluid_limit` switches to the configuration V9 and V9b need, and it is a
    different physical configuration rather than a numerical option, so it is
    off by default and everything already measured is untouched.

    **The production configuration has no fluid limit, and that is measured
    rather than feared.** Stepped past a few Maxwell times the surface
    deflection grows exponentially at about `2e-02` per unit time - and so does
    a plain `CoupledInternalVariableSolver` on the same mantle with the same
    load and no gravitational coupling whatever, at about `1e-02`. Self-gravity
    roughly doubles the rate; it does not cause it. The cause is the reference
    state: `hydrostatic_prestress_advection_and_buoyancy_term` linearises about
    a body of *uniform* density in a *constant* gravity field, which is not a
    hydrostatic equilibrium, and the linearised operator about a non-equilibrium
    state has a growing mode. Over a glacial cycle - ten Maxwell times, which is
    what the production GIA demos run - the e-folding time of about 50 is
    invisible. A fluid-limit gate steps to a hundred and it is not.

    Two changes make the limit exist and both are physically motivated:

    - `g = 0` in the approximation, which removes the volume prestress term
      entirely and with it the non-hydrostatic reference state;
    - an explicit Airy restoring stress `B_mu rho_0 g_s u_r` at Re, which is the
      weight of the deflected surface mass sheet. A *uniform* `rho_0` has no
      gradient, so no volume term can supply it, and
      `CompressibleInternalVariableApproximation.hydrostatic_prestress_advection`
      returns 0 on the grounds that it is absorbed into the volume term - which
      for uniform density absorbs it into nothing.

    The fluid limit is then Airy isostasy, `zeta = -sigma_hat / rho_0`, which is
    exactly what road-map V9b asserts, and it is reached: measured
    `-9.9964589e-04` against `-1.0e-03`, 3.5e-04 relative, on the coarse mesh -
    **with the coupling switched off**. With it on at the nominal `Lambda` the
    run relaxes cleanly towards Airy for about a hundred Maxwell times, reaching
    `-8.9e-04`, and then diverges on an `m = 1` mode: at `Lambda = 1.1116` this
    self-gravitating body is *supercritical*, which is not surprising once `g`
    has been set to zero, since the reference gravity that would stabilise it
    has gone with it. `lam_factor` scales `Lambda` for exactly that reason.
    Measured on the coarse mesh: unstable at `1.0`, and at `0.25` the coupled
    system relaxes to `-9.9956e-04`, 4.4e-04 from Airy. Rotation is not
    implicated - the divergence is identical with it switched off.
    """
    X = SpatialCoordinate(parent)
    sigma = SIGMA_HAT * cos(LOAD_DEGREE * atan2(X[1], X[0]))
    lam = LAMBDA * lam_factor

    gravity_bcs = {
        gen.CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
        gen.CURVE_RE: {"interior_sigma": sigma},
    }
    Z, layout = self_gravitating_gia_space(
        sub, parent, gravity_bcs=gravity_bcs, rotation=rotation,
        self_gravity_number=lam)

    z = Function(Z)
    z.subfunctions[layout.displacement].rename("displacement")
    z.subfunctions[layout.potential].rename("potential")

    # Each keyword is passed only when it differs from the default, so that a
    # gate which has monkey-patched `approximation` with a one-argument factory
    # keeps working on the configurations it was written for.
    approx_kwargs = {}
    if fluid_limit:
        approx_kwargs["g"] = 0.0
    if lam_factor != 1.0:
        approx_kwargs["self_gravity_number"] = lam
    approx = approximation(**approx_kwargs)
    Xm = SpatialCoordinate(sub)
    dx_m = Measure("dx", domain=sub,
                   intersect_measures=(Measure("dx", domain=parent),))
    # The disc's polar second moment. In 2-D `C` is computable from the
    # reference density; in 3-D `C - A` is the dynamical ellipticity of the
    # hydrostatic figure and is an input, because a spherically symmetric
    # rho_0(r) has C = A identically.
    C = assemble(approx.density * dot(Xm, Xm) * dx_m)

    sigma_m = SIGMA_HAT * cos(LOAD_DEGREE * atan2(Xm[1], Xm[0]))
    surface_stress = sigma_m
    if fluid_limit:
        # The weight of the deflected surface mass sheet, `rho_0 g_s u_r`,
        # implicit in the unknown so the Jacobian carries it. The sign is
        # measured, not asserted: with the opposite one the deflection diverges
        # at 37 % per step within ten steps, so the rejection region is not
        # subtle. `g_s = 1` is the surface gravity in the demo's units.
        normal = Xm / sqrt(dot(Xm, Xm))
        surface_stress = surface_stress + approx.density * dot(
            split(z)[layout.displacement], normal)
    bcs = {
        # The load, pressing on the surface. Same amplitude and degree as the
        # sheet in the potential; they are the same physical object.
        gen.CURVE_RE: {"normal_stress": B_MU * surface_stress},
    }
    if fluid_core is None:
        # The legacy switch: the core is rigid, no radial displacement, which
        # also kills the rho_below sheet the divergence form would otherwise
        # need at the CMB. Retained so that the difference between the two CMB
        # treatments stays a measured quantity - see `gadopt.gia_gravity.FluidCore`.
        bcs[gen.CURVE_RC] = {"un": 0.0}
    else:
        solver_kwargs["fluid_core"] = fluid_core

    if declare_nullspace and "nullspace" not in solver_kwargs:
        solver_kwargs["nullspace"] = rigid_rotation_nullspace(Z, layout)

    solver = SelfGravitatingGIASolver(
        z, approx, layout=layout, dt=dt, bcs=bcs,
        rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH,
        solver_parameters=solver_parameters,
        solver_parameters_extra=solver_parameters_extra,
        **solver_kwargs)
    return solver, z, layout, bcs, C


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
def deflection_amplitude(solver):
    r"""The `cos 2 phi` amplitude of the radial displacement at the surface.

    The physical quantity the fluid-limit gates read, and the one the
    steady-state criterion is stated on: a norm of the whole displacement would
    let a converged surface hide behind a still-relaxing interior.

    Re is an *exterior* facet of the mantle submesh - it is only an interior
    facet of the parent, which is what the stand-off buffer is for - so this is
    a plain `ds` and not a `dS`.
    """
    sub = solver.mesh
    X = SpatialCoordinate(sub)
    n = X / sqrt(dot(X, X))
    u_r = dot(solver.displacement, n)
    dss = Measure("ds", domain=sub)(gen.CURVE_RE)
    length = assemble(Constant(1.0) * dss)
    return 2 * assemble(u_r * cos(LOAD_DEGREE * atan2(X[1], X[0])) * dss) / length


def fluid_limit_residual(solver, *, with_floor=False):
    r"""`||s|| / (2 mu_0 ||e_dev||)`: how far the deviatoric stress is from zero.

    **The independent certificate that the fluid limit has actually been
    reached**, rather than the circular one that the answer stopped changing.
    The internal-variable evolution is `dm/dt = (e_dev - m)/tau`, so a steady
    state has `m = e_dev` exactly, and with `mu_0 = sum_k mu_k` the deviatoric
    stress `s = 2 mu_0 e_dev - sum_k 2 mu_k m_k` is then identically zero. A
    relaxed Maxwell body supports no deviatoric stress: what holds the surface
    up in the fluid limit is the bulk term alone, which is compressible Airy
    isostasy.

    So this number falling to zero is a statement about the *physics* of the
    state, computable from a single snapshot, and it is not implied by a small
    change per step - a run with too large a `dt` and a stalled Newton step
    would report the second and not the first.

    Normalised by the deviatoric strain the stress is built from, so it is the
    fraction of the elastic stress that survives and not an absolute size.

    **It does not go to zero, and the floor is the discretisation rather than
    an unrelaxed state.** The steady internal-variable equation is
    `int (m - e_dev) phi = 0` for every `phi` in the internal-variable space,
    i.e. `m` is the L2 *projection* of `e_dev` onto that space - and on this
    mesh `u` is Q2, so `e_dev` lives in `Q(1,2) x Q(2,1)` and not in the DQ1 the
    internal variable has. The projection error is `O(h^2)` and measures a few
    parts in a thousand on the development mesh. `with_floor=True` returns
    `(residual, floor)` with the floor computed by projecting `e_dev` directly:
    the fluid limit is reached when the two agree, not when the residual
    reaches zero, and a check written against zero would never pass on any
    mesh.
    """
    # The *sub-functions* and a plain submesh measure, not `solution_split` and
    # `dx_m`: this is a diagnostic on a state rather than a residual term, and
    # a `split` of a mixed function spanning two meshes carries both domains,
    # which `project` refuses with "Found multiple domains".
    u = solver.displacement
    ivs = [solver.solution.subfunctions[i]
           for i in solver.layout.internal_variables]
    approx = solver.approximation
    s = approx.deviatoric_stress(u, ivs)
    e = approx.deviatoric_strain(u)
    dxm = Measure("dx", domain=solver.mesh)
    scale = 2 * approx.mu0
    den = sqrt(assemble(inner(scale * e, scale * e) * dxm))
    if den <= 0.0:
        return (0.0, 0.0) if with_floor else 0.0
    residual = sqrt(assemble(inner(s, s) * dxm)) / den
    if not with_floor:
        return residual

    projected = Function(ivs[0].function_space()).project(e)
    floor = sqrt(assemble(
        inner(scale * (e - projected), scale * (e - projected)) * dxm)) / den
    return residual, floor


def run_to_steady_state(solver, *, dt, max_steps=200, rtol=1e-8,
                        verbose=True, every=1):
    r"""Steps the coupled solver to the fluid limit and reports whether it got there.

    **The loop is here and not on the solver class, deliberately.**
    `SelfGravitatingGIASolver` derives from `CoupledInternalVariableSolver`,
    which carries the internal variables *inside* the mixed solution and reads
    their old values through `solution_old_split`; `StokesSolverBase.solve`
    ends with `solution_old.assign(self.solution)`. So the whole time-stepping
    convention of that family is *one `solve()` call per step and nothing
    else* - unlike `InternalVariableSolver`, which keeps the internal variables
    outside the solution and interpolates them forward by hand. There is no
    per-step bookkeeping for a driver to get wrong, and adding a loop to the
    class would only hide that.

    Keeping the solver a single-step object also keeps `dt` honest.
    `approximation.mu = approximation.effective_viscosity(dt)` is evaluated in
    the constructor and baked into the Nitsche penalty and the
    internal-variable equations, so a solver cannot change its own time step -
    growing `dt` towards the fluid limit means building a new solver and
    discarding the block-0 factorisation. A single-step object makes that
    impossible to do by accident.

    The criterion is the relative change of the *surface deflection* between
    consecutive steps. Backward Euler on this linear system relaxes
    geometrically, with ratio `r = delta_n / delta_{n-1}` measured from the last
    two steps, so the tail that has not been taken is
    `delta r / (1 - r)` - reported as `remaining`, because "the last step was
    small" and "there is nothing left" are different statements and only the
    second one is the fluid limit.

    Args:
      solver: a built `SelfGravitatingGIASolver`.
      dt: its time step, for reporting only - the solver already has it.
      max_steps: give up after this many.
      rtol: stop when the relative change of the deflection falls below it.
      verbose: print the per-step table.
      every: print every `n`th row (the first, the last and the stop are always
        printed).

    Returns:
      A dict with `steps`, `time`, `deflection`, `history`, `change` (the final
      relative change), `ratio`, `remaining` (the geometric estimate of the
      untaken tail), `fluid_residual` and its `fluid_floor`, `relaxed` (the
      fluid-limit certificate: the residual is at its floor) and `converged`
      (the steady-state criterion). The last two are different statements and
      both are reported.
    """
    history, changes = [], []
    previous = None
    converged = False

    if verbose:
        print(f"\nTime stepping to the fluid limit: dt {dt}, Maxwell time "
              f"{float(solver.approximation.maxwell_times[0]):.4g}, "
              f"rtol {rtol:g}, at most {max_steps} steps")
        print(f"  {'step':>5s}{'t':>10s}{'deflection at Re':>20s}"
              f"{'rel change':>14s}{'dev stress':>14s}")

    for step in range(1, max_steps + 1):
        solver.solve()
        zeta = deflection_amplitude(solver)
        history.append(zeta)
        change = (abs(zeta - previous) / abs(zeta) if previous is not None
                  and zeta != 0.0 else float("inf"))
        if previous is not None:
            changes.append(change)
        previous = zeta

        converged = change < rtol
        show = verbose and (step == 1 or converged or step == max_steps
                            or step % every == 0)
        if show:
            print(f"  {step:5d}{step * dt:10.3f}{zeta:20.10e}"
                  f"{change:14.3e}{fluid_limit_residual(solver):14.3e}")
        if converged:
            break

    ratio = (changes[-1] / changes[-2]
             if len(changes) >= 2 and changes[-2] > 0.0 else float("nan"))
    remaining = (changes[-1] * ratio / (1.0 - ratio)
                 if changes and ratio == ratio and ratio < 1.0
                 else float("nan"))
    residual, floor = fluid_limit_residual(solver, with_floor=True)
    relaxed = residual < 1.5 * floor

    if verbose:
        print(f"  steps taken                       {step}")
        print(f"  final relative change per step    {changes[-1]:.3e}"
              if changes else "  final relative change per step    n/a")
        print(f"  geometric ratio of the change     {ratio:.6f}")
        print(f"  estimated remaining change        {remaining:.3e}")
        print(f"  elastic residual: dev stress      {residual:.3e}")
        print(f"                    its floor       {floor:.3e}   "
              "(the DQ1 projection error of the deviatoric strain; a relaxed "
              "Maxwell body supports no deviatoric stress, so this is as low "
              "as the discretisation lets the residual go)")
        print(f"  fluid limit reached               "
              f"{'YES' if relaxed else 'NO - still carrying elastic stress'}")
        print(f"  steady state reached              "
              f"{'YES' if converged else 'NO - hit the step limit'}")

    return {"steps": step, "time": step * dt, "deflection": history[-1],
            "history": history, "change": changes[-1] if changes else None,
            "ratio": ratio, "remaining": remaining,
            "fluid_residual": residual, "fluid_floor": floor,
            "relaxed": relaxed, "converged": converged}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def block_asymmetry(F, z):
    """Per-block `max|A_ij - A_ji^T|`, and the max-abs grid beside it.

    `mat_type="nest"` with `getNestSubMatrix`, which is spike S5's route B: it
    assembles with `Real` blocks present (monolithic `aij` does not), it is
    sharp to ~1e-17, and it *names the block*, which for a sign-error hunt
    across twenty-odd coupling terms is worth more than one scalar. It needs a
    global dense transpose, so it is serial.
    """
    Z = z.function_space()
    A = assemble(derivative(F, z), mat_type="nest").petscmat
    n = len(Z)
    diff = np.zeros((n, n))
    size = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Mij = A.getNestSubMatrix(i, j)
            Mji = A.getNestSubMatrix(j, i)
            if Mij is None or Mji is None:
                continue
            aij = Mij.convert("dense").getDenseArray()
            aji = Mji.convert("dense").getDenseArray()
            diff[i, j] = np.abs(aij - aji.T).max()
            size[i, j] = np.abs(aij).max()
    return diff, size


def mechanics_baseline(sub, bcs, dt=1.0):
    """One `CoupledInternalVariableSolver` on the mantle alone, measured."""
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    Zm = MixedFunctionSpace([V, S])
    zm = Function(Zm)
    solver = CoupledInternalVariableSolver(
        zm, approximation(), dt=dt, bcs=bcs, solver_parameters="direct")

    diff, size = block_asymmetry(solver.F, zm)
    A = assemble(derivative(solver.F, zm), mat_type="nest").petscmat
    dense = np.block(
        [[A.getNestSubMatrix(i, j).convert("dense").getDenseArray()
          for j in range(2)] for i in range(2)])
    return {
        "dim": Zm.dim(),
        "global": np.abs(dense - dense.T).max() / np.abs(dense).max(),
        "uu": diff[0, 0], "mm": diff[1, 1],
        "um_pair": diff[0, 1], "um_size": size[0, 1], "mu_size": size[1, 0],
        "uu_size": size[0, 0],
    }


def gate_g0(parent, sub, bcs, dt=1.0):
    """G0: the mechanics asymmetry, with the coupling absent.

    A plain `CoupledInternalVariableSolver` on the mantle alone, same mesh, same
    approximation constants, same boundary conditions, same time step. Without
    this number the coupled symmetry test has no reference and fails for the
    wrong reason.

    Measured **twice**, with the boundary conditions and without them, because
    on this configuration there are *two* independent sources of mechanics
    asymmetry and the review only knew about one:

    - the `(u, m)` pair, structural and boundary-independent, from
      `viscosity_term` differentiating to a full DG1 tensor in the internal
      variable while the internal-variable source uses `deviatoric_strain`;
    - the `(u, u)` block, which is symmetric to roundoff with no boundary
      condition, with a strong `u` condition and with `normal_stress`, and is
      asymmetric at percent level with **`un = 0`** - the free-slip Nitsche
      block. Measured here at 2.33 against a block max of 1.92e+02, and
      unchanged when the bulk contribution is switched off, so it is the
      deviatoric pair rather than the bulk one.

      **Its cause is `mu`, not `nabla_grad`** (corrected 2026-08-01; this
      docstring used to blame a structural mismatch - the adjoint-consistency
      term carrying the full `nabla_grad` of the test function against a
      consistency partner keeping only `n . stress . n`). That story cannot be
      right, because it would be independent of the time step and the
      asymmetry is not: 8.39e-01, 1.53e-01, 1.66e-02, 1.68e-03, 1.68e-04 at
      `dt = 1e0 ... 1e-4`, exactly linear over four decades and tracking
      `dt/(tau + dt)` to three digits. `viscosity_term` takes
      `mu = eq.approximation.mu`, which `CoupledInternalVariableSolver.__init__`
      has already overwritten with
      `effective_viscosity(dt) = mu_0 tau/(tau + dt)`, for the Nitsche penalty
      and adjoint-consistency terms - while their consistency partner uses the
      full `stress`. In the *segregated* solver `update_m` substitutes the
      internal variable, so the stress's effective deviatoric modulus is that
      same `mu_0 tau/(tau + dt)` and the pair matches; in the *coupled* solver
      `m` is an independent unknown, the `(u, u)` part of the stress carries
      `mu_0`, and the mismatch is `mu_0/mu_eff = 1 + dt/tau`. That is the only
      account explaining the recorded fact that this asymmetry is
      `CoupledInternalVariableSolver`-only, which a structural mismatch in a
      shared term could not.

    **And it has since been fixed at source, so this docstring describes
    history** (2026-08-01). `CoupledInternalVariableSolver.__init__` now hands
    the Nitsche terms `mu0` rather than `effective_viscosity(dt)`, which is the
    coefficient its own stress tangent carries when `m` is an independent
    unknown, and the `(u, u)` asymmetry is gone: FC-4 measures 7.54e-17 for
    `un = 0` where it measured 1.23e-02. The correction above is what
    identified the mechanism, and `NOTES/FINDING-FREESLIP-NITSCHE-ASYMMETRY.md`
    carries the algebra - including that the choice *flips back* under static
    condensation, because condensing substitutes the backward-Euler update into
    the stress and takes its `u`-tangent from `mu0` to `effective_viscosity(dt)`
    exactly.

    So the number this gate quarantines is now the `(u, m)` pair alone. Keep
    measuring the `(u, u)` block anyway: the old value returning is the
    signature of that flag being set the wrong way for the configuration.
    """
    with_bcs = mechanics_baseline(sub, bcs, dt=dt)
    without = mechanics_baseline(sub, {}, dt=dt)

    print("\nG0 - mechanics asymmetry baseline (coupling OFF)")
    print(f"  mantle cells                       {sub.num_cells()}")
    print(f"  mixed dim [CG2 vector, DG1 tensor] {with_bcs['dim']}")
    print(f"  {'':34s}{'with bcs':>14s}{'no bcs':>14s}")
    for label, key in [("global max|M-M^T|/max|M|", "global"),
                       ("(0,0) u-u   max|A - A^T|", "uu"),
                       ("(1,1) m-m   max|A - A^T|", "mm"),
                       ("(0,1)/(1,0) max|A01-A10^T|", "um_pair"),
                       ("            max|A01|", "um_size"),
                       ("            max|A10|", "mu_size"),
                       ("            max|A00|", "uu_size")]:
        print(f"  {label:<34s}{with_bcs[key]:14.4e}{without[key]:14.4e}")
    print("  Reference (review B5, 4x4 Cartesian square, no bcs): global "
          "3.2e-02, pair 1.9e-01 against 2.1e-01.")
    return {"with_bcs": with_bcs, "no_bcs": without}


def gate_v4(solver):
    """V4: the source form with `v = 1` assembles to machine zero.

    Exact by construction of the divergence form rather than `O(h^p)`, and the
    constant is a real `Function` of unit degrees of freedom so that the
    gradient is the discrete one and not UFL's symbolic zero.
    """
    value = assemble(solver.source_mass_form())
    u = solver.solution_split[solver.layout.displacement]
    # An order-of-magnitude reference: the size of the source's own integrand
    # with the constant removed. `1` is the non-dimensional length scale, so no
    # division is written.
    scale = float(solver.Lambda) * assemble(
        solver.approximation.density * sqrt(dot(u, u)) * solver.dx_m)
    print("\nV4 - discrete mass conservation of the divergence source")
    print(f"  assemble(source form, v = 1)      {value:.6e}")
    print(f"  scale  Lambda int rho_0 |u| dx    {scale:.6e}")
    print(f"  relative                          "
          f"{abs(value) / scale if scale else 0.0:.3e}")
    return {"value": value, "scale": scale}


def report(solver):
    """Diagnostics worth having every run, cheaply."""
    layout = solver.layout
    print("\nSolved")
    print(f"  ||u||                             {norm(solver.displacement):.6e}")
    print(f"  ||psi||                           {norm(solver.potential):.6e}")
    print(f"  net enclosed mass                 "
          f"{solver.total_enclosed_mass():.6e}")
    if layout.rotation:
        print(f"  rotation                          {solver.rotation_values()}")
        print(f"  inertia                           "
              f"{solver.inertia_perturbation()}")

    # The geoid at the surface, as an amplitude: the whole point of the project,
    # and the quantity `dphi` feeds into the sea-level equation.
    parent = solver.potential_mesh
    if float(solver.approximation.g) == 0.0:
        # `N = psi / g_0`, so the fluid-limit configuration, which sets `g = 0`
        # to remove the non-hydrostatic reference state, has no geoid to print.
        # Skipped rather than printed as `inf`, which reads as a defect.
        print("  geoid at Re                       not defined at g = 0")
    else:
        dS_Re = solver.form.dS(gen.CURVE_RE)
        length = assemble(avg(Constant(1.0)) * dS_Re)
        N = solver.geoid()
        X = SpatialCoordinate(parent)
        phi = atan2(X[1], X[0])
        mean = assemble(avg(N) * dS_Re) / length
        cos2 = 2 * assemble(avg(N * cos(2 * phi)) * dS_Re) / length
        print(f"  geoid at Re: mean                 {mean:.6e}")
        print(f"               cos 2phi amplitude   {cos2:.6e}")

    for bc_id, modes in solver.coefficients().items():
        strongest = max(modes.items(), key=lambda kv: abs(kv[1]))
        print(f"  boundary {bc_id}: strongest mode      "
              f"{strongest[0]} = {strongest[1]:.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr", type=float, default=0.1,
                        help="radial spacing through the mantle")
    parser.add_argument("--nazim", type=int, default=64,
                        help="azimuthal cells, divisible by 4")
    parser.add_argument("--truncation", type=int, default=5,
                        help="DtN truncation M on both boundaries")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="time step, in Maxwell times")
    parser.add_argument("--steps", type=int, default=1,
                        help="maximum number of time steps; 1 is the single "
                             "backward-Euler step the gates so far have used, "
                             "anything larger runs to the fluid limit")
    parser.add_argument("--lam-factor", type=float, default=1.0,
                        help="scale Lambda, keeping B_mu; the coupled "
                             "fluid-limit configuration is supercritical at 1 "
                             "and relaxes at 0.25")
    parser.add_argument("--rtol", type=float, default=1e-7,
                        help="steady-state tolerance on the relative change "
                             "of the surface deflection per step")
    parser.add_argument("--fluid-limit", action="store_true",
                        help="the configuration that HAS a fluid limit: g = 0 "
                             "in the approximation and an explicit Airy "
                             "restoring stress at Re; see build_solver")
    parser.add_argument("--no-nullspace", action="store_true",
                        help="do NOT declare the rigid-rotation kernel; only "
                             "useful for measuring what declaring it buys")
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--gates", action="store_true",
                        help="run G0 and V4")
    parser.add_argument("--output", action="store_true",
                        help="write two VTK files, one per mesh")
    parser.add_argument("--monitor", action="store_true",
                        help="print SNES and KSP convergence")
    args, _ = parser.parse_known_args()

    parent, sub = build_meshes(args.dr, args.nazim)
    print(f"parent cells {parent.num_cells()}, mantle cells {sub.num_cells()}")

    monitors = {"snes_monitor": None, "ksp_monitor": None,
                "snes_converged_reason": None,
                "ksp_converged_reason": None} if args.monitor else None
    solver, z, layout, bcs, C = build_solver(
        parent, sub, dt=args.dt, truncation=args.truncation,
        rotation=not args.no_rotation, solver_parameters_extra=monitors,
        declare_nullspace=not args.no_nullspace,
        fluid_limit=args.fluid_limit, lam_factor=args.lam_factor)
    print(f"fields {layout.n_fields}, dim {z.function_space().dim()}, "
          f"{len(layout.multipliers)} multipliers, "
          f"{len(layout.rotation)} rotation")
    print(f"B_mu {B_MU}  Lambda {LAMBDA * args.lam_factor}  "
          f"theta_psi {float(solver.theta_psi):.7f}")
    if layout.rotation:
        print(f"Omega^2 {OMEGA_SQ_EARTH}  "
              f"theta_rot(m3) {float(solver._theta_rot(2)):.7e}  C {C:.6f}")

    if args.gates:
        gate_g0(parent, sub, bcs, dt=args.dt)

    if args.steps > 1:
        state = run_to_steady_state(
            solver, dt=args.dt, max_steps=args.steps, rtol=args.rtol,
            every=max(1, args.steps // 20))
        if args.fluid_limit:
            # V9b's number, stated before the run rather than read off it.
            airy = -SIGMA_HAT
            print(f"  Airy fluid limit -sigma_hat/rho_0 {airy:.6e}")
            print(f"  measured                          "
                  f"{state['deflection']:.6e}   relative "
                  f"{abs(state['deflection'] - airy) / abs(airy):.3e}")
    else:
        solver.solve()
    report(solver)

    if args.gates:
        gate_v4(solver)

    if args.output:
        # A mixed function spanning two meshes cannot go into one VTKFile: the
        # displacement is the submesh's, the potential the parent's, and the
        # Real fields have no visual output at all.
        VTKFile(os.path.join(HERE, "selfgrav_gia_mantle.pvd")).write(
            solver.displacement)
        geoid = Function(
            FunctionSpace(parent, "CG", 2), name="geoid").interpolate(
                solver.geoid())
        VTKFile(os.path.join(HERE, "selfgrav_gia_parent.pvd")).write(
            solver.potential, geoid)


if __name__ == "__main__":
    main()
