"""Helpers of `tests/unit/test_gia_gravity.py` that the GIA case drivers also use.

The case drivers in `tests/gia` and the unit tests in `test_gia_gravity.py` import
these builders, meshes and constants from here, so that a driver does not
depend on a test module.
"""

import gadopt  # noqa: F401  (imported before firedrake)
import firedrake as fd
import numpy as np
from gadopt import CompressibleInternalVariableApproximation, CylindricalDtN
from gadopt.gia_couplings import CentreOfMass, FluidCore, Rotation, SurfaceLoad
from gadopt.gia_presets import OMEGA_SQ_EARTH
from gadopt.gia_solver import SelfGravitatingGIASolver
from gadopt.utility import scalar_value


# The production values of the elastic number and the self-gravity number, and
# the amplitude of the surface load.
B_MU = 1.2769
LAMBDA = 1.1116
SIGMA_HAT = 1.0e-3

# The mesh tags of the mantle cells and of the four circles, shared with the
# 3-D mesh; the radii of the CMB and the surface, the cells around and the
# layer depth of the gmsh annulus.
CELL_MANTLE = 101
CURVE_RE, CURVE_RC, CURVE_OUTER, CURVE_INNER = 2, 3, 4, 5
RC, RE = 1.2037, 2.2037
N_AZIMUTHAL = 32
DR_MANTLE = 0.2


def approximation(**kwargs):
    """A fresh approximation every time.

    The earlier solver, which other test files still build, writes `mu` into
    the approximation it is given; the new one does not, but a fresh object per
    solver keeps both safe.
    """
    settings = dict(
        bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
        g=1.0, B_mu=B_MU, self_gravity_number=LAMBDA)
    settings.update(kwargs)
    return CompressibleInternalVariableApproximation(**settings)


def gravity_bcs(parent, *, truncation=3, sheet=True):
    """The potential's conditions: DtN on both stand-off circles.

    `sheet=True` adds the `cos 2 phi` load as an `interior_sigma` on Re, the
    form the earlier solver takes. The new solver refuses a sheet here; it
    takes the load as a `SurfaceLoad` coupling, so `build` passes
    `sheet=False`.
    """
    X = fd.SpatialCoordinate(parent)
    bcs = {
        CURVE_OUTER: {"dtn": CylindricalDtN(truncation)},
        CURVE_INNER: {"dtn": CylindricalDtN(truncation)},
    }
    if sheet:
        bcs[CURVE_RE] = {
            "interior_sigma": SIGMA_HAT * fd.cos(2 * fd.atan2(X[1], X[0]))}
    return bcs


def mechanics_bcs(sub):
    """The rigid core (`un = 0` on Rc) and the `cos 2 phi` traction on Re.

    The traction is the weight `B_mu g sigma` of the load with `g = 1`, the
    form the earlier solver takes. The new solver writes that traction itself
    from a `SurfaceLoad`, and refuses a `normal_stress` on the same boundary.
    """
    Xm = fd.SpatialCoordinate(sub)
    return {
        CURVE_RC: {"un": 0.0},
        CURVE_RE: {"normal_stress":
                   B_MU * SIGMA_HAT * fd.cos(2 * fd.atan2(Xm[1], Xm[0]))},
    }


def load_density(mesh, degree=2):
    """The prescribed surface density `SIGMA_HAT cos(degree phi)`, UFL on `mesh`."""
    X = fd.SpatialCoordinate(mesh)
    return SIGMA_HAT * fd.cos(degree * fd.atan2(X[1], X[0]))


def new_fields(meshes):
    """Fresh `(u, psi)`: P2 displacement on the mantle, P2 potential on the parent.

    The degrees are those the earlier space factory used by default.
    """
    parent, sub = meshes
    u = fd.Function(fd.VectorFunctionSpace(sub, "CG", 2), name="displacement")
    psi = fd.Function(fd.FunctionSpace(parent, "CG", 2), name="potential")
    return u, psi


def polar_moment(meshes):
    """`C = int_mantle r^2 dx`, the polar moment of the unit-density mantle (2-D)."""
    parent, sub = meshes
    Xm = fd.SpatialCoordinate(sub)
    dx_m = fd.Measure("dx", domain=sub,
                      intersect_measures=(fd.Measure("dx", domain=parent),))
    return fd.assemble(fd.dot(Xm, Xm) * dx_m)


def coupling(solver, cls):
    """The one coupling of class `cls` that `solver` holds."""
    found = [c for c in solver.couplings if isinstance(c, cls)]
    assert len(found) == 1, f"{len(found)} couplings of class {cls.__name__}"
    return found[0]


def build(meshes, *, rotation=True, truncation=3, sheet=True,
          approximation_kwargs=None, declare_nullspace=False,
          dtn_representation="lowrank", couplings=(), **kwargs):
    """The annulus solver: rigid core, `cos 2 phi` load on Re, optional rotation.

    Args:
      meshes: the `(parent, sub)` pair of the module fixture.
      rotation: add the `Rotation` coupling (`m3` only, in 2-D) with the
        mantle's polar moment and the Earth's `Omega^2`.
      truncation: the `CylindricalDtN` truncation on both DtN circles.
      sheet: `True` gives the load as a `SurfaceLoad`, which loads the
        mechanics, the potential and the rotation row. `False` gives the
        traction of `mechanics_bcs` alone, a load with no mass (the solver
        warns); a test that compares with a mechanics-only solver uses it.
      approximation_kwargs: settings merged into `approximation`.
      declare_nullspace: declare the rigid rotation, `nullspace="rigid_rotation"`.
      dtn_representation: `"lowrank"` (the default of the solver) or
        `"multiplier"`.
      couplings: further couplings, appended to those above.
      kwargs: passed to `SelfGravitatingGIASolver` (`solver_parameters`,
        `solver_parameters_extra`, `scaling_factor`, `nullspace`,
        `internal_variable_family`, ...). The Quadrature family refuses the
        rigid core's `un` condition; build a fluid core for it
        (`TestFluidCore.build_fluid`).

    Returns:
      `(solver, u, psi)`, with `u` and `psi` the Functions the solver writes
      after every solve.
    """
    parent, sub = meshes
    u, psi = new_fields(meshes)
    if sheet:
        bcs = {CURVE_RC: {"un": 0.0}}
        # `g_surface = 1` writes the traction `B_mu SIGMA_HAT cos 2phi` of
        # `mechanics_bcs` and puts the same density in the potential row.
        loads = [SurfaceLoad(boundary=CURVE_RE, sigma=load_density(sub),
                             g_surface=1.0)]
    else:
        bcs = mechanics_bcs(sub)
        loads = []
    if rotation:
        # `Omega_sq` is the default of the earlier solver; this solver requires it.
        loads.append(Rotation(C=polar_moment(meshes), Omega_sq=OMEGA_SQ_EARTH))
    if declare_nullspace:
        kwargs.setdefault("nullspace", "rigid_rotation")
    solver = SelfGravitatingGIASolver(
        u, psi, approximation(**(approximation_kwargs or {})), dt=1.0,
        gravity_bcs=gravity_bcs(parent, truncation=truncation, sheet=False),
        bcs=bcs, couplings=[*loads, *couplings],
        dtn_representation=dtn_representation, **kwargs)
    return solver, u, psi


def real_block_index_sets(solver):
    """The block-1 index sets, taken the way `DtNTwoBlockSchurPC` takes them.

    `Z.field_ises[i_R:]`, i.e. Firedrake's own authority on where each
    sub-field sits in the monolithic row space, sliced at the first `Real`
    sub-field. `DtNTwoBlockSchurPC.initialize` merges exactly this slice into
    its "1" split, so a probe that reads through it exercises the
    preconditioner's own indexing rather than a second, parallel-untested one.
    """
    return solver.solution_space.field_ises[solver.real_indices[0]:]


def read_real_rows(cofunction, ises, comm):
    """The `Real` entries of an assembled residual, gathered on every rank.

    **Not `.dat.data_ro[0]`.** `demos/gravity/CLAUDE.md` records that reading an
    `R` coefficient that way bypasses the ghosting and reduction machinery, and
    the whole point of this probe is that it must mean the same thing in
    parallel as in serial - the 3-D production gate runs on 48 ranks and a probe
    that silently read a stale or unowned entry would return plausible garbage
    rather than fail.

    So the access pattern is `_RealBlockPCBase._gather`'s, term for term:
    zero-fill a buffer of the global size, write only the entries this rank
    owns, and `Allreduce`. Each `Real` sub-field carries one global degree of
    freedom, so an index set here holds one index on its owner and none
    anywhere else, and the sum-reduction is exact rather than an average.

    Run on 2 ranks: the recovered block agrees with the serial one on every
    entry that any assertion here looks at, and the only difference the rank
    count makes is the one recorded at `SERIAL`.
    """
    buf = np.zeros(len(ises))
    with cofunction.dat.vec_ro as vec:
        lo, hi = vec.owner_range
        local = vec.array_r
        for position, iset in enumerate(ises):
            indices = iset.getIndices()
            assert len(indices) <= 1, (
                f"sub-field {position} of block 1 owns {len(indices)} degrees "
                "of freedom; a Real space has exactly one, globally")
            if len(indices) == 1:
                index = int(indices[0])
                assert lo <= index < hi, (
                    f"field_ises gave sub-field {position} the global index "
                    f"{index}, outside this rank's owner range [{lo}, {hi})")
                buf[position] = local[index - lo]
    out = np.zeros(len(ises))
    comm.Allreduce(buf, out)
    return out


def real_block(solver, F=None):
    """The full `n x n` `Real` block of the Jacobian, by residual differencing.

    `F(e_j) - F(0)` read on the `Real` rows is the `j`th column exactly - the
    coupled residual is linear in every `Real` unknown, Newtonian rheology
    making the whole Jacobian constant - so this is a *measurement* of the
    block and not a finite difference with a step size to argue about. Measured
    against `mat_type="nest"` it agrees to 2.234e-16 relative.

    Differencing rather than nest assembly, and the reason is not stylistic:
    the shipped presets run `mat_type: matfree`, and asking for a nest
    Jacobian materialises every block including `(u,u)` and `(psi,psi)`, which
    is the assembled operator the configuration exists to avoid. At production
    size that is the most likely explanation for the 180 GB that killed the job
    this gate was first attempted in. Here it costs `n + 1` residual
    assemblies, and nothing else.

    Args:
      solver: the solver; its mixed solution is overwritten (left at zero).
      F: the residual to difference, `solver.F` by default.

    Returns:
      The block as a dense array, in `solver.real_indices` order.
    """
    F = solver.F if F is None else F
    z = solver.solution
    comm = solver.solution_space.mesh().comm
    ises = real_block_index_sets(solver)
    real = solver.real_indices

    z.assign(0.0)
    base = read_real_rows(fd.assemble(F), ises, comm)
    block = np.zeros((len(real), len(real)))
    for column, field in enumerate(real):
        z.assign(0.0)
        z.subfunctions[field].assign(1.0)
        block[:, column] = read_real_rows(fd.assemble(F), ises, comm) - base
    z.assign(0.0)
    return block


def predicted_real_diagonal(solver):
    """The diagonal of the `Real` block as the derivation gives it, per field.

    Written from the constants of the form, never assembled:

    * a DtN multiplier: `theta_psi * (-scale_k * A_h)`, from
      `DtNGravityForm.multiplier_diagonal`, which derives the sign and uses the
      *discrete* boundary measure `A_h`;
    * a rotation row: `theta_rot_i * K_i = s_i c B_mu Omega^2 K_i`, with
      `K = (C - A, C - A, C)`; the row divides `K_i m_i` by the volume of the
      measure it is integrated on, so the volume cancels from the entry;
    * the core pressure and the centre-of-mass multipliers: exactly zero,
      because they are Lagrange multipliers.

    `theta_psi` is read as the solver's number, never recomputed, so that the
    floor of `B_mu` at zero is the one the residual carries.

    Args:
      solver: the solver.

    Returns:
      The diagonal as an array in `solver.real_indices` order.
    """
    by_name = {}
    names = solver.field_names
    multipliers = [names[i] for i in solver.multiplier_indices]
    if multipliers:
        values = solver.theta_psi_value * solver.form.multiplier_diagonal()
        assert len(values) == len(multipliers)
        by_name.update(zip(multipliers, values))
    rotations = [c for c in solver.couplings if isinstance(c, Rotation)]
    if rotations:
        rotation, = rotations
        ctx = solver.coupling_context
        for k, name in enumerate(rotation.slots(ctx)):
            if name is None:
                continue
            by_name[name] = (rotation.CLOSURE_SIGNS[k]
                             * scalar_value(solver.scaling_factor)
                             * scalar_value(solver.row_scale_B_mu)
                             * scalar_value(rotation.Omega_sq)
                             * scalar_value(rotation.closure_constants[k]))
    for name in names:
        if name == "core_pressure" or name.startswith("centre_of_mass_"):
            by_name[name] = 0.0
    missing = [names[i] for i in solver.real_indices if names[i] not in by_name]
    assert not missing, f"no predicted diagonal entry for {missing}"
    return np.array([by_name[names[i]] for i in solver.real_indices])


class FluidCoreBuilder:
    """The annulus with a fluid core on Rc, for the tests of `FluidCore` and the cases."""

    #: The keywords of `build_fluid` that go to `FluidCore`; the rest go to
    #: the solver.
    FLUID_CORE_KEYS = ("boundary", "rho_core", "rho_mantle", "g",
                       "buoyancy_density")

    @staticmethod
    def build_fluid(meshes, *, rotation=False, sheet=True, extra_bcs=None,
                    approximation_kwargs=None, **kwargs):
        """The annulus with a fluid core on Rc and the `cos 2 phi` load on Re.

        Args:
          meshes: the `(parent, sub)` pair of the module fixture.
          rotation: add the `Rotation` coupling with the mantle's polar moment.
          sheet: `True` gives the load as a `SurfaceLoad`; `False` gives no
            load, for a test that writes its own traction in `extra_bcs`.
          extra_bcs: mechanics conditions merged into `bcs`.
          approximation_kwargs: settings merged into `approximation`.
          kwargs: `boundary`, `rho_core`, `rho_mantle`, `g` and
            `buoyancy_density` go to `FluidCore` (defaults `CURVE_RC` and
            2.0); everything else goes to `SelfGravitatingGIASolver`
            (`dtn_representation`, `internal_variable_family`,
            `solver_parameters`, `nullspace`, ...). The history family is
            the Quadrature family unless the caller names another: the
            fluid core is shared code, and the Quadrature family is the one
            that stays.

        Returns:
          `(solver, u, psi)`.
        """
        parent, sub = meshes
        kwargs.setdefault("internal_variable_family", "Quadrature")
        settings = {"boundary": CURVE_RC, "rho_core": 2.0}
        settings.update({k: kwargs.pop(k)
                         for k in FluidCoreBuilder.FLUID_CORE_KEYS if k in kwargs})
        u, psi = new_fields(meshes)
        couplings = [FluidCore(**settings)]
        if sheet:
            couplings.append(SurfaceLoad(boundary=CURVE_RE,
                                         sigma=load_density(sub),
                                         g_surface=1.0))
        if rotation:
            couplings.append(Rotation(C=polar_moment(meshes),
                                      Omega_sq=OMEGA_SQ_EARTH))
        solver = SelfGravitatingGIASolver(
            u, psi, approximation(**(approximation_kwargs or {})), dt=1.0,
            gravity_bcs=gravity_bcs(parent, sheet=False),
            bcs=dict(extra_bcs or {}), couplings=couplings, **kwargs)
        return solver, u, psi


class CentreOfMassFrame:
    """The frame configuration: fluid core, centre-of-mass multipliers, a sheet on Re.

    The reference gravity is the 2-D Gauss law for the configuration's own
    density, `g(r) = Lambda M(<r) / (2 pi r)`. With the constant `g = 1` of the
    other tests a rigid translation costs energy and the multiplier would
    carry a real force; the constraint would still hold, but the
    configuration would not be the one the frame exists for.
    """

    RHO_CORE = 2.0

    @classmethod
    def gravity_of_r(cls, r):
        """`g(r)` for a uniform core of `RHO_CORE` and a unit-density mantle."""
        mass = cls.RHO_CORE * np.pi * RC ** 2 + np.pi * (r ** 2 - RC ** 2)
        return LAMBDA * mass / (2 * np.pi * r)

    def build(self, meshes, *, load_degree=1, rotation=False,
              approximation_kwargs=None, dtn_representation="multiplier",
              **solver_kwargs):
        """The frame configuration: fluid core, degree-`load_degree` sheet on Re.

        Args:
          meshes: the module fixture `(parent, sub)`.
          load_degree: the azimuthal degree of the load sheet on Re. Degree 1
            has a mass dipole, which the frame multipliers must cancel.
          rotation: carry the rotation closure and its moment of inertia.
          approximation_kwargs: extra keywords for the approximation.
          dtn_representation: the DtN representation. `"multiplier"` by
            default, the representation the frame tests have always used.
          solver_kwargs: passed to `SelfGravitatingGIASolver`.

        Returns:
          `(solver, u, psi)`.
        """
        parent, sub = meshes
        Xm = fd.SpatialCoordinate(sub)
        rm = fd.sqrt(fd.dot(Xm, Xm))
        u, psi = new_fields(meshes)
        couplings = [
            FluidCore(boundary=CURVE_RC, rho_core=self.RHO_CORE,
                      g=fd.Constant(self.gravity_of_r(RC))),
            CentreOfMass(),
            # The weight of the sheet with the surface gravity of this
            # reference state, so that the load exerts no net force.
            SurfaceLoad(boundary=CURVE_RE,
                        sigma=load_density(sub, load_degree),
                        g_surface=self.gravity_of_r(RE))]
        if rotation:
            couplings.append(Rotation(C=polar_moment(meshes),
                                      Omega_sq=OMEGA_SQ_EARTH))
        solver = SelfGravitatingGIASolver(
            u, psi,
            approximation(g=self.gravity_of_r(rm), **(approximation_kwargs or {})),
            dt=1.0, gravity_bcs=gravity_bcs(parent, sheet=False), bcs={},
            couplings=couplings, dtn_representation=dtn_representation,
            **{"internal_variable_family": "Quadrature", **solver_kwargs})
        return solver, u, psi
