r"""This module contains classes that augment default Firedrake preconditioners.

"""

import functools
import sys

import firedrake as fd
import numpy as np

try:
    from scipy.linalg import lu_factor as _lu_factor, lu_solve as _lu_solve
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy ships in the firedrake venv
    _HAVE_SCIPY = False
from ufl import as_vector as ufl_as_vector
from ufl import Form as ufl_Form
from ufl.algorithms import expand_derivatives
from ufl.indexed import Indexed
from firedrake.dmhooks import get_function_space
from firedrake.petsc import PETSc
from mpi4py import MPI
from firedrake.assemble import get_assembler
from firedrake import dmhooks
from firedrake.slate import AssembledVector
from firedrake.slate.static_condensation.la_utils import LAContext
from .nullspaces import near_incompressible_modes, rigid_body_modes
from .utility import InteriorBC


class FreeSurfaceMassInvPC(fd.MassInvPC):
    """Version of MassInvPC that includes free surface variables."""

    def form(
        self,
        pc: fd.PETSc.PC,
        tests: list[fd.Argument | Indexed],
        trials: list[fd.Argument | Indexed | fd.Function],
    ) -> tuple[fd.Form, list[fd.DirichletBC]]:
        """Sets the form.

        Args:
          pc:
            PETSc preconditioner
          tests:
            List of Firedrake test functions
          trials:
            List of Firedrake trial functions
        """
        appctx = self.get_appctx(pc)

        # N.B. trials[0] is pressure
        mu = appctx.get("mu", 1.0)
        a = fd.inner(1 / mu * trials[0], tests[0]) * fd.dx

        ds = appctx["ds"]
        bcs = []
        for bc_id, (eta_ind, _) in appctx["free_surface"].items():
            a += 1 / mu * fd.inner(trials[eta_ind - 1], tests[eta_ind - 1]) * ds(bc_id)

            bcs.append(InteriorBC(trials.function_space()[eta_ind - 1], 0, bc_id))

        return a, bcs


def near_nullspace_basis(V, modes: str, *, max_degree: int = 1,
                         divfree_tol: float = 1e-8):
    """The near-nullspace named by `modes`, on the function space `V`.

    Split out of `SPDAssembledPC.initialize` so that the name can be validated
    without a PETSc `PC` in hand. That matters because a `ValueError` raised
    inside a python PC callback does not reach the caller as itself - see
    `_RealBlockPCBase._loud`.

    Args:
      V: the function space the modes are built on.
      modes: ``"rigid"`` for the six rigid-body modes, or ``"incompressible"``
        for those plus the low-degree divergence-free fields.
      max_degree, divfree_tol: passed to `near_incompressible_modes`; ignored
        for ``"rigid"``.

    Returns:
      A Firedrake `VectorSpaceBasis`.

    Raises:
      ValueError: if `modes` is not a recognised name. ``"none"`` is handled by
        the caller, which skips this function entirely.
    """
    if modes == "rigid":
        return rigid_body_modes(
            V, rotational=True, translations=list(range(V.value_size)))
    if modes == "incompressible":
        return near_incompressible_modes(
            V, max_degree=max_degree, divfree_tol=divfree_tol)
    raise ValueError(
        f"near_nullspace must be 'none', 'rigid' or 'incompressible', "
        f"not {modes!r}")


class _AssembledBlockPC(fd.AssembledPC):
    """`AssembledPC` with an SPD flag and a selectable near-nullspace.

    This class carries two settings that are **independent of each other**, and
    it exists in this shape because they used to be tangled together.

    **The SPD flag.** Setting the PETSc MatOption MAT_SPD (for Symmetric
    Positive Definite matrices) switches the Krylov method used for eigenvalue
    estimates to CG - both in the Chebyshev smoothers and in GAMG's
    smoothed-aggregation setup - and propagates the SPD flag to the coarse-grid
    operators. All of these are benign for a genuinely SPD block, and wrong for
    a block that is not. Use it on a fieldsplit_0 block: the Stokes velocity
    block, or the gravity potential block.

    **The near-nullspace.** GAMG builds coarse spaces that reproduce whatever
    near-nullspace it is handed, so the choice must match the block's slow
    modes:

    ``"none"``
        Smoothed aggregation alone.
    ``"rigid"``
        The six rigid-body modes. The correct choice for an elasticity block at
        a moderate bulk/shear ratio.
    ``"incompressible"``
        The rigid-body modes **and** the low-degree divergence-free fields, from
        `near_incompressible_modes`. A strict superset of ``"rigid"``. Use it
        when the volumetric penalty
        $\\lambda\\int(\\nabla\\cdot u)(\\nabla\\cdot v)$ makes the
        divergence-free space the slow one, which happens as the bulk/shear
        ratio grows and equally at any large $dt/\\tau$.

    **Why the modes are built here and not passed in.** A `near_nullspace`
    supplied on an outer mixed space never reaches GAMG underneath
    `DtNTwoBlockSchurPC`, and nothing says so. Firedrake composes the basis onto
    the outer space's field index sets and `PCFIELDSPLIT` reads it back by
    querying those index sets
    (`src/ksp/pc/impls/fieldsplit/fieldsplit.c:721`, in `PCSetUp_FieldSplit`).
    `DtNTwoBlockSchurPC` registers *merged* index sets of its own, and the
    nested split inside block 0 builds fresh ones from a sub-DM, so the query
    matches nothing and the modes are silently dropped: no error, no warning,
    and the only symptom is GAMG coarsening an elasticity block on smoothed
    aggregation alone. This class stops relying on propagation. It asks its own
    DM for its own function space, builds the modes there *after* `AssembledPC`
    has assembled the block, and hangs them on the assembled matrix where GAMG
    will actually look.

    Both settings are read as PETSc options under this PC's prefix, so a driver
    can now ask for **any** combination::

        "..._pc_python_type": "gadopt.SPDAssembledPC",
        "..._spd": True,                    # default for this class
        "..._near_nullspace": "rigid",      # "none" | "rigid" | "incompressible"

    and the ``"incompressible"`` mode set takes two more::

        "..._solenoidal_max_degree": 1      # the default (before filtering)
        "..._solenoidal_divfree_tol": 1e-8  # relative L2 divergence cut

    Degree 1 already spans the complete space of linear divergence-free fields;
    raising it saturates GAMG's aggregates and can drive a `DIVERGED_NANORINF`
    unless the aggregate size is raised too - see `near_incompressible_modes`.

    `SPDAssembledPC`, `RigidBodyAssembledPC` and `NearlyIncompressibleAssembledPC`
    are siblings that only change the defaults below. They are siblings rather
    than a chain because a name asserting a property must not be a base for a
    class that switches it off: `RigidBodyAssembledPC` does not set MAT_SPD, and
    inheriting it from something called `SPDAssembledPC` would say otherwise.
    Keep using all three by name; every caller reaches them through the *string*
    in a solver-options dictionary, and a broken name there surfaces as
    `PETSc.Error: error code 101` naming nothing.

    Note:
      The two mode-carrying names default to ``spd = False``, which is what they
      did before this class absorbed them. That is deliberate: the
      published iteration counts for the near-incompressible arms were measured
      without the flag, and turning it on by default would silently invalidate
      them. Setting ``"..._spd": True`` on those subclasses is now a one-line
      change, and is expected to help on a genuinely symmetric block. Check that
      the block *is* symmetric first - a Nitsche or prestress-advection term can
      break the symmetry that MAT_SPD asserts.
    """

    #: Default for the MAT_SPD flag; overridden by the ``spd`` PETSc option.
    _spd = True
    #: Default mode set; overridden by the ``near_nullspace`` PETSc option.
    _near_nullspace = "none"
    _default_max_degree = 1
    _default_divfree_tol = 1e-8

    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        opts = PETSc.Options(pc.getOptionsPrefix() or "")
        mat = self.P.petscmat

        self._spd_wanted = opts.getBool("spd", self._spd)
        if self._spd_wanted:
            mat.setOption(mat.Option.SPD, True)

        modes = opts.getString("near_nullspace", self._near_nullspace)
        if modes == "none":
            return

        try:
            basis = near_nullspace_basis(
                get_function_space(pc.getDM()).collapse(), modes,
                max_degree=opts.getInt(
                    "solenoidal_max_degree", self._default_max_degree),
                divfree_tol=opts.getReal(
                    "solenoidal_divfree_tol", self._default_divfree_tol),
            )
        except ValueError as exc:
            # PETSc flattens a Python exception raised inside a python PC, so
            # the message has to be printed before it is raised - see
            # `_RealBlockPCBase._loud`, which measured that behaviour.
            raise _RealBlockPCBase._loud(exc)
        mat.setNearNullSpace(basis.nullspace())

    def update(self, pc: PETSc.PC):
        """Re-assembles the block, then puts the SPD flag back.

        **`MatAssemblyEnd` clears MAT_SPD** unless MAT_SPD_ETERNAL was set --
        PETSc `src/mat/interface/matrix.c:6345`,
        `if (!mat->spd_eternal) mat->spd = PETSC_BOOL3_UNKNOWN;`. The base class
        re-assembles into the same matrix on every update, so without this the
        flag is live for the first solve and absent from every one after it, and
        nothing says so. petsc4py exposes no `SPD_ETERNAL` option, so the flag is
        simply set again.

        The near-nullspace needs no such repair: it is an attribute of the
        matrix rather than a state flag, so re-assembly leaves it in place, and
        rebuilding the modes here would pay for them on every timestep.
        """
        super().update(pc)
        if getattr(self, "_spd_wanted", False):
            mat = self.P.petscmat
            mat.setOption(mat.Option.SPD, True)


class SPDAssembledPC(_AssembledBlockPC):
    """`_AssembledBlockPC` with the MAT_SPD flag and no near-nullspace.

    The historical name, and the defaults it always had. Equivalent to
    ``spd = True``, ``near_nullspace = "none"``. See `_AssembledBlockPC` for both
    settings and for how to override either from a solver-options dictionary.

    Use it on a block that is genuinely symmetric positive definite: the Stokes
    velocity block, or the gravity potential block, whose shift makes it
    strictly SPD.
    """

    _spd = True
    _near_nullspace = "none"


class RigidBodyAssembledPC(_AssembledBlockPC):
    """`SPDAssembledPC` defaulting to the six rigid-body modes, without MAT_SPD.

    Equivalent to `_AssembledBlockPC` with ``near_nullspace = "rigid"`` and
    ``spd = False``. See that class for why the modes are built from this PC's
    own block rather than propagated in, and for how to override either setting.

    Select it by name on whichever split holds the displacement::

        "dtn_fieldsplit_0_fieldsplit_0_pc_python_type":
            "gadopt.RigidBodyAssembledPC",

    **Naming `firedrake.AssembledPC` there instead is not a milder choice, it
    is the defect**: the block still assembles, the solve still converges, and
    the near-nullspace the caller declared is simply absent from GAMG's setup.
    """

    _spd = False
    _near_nullspace = "rigid"


class NearlyIncompressibleAssembledPC(_AssembledBlockPC):
    r"""`AssembledPC` seeding GAMG with rigid-body *and* divergence-free modes.

    The compressible internal-variable stress carries a volumetric penalty
    $\lambda\int(\nabla\cdot u)(\nabla\cdot v)$ with $\lambda =
    "bulk_shear_ratio"\times"bulk_modulus"$ (`approximations.py`,
    `InternalVariableApproximation.stress`). As the ratio grows -- towards the
    incompressible limit, and equally at any large $dt/\tau$, where the
    eliminated internal variable leaves the shear stiffness incremental while
    the volumetric one is untouched -- the operator's slow modes migrate into
    the **divergence-free** space. GAMG coarsens the slow modes onto whatever
    near-nullspace it is given, and the six rigid-body modes
    (`RigidBodyAssembledPC`) do not span that space, so smoothed aggregation
    stalls: measured on the coupled 3-D solve as `DIVERGED_ITS` at
    $K/\mu = 100$ and a walltime kill at $K/\mu = 1000$.

    This subclass hands GAMG `near_incompressible_modes` -- the rigid modes plus
    the low-degree divergence-free fields -- built from its *own* block after
    assembly, for the same reason `RigidBodyAssembledPC` does: a `near_nullspace`
    declared on an outer mixed space never reaches a nested GAMG (see that
    class). It is a strict superset of `RigidBodyAssembledPC`; where a plain
    `firedrake.AssembledPC` is used on the displacement block with no
    near-nullspace at all, it is a superset of that too.

    The number of divergence-free fields is set by the highest polynomial
    degree they carry, read as an integer PETSc option under this PC's prefix,
    with the candidates then filtered so only those still solenoidal on the
    actual (possibly curved) mesh survive -- see `near_incompressible_modes`::

        "..._solenoidal_max_degree": 1      # the default (before filtering)
        "..._solenoidal_divfree_tol": 1e-8  # relative L2 divergence cut

    Degree 1 already spans the complete space of linear divergence-free fields;
    raising it saturates GAMG's aggregates and can drive a `DIVERGED_NANORINF`
    unless the aggregate size is raised too -- see `near_incompressible_modes`.

    A finite mode set cannot span the whole (infinite-dimensional)
    divergence-free space, so this is expected to *reduce* rather than remove
    the ratio penalty; it is the cheap, adjoint-safe probe that bounds how much
    of the near-incompressibility failure is a missing near-nullspace as opposed
    to genuine saddle-point ill-conditioning, before committing to an
    augmented-Lagrangian block solve or a mixed (u, p) reformulation. Being a
    preconditioner only, it changes no residual and so leaves the taped adjoint
    untouched.

    Select it by name on whichever block holds the displacement, e.g. the
    single-field segregated solve::

        "pc_type": "python",
        "pc_python_type": "gadopt.NearlyIncompressibleAssembledPC",

    or the displacement split of the coupled solver, in place of
    `gadopt.RigidBodyAssembledPC`.
    """

    _spd = False
    _near_nullspace = "incompressible"


class DtNTwoBlockSchurPC(fd.PCBase):
    """Schur fieldsplit whose two blocks are described by index sets.

    The gravitational Poisson solver gives every treated angular mode its own
    scalar Real-space sub-field, so its mixed space carries 1 + n fields (plus
    a cross-mesh dummy where one is used). PETSc enumerates the sub-fields of
    the DM before grouping them and refuses more than 128 of them
    (`PCFieldSplitSetDefaults`: "Cannot currently support N > 128 fields"),
    which caps the coupled DtN solve at a truncation of L = 6 on a two-boundary
    shell. That enumeration is only reached when no split has been registered
    yet, so registering the two blocks directly as index sets - the potential
    (and dummy) degrees of freedom in one, every multiplier degree of freedom
    in the other - bypasses it, leaving the Schur factorisation, the potential
    block solver and the multiplier solve exactly as they are below the cap.
    Only the description of which degrees of freedom belong to which block
    changes; the weak form, the mixed space and the taped variational problem
    are untouched, so the adjoint is unaffected.

    The two blocks are found by introspecting the mixed space of the operator
    for its Real sub-fields, never from the application context, which pyadjoint
    drops from the kwargs of the adjoint solve on the multiplier path (the
    low-rank path supplies it through `LowRankVariationalSolver`'s
    `adj_kwargs`, and this class does not need it there either).

    Options for the inner fieldsplit are read under a `dtn_` prefix, e.g.
    `Gravity_dtn_fieldsplit_0_ksp_type`. Never supply `pc_fieldsplit_%d_fields`
    options there: the splits are already defined and the field lists would be
    silently ignored.

    Users can provide this class as a `pc_python_type` entry to a PETSc solver
    option dictionary; the preconditioning operator must be matrix-free.

    ## `update` is a no-op, and a state-dependent Jacobian is still followed

    `update` does nothing because the only thing this class owns is the pair of
    index sets, and those do not change. That is not the same as freezing the
    nested preconditioner. The inner fieldsplit is a separate PETSc object that
    keeps its own setup state against the preconditioning matrix's object
    state, and `PCApply` calls `PCSetUp` on it first
    (`petsc/src/ksp/pc/interface/precon.c:542`). Every Newton iteration
    assembles the matrix-free Jacobian, `MatAssemblyEnd` moves the outer `Mat`'s
    object state, and `PCSetUp_FieldSplit` therefore re-extracts its sub-matrices
    with `MAT_REUSE_MATRIX`; Firedrake's `createSubMatrix` assembles into the
    reused sub-matrix, which moves every sub-matrix's state in turn, so every
    sub-preconditioner's `update` runs once per Newton iteration.

    Measured on the 2-D annulus at exponent 3 and transition stress 1e-3
    (`NOTES/PLAN-POWER-LAW-SELFGRAVITY.md` section 2): `InternalVariableSCPC`,
    `SPDAssembledPC` on the potential block and `DtNMultiplierDenseSchurPC` are
    all updated at every Newton iteration, and `InternalVariableSCPC.
    assembly_count` equals the Newton iteration count of the solve. So a
    state-dependent Jacobian is a supported configuration of this class, and
    `SelfGravitatingGIASolver` runs a power law on it.

    Two consequences worth knowing before tuning anything:

    - **`snes_lag_preconditioner` and `ksp_reuse_preconditioner` reach only this
      python PC.** `PCSetReusePreconditioner` sets a flag on the outer PC alone
      (`precon.c:1338-1346`) and `PCPYTHON` composes no
      `PCSetReusePreconditioner_C`, so the inner fieldsplit sees only its own
      flag and re-runs its setup inside `apply` whenever the Jacobian's object
      state changed, whatever lag the outer PC carries. Measured: with
      `snes_lag_preconditioner -2` the outer `update` never runs again and
      `InternalVariableSCPC.update` still runs at every Newton iteration.
      `snes_lag_jacobian` is the setting that lags the nested preconditioner,
      because it stops the state bump at its source.
    - **Every assembled sub-block is reassembled at every Newton iteration**,
      the constant potential block included: `AssembledPC.update` reassembles
      unconditionally. That work is correct and wasted, and how much it costs
      at production rank counts belongs to the 3-D measurement (parent plan
      S4).

    The gravitational Poisson solver is the special case where nothing is ever
    rebuilt, because its Jacobian is constant by construction (see `update`).
    """

    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        A, P = pc.getOperators()
        ctx = P.getPythonContext()
        W = ctx.a.arguments()[0].function_space()

        real = [i for i, V in enumerate(W)
                if V.ufl_element().family() == "Real"]
        if not real:
            raise ValueError(
                f"{type(self).__name__} needs Real sub-fields to split off, "
                "but the mixed space has none; use a plain fieldsplit.")
        i_R, n = real[0], len(real)
        # Anything but a contiguous trailing run of Real sub-fields would
        # leave sub-fields out of both blocks - a silently wrong split.
        if real != list(range(i_R, len(W))):
            raise ValueError(
                f"{type(self).__name__} requires the Real sub-fields to be "
                f"contiguous and last, but sub-fields {real} of {len(W)} are "
                "Real.")
        if i_R == 0:
            raise ValueError(
                f"{type(self).__name__} requires at least one non-Real "
                "sub-field to form the first block, but sub-field 0 is Real.")

        # field_ises is Firedrake's own authority on where each sub-field lives
        # in the monolithic row space, and the merged sets are exact in-order
        # concatenations of them, which is what lets the matrix-free submatrix
        # extraction recognise them as whole fields.
        field_ises = W.dof_dset.field_ises

        def merge(ises):
            indices = np.concatenate([iset.getIndices() for iset in ises])
            return PETSc.IS().createGeneral(
                indices.astype(PETSc.IntType), comm=pc.comm)

        inner = PETSc.PC().create(comm=pc.comm)
        inner.incrementTabLevel(1, parent=pc)
        inner.setOptionsPrefix((pc.getOptionsPrefix() or "") + "dtn_")
        inner.setOperators(A, P)
        inner.setType(PETSc.PC.Type.FIELDSPLIT)
        inner.setFieldSplitIS(("0", merge(field_ises[:i_R])),
                              ("1", merge(field_ises[i_R:i_R + n])))
        inner.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        inner.setFromOptions()
        inner.setUp()

        # Pre-registering the index sets skips the branch of
        # PCFieldSplitSetDefaults that would otherwise hand each sub-KSP a
        # sub-DM, and AssembledPC (hence SPDAssembledPC) resolves its function
        # space and its split solver context through that DM. Thread it on by
        # hand: createSubDM invokes Firedrake's own hook, which pushes the
        # split context onto the new DM as a side effect. The DM is left
        # inactive so that KSPSetUp does not try to build operators from it.
        ksp_potential, ksp_real = inner.getFieldSplitSchurGetSubKSP()
        _, subdm = pc.getDM().createSubDM(list(range(i_R)))
        ksp_potential.setDM(subdm)
        ksp_potential.setDMActive(PETSc.KSP.DMActive.ALL, False)
        # **And the same for the multiplier KSP when its PC uses the DM.** A
        # `PCBase` subclass resolves its appctx through `pc.getDM()` and needs
        # the split context. `pc_type: none` does not use a DM and must not ask
        # Firedrake to split one: a taped linear residual can contain
        # `Action(MatrixBase, u)`, which Firedrake cannot split when `u` is a
        # cross-mesh Real field. Both shipped low-rank presets use `none` here.
        #
        # Without the DM on an active block-1 PC, the recorded symptom is
        # `AttributeError: 'NoneType' object has no attribute 'appctx'`. It does
        # not name this preconditioner or the faulty block. Reverting these
        # three lines and running
        # `tests/unit/test_dtn_multiplier_pc.py::TestTheSolveAgrees` gives a
        # SEGMENTATION FAULT (exit 139), not a Python exception**. This DM makes
        # `DtNMultiplierDiagPC` available when the caller selects it.
        if ksp_real.getPC().getType() != PETSc.PC.Type.NONE:
            _, subdm_real = pc.getDM().createSubDM(
                list(range(i_R, len(W))))
            ksp_real.setDM(subdm_real)
            ksp_real.setDMActive(PETSc.KSP.DMActive.ALL, False)

        self.pc = inner

    def update(self, pc: PETSc.PC):
        """Updates the preconditioner state; nothing to do here.

        This class owns the two index sets and nothing else, and the index sets
        are a property of the mixed space, so a repeated setup can only rebuild
        what is already held. Everything that does depend on the operator lives
        in the inner fieldsplit, which keeps its own setup state and rebuilds
        itself inside `apply`; the class docstring gives the mechanism and the
        measurement. So this no-op is correct for a state-dependent Jacobian as
        well as for a constant one.

        For the gravitational Poisson solver nothing anywhere is rebuilt,
        because that Jacobian is constant by construction: the density and the
        gravitational constant enter the residual only through terms linear in
        the test function, so they vanish under differentiation, and every
        remaining coefficient (the Robin shift, the DtN eigenvalues and the
        constraint-row scalings) is fixed when the form is built.

        Args:
          pc: PETSc preconditioner.
        """
        pass

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """Applies the inner fieldsplit.

        Args:
          pc: PETSc preconditioner.
          x: Vector the preconditioner is applied to.
          y: Vector receiving the result.
        """
        self.pc.apply(x, y)

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """Applies the transpose of the inner fieldsplit.

        Args:
          pc: PETSc preconditioner.
          x: Vector the preconditioner is applied to.
          y: Vector receiving the result.
        """
        self.pc.applyTranspose(x, y)

    def view(self, pc: PETSc.PC, viewer=None):
        """Prints a description of the preconditioner.

        Args:
          pc: PETSc preconditioner.
          viewer: PETSc viewer.
        """
        super().view(pc, viewer)
        # The base class quietly returns on a missing or non-ASCII viewer, so
        # repeat its test before writing anything of our own.
        if viewer is None or viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        if hasattr(self, "pc"):
            viewer.printfASCII("Two-block Schur fieldsplit defined by index sets\n")
            self.pc.view(viewer)


class _RealBlockPCBase(fd.preconditioners.base.PCBase):
    """Shared plumbing for the two multiplier preconditioners.

    The multiplier block is tiny -- 72 `Real` fields at L = 5, 75 with rotation
    -- so the linear algebra is done in numpy redundantly on every rank, which
    for a 75x75 problem is free and avoids needing a parallel dense solver.

    **The arithmetic here is layout-agnostic and nothing in it assumes where
    the `Real` dofs live.** `_gather` zero-fills, writes only its own
    `[lo:hi]` slice and `Allreduce`s; `apply` writes only the local owned slice
    of the redundantly-computed solution. Both are correct under *any*
    distribution, including a future Firedrake that scatters the `Real` block
    across ranks. That today every `Real` dof happens to sit on rank 0
    (measured: at 48 ranks the block-1 index set is 75 contiguous entries all
    owned by rank 0) is **descriptive performance context, not a load-bearing
    invariant** -- recorded that way so nobody "fixes" correct code.

    The one genuine ordering dependency is global-dof order against
    `multiplier_keys` order, which the merged index set supplies and which
    `GIASpaceLayout.real_fields` pins.

    Neither ever asks anyone to assemble or differentiate anything on the
    `Real` space, which is what lets them work where `jacobi`, `selfp`,
    `AssembledPC` and `-pc_fieldsplit_schur_precondition full` all fail. Those
    four are dead for reasons that are properties of the *route*, not of the
    object; see `DtNTwoBlockSchurPC` and the record in the B2 probe.
    """

    def initialize(self, pc):
        raise NotImplementedError

    def update(self, pc):
        """The base does nothing; a subclass whose data moves overrides this.

        `DtNMultiplierDiagPC` needs no rebuild under any rheology. Its data is
        the exact block-1 diagonal, built from three contributions and none of
        them a function of the displacement or the internal variables:
        `theta_psi` (the row scaling, made of `scaling_factor`, `B_mu` and
        `Lambda`), the DtN mode scale times the discrete boundary area, and the
        rotation rows' `_theta_rot * _closure_constant` (made of `Omega_sq` and
        the rotation moments). So the diagonal is the same matrix at every
        Newton iteration of a power-law solve as it is at the first.

        `DtNMultiplierDenseSchurPC` does have data that moves, because its
        complement is formed from the block-0 operator, and it overrides this
        method with its own rebuild rule.

        Args:
          pc: PETSc preconditioner.
        """

    @staticmethod
    def _loud(exc):
        """Write the message to stderr, then hand the exception back to raise.

        **PETSc flattens a Python exception raised inside a python PC into
        `PETSc.Error: error code 101`** -- measured, on exactly the misuse this
        class most expects. The exception object is then useless to whoever
        reads the log, because the text never reaches them. Printing it first
        puts the named cause directly above the 101 in the output, which is the
        only place it can still be read.

        Returns the exception rather than raising it, so the call site keeps
        `raise` and static analysis still sees a raise.
        """
        print(f"\n[{__name__}] {type(exc).__name__}: {exc}\n",
              file=sys.stderr, flush=True)
        return exc

    def _gather(self, comm, vec, n_global):
        buf = np.zeros(n_global)
        lo, hi = vec.owner_range
        buf[lo:hi] = vec.array_r
        out = np.zeros(n_global)
        comm.Allreduce(buf, out)
        return out

    def apply(self, pc, x, y):
        self._apply(pc, x, y, self._solve)

    def applyTranspose(self, pc, x, y):
        """Precondition with S^T, not S.

        The adjoint inner solve calls `applyTranspose`. For a symmetric block --
        the exact diagonal -- S^T = S and the default forwards to `_solve`. A PC
        whose block is asymmetric -- the dense Schur complement, whose relative
        asymmetry is ~0.34 by construction -- must override `_solve_transpose` to
        solve S^T x = b. If it does not, the adjoint solve is preconditioned by
        the wrong operator: still a valid preconditioner in the sense that it
        changes no residual, but a poor one, so the outer Krylov pays for it.
        """
        self._apply(pc, x, y, self._solve_transpose)

    def _apply(self, pc, x, y, solve):
        comm = pc.comm.tompi4py()
        rhs = self._gather(comm, x, self._n)
        sol = solve(rhs)
        lo, hi = y.owner_range
        y.array_w[:] = sol[lo:hi]

    def _solve(self, rhs):
        raise NotImplementedError

    def _solve_transpose(self, rhs):
        """Solve S^T x = rhs. A symmetric block inherits the forward solve.

        Backward-compatible default: `DtNMultiplierDiagPC`'s block is diagonal,
        hence symmetric, so its transpose solve is its forward solve unchanged.
        """
        return self._solve(rhs)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is not None:
            viewer.printfASCII(f"{type(self).__name__} on {self._n} Real rows\n")


class DtNMultiplierDiagPC(_RealBlockPCBase):
    r"""Invert the multiplier block's diagonal exactly. It is diagonal, and known.

    **Opt-in. This is not any preset's default and must not be made one** --
    both shipped presets run block 1 at `pc_type: none`, and flipping that would
    silently move every number the current campaign is producing. Select it by
    name::

        "dtn_fieldsplit_1_pc_type": "python",
        "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDiagPC",

    The `(c, c)` entries are exact and readable straight off the form as
    `-theta_psi * scale_k * A_h` (`DtNGravityForm.multiplier_diagonal`, which
    derives the sign and the discrete area there), with no assembly and no
    `MatGetDiagonal`. Setup cost is therefore *nothing* -- no block-0 solves, no
    factorisation -- which is what makes this the one to reach for first.

    **The diagonal is independent of the rheology**, so this preconditioner
    serves a power law with no rebuild rule and no refusal. Its three
    contributions are `theta_psi` (from `scaling_factor`, `B_mu` and `Lambda`),
    the mode scale times the discrete boundary area, and the rotation rows'
    closure constant (from `Omega_sq` and the rotation moments); none of them
    is a function of the displacement or of the internal variables, so the
    exact diagonal at Newton iteration k is the exact diagonal at iteration 0.

    What it misses is the Schur correction `C A00^{-1} B`, the DtN feedback
    through the potential, which is small for a boundary stood off to 2 Re and
    smaller for higher modes. So it is an approximate inverse of S, not an
    exact one, and it is only ever used as a preconditioner.

    ## Scoped to `SelfGravitatingGIASolver`. Not for `GravitySolver`.

    `DtNTwoBlockSchurPC` serves two solvers and **their block-1 rows carry
    different scalings**: the coupled solver multiplies every constraint row by
    `theta_psi`, `GravitySolver` leaves them unscaled. So one diagonal cannot
    serve both, and if this ever becomes a default it must be wired **per
    solver**, not on the shared preconditioner.

    Two independent reasons to leave the gravity-alone path at `pc_type: none`,
    both measured rather than argued:

    1. **The diagonal differs.** `gadopt/gravity_solver.py` applies no row
       scaling, so its `(c,c)` entries are `-scale_k * A_h` with no `theta_psi`.
       Using the coupled diagonal there would be wrong by that factor.
    2. **That path owns the shipped, verified adjoint**
       (`tests/unit/test_gravity_adjoint.py`), and it is *not* the deferred
       one -- while this class takes its diagonal from the **appctx**, which
       `DtNTwoBlockSchurPC`'s own docstring records pyadjoint **dropping from
       the kwargs of the adjoint solve** on that path (the low-rank path
       supplies it through `LowRankVariationalSolver`'s `adj_kwargs`; the
       multiplier path does not). That is precisely why that class
       introspects the operator instead of using appctx. An appctx-carried
       diagonal is therefore the wrong mechanism on the one path with a live
       adjoint, and no guard in this class can see the difference: the count is
       right, the forward solve is right, and the failure appears only on a
       taped replay.

    `GravitySolver` supplies no `dtn_block1_diagonal` at all, so selecting this
    preconditioner there fails rather than degrading -- but **measured, it
    fails as a bare `PETSc.Error: error code 101`**, naming neither this class
    nor the missing key, because the `ValueError` below is raised inside a
    python PC and PETSc flattens it. That is the same unhelpful failure mode
    recorded elsewhere in this project, and it is another reason the wiring
    must be per solver rather than shared. Anyone wiring it for that solver must supply the **unscaled**
    diagonal and must first establish what happens to appctx under `pyadjoint`.

    For the coupled solver the same appctx caveat applies to *its* adjoint,
    which is deferred by decision. The failure mode is a loud `ValueError`
    naming the trap, never a silent zero.

    The diagonal is taken from the appctx under `dtn_block1_diagonal`, which
    the coupled solver supplies; reaching it requires the sub-DM this module
    now threads onto the multiplier KSP.

    Measured on the 3-D coarse coupled system: block-0 applications 111 -> 48,
    wall time 1.8x. Those figures come from the B2 probe and are **not** yet
    reproduced by a shipped-code verification run.
    """

    _prefix = "dtn_multiplier_diag_"

    def initialize(self, pc):
        appctx = self.get_appctx(pc)
        diag = appctx.get("dtn_block1_diagonal")
        if diag is None:
            raise self._loud(ValueError(
                "DtNMultiplierDiagPC needs the block-1 diagonal in the "
                "appctx under 'dtn_block1_diagonal'. SelfGravitatingGIASolver "
                "supplies it; a caller assembling this system by hand must "
                "pass appctx={'dtn_block1_diagonal': ...} with one entry per "
                "Real sub-field: DtN multipliers, optional core pressure, then "
                "any rotation rows.\n"
                "Two common causes, and the second is not a mistake in your "
                "options:\n"
                "  * this is GravitySolver, which supplies no diagonal and "
                "whose rows are UNSCALED - see the class docstring; that path "
                "is deliberately left at pc_type: none.\n"
                "  * this is an ADJOINT solve on the multiplier path. pyadjoint "
                "drops appctx from the kwargs of that adjoint solve (see "
                "DtNTwoBlockSchurPC), so a preconditioner that reads appctx "
                "cannot be used on its taped replay; the low-rank path "
                "supplies the context through LowRankVariationalSolver's "
                "adj_kwargs. This is a loud failure by design; it must never "
                "become a silent zero diagonal."))
        A, _ = pc.getOperators()
        self._n = A.getSizes()[0][1]
        diag = np.asarray(diag, dtype=float)
        if diag.size != self._n:
            raise self._loud(ValueError(
                f"the multiplier block is {self._n} wide but the supplied "
                f"diagonal has {diag.size} entries. Include the optional core "
                "pressure and rotation rows exactly when they are in the mixed "
                "space."))
        if np.any(diag == 0.0):
            zero = np.flatnonzero(diag == 0.0).tolist()
            raise self._loud(ValueError(
                f"the block-1 diagonal contains a zero entry at positions "
                f"{zero}, so DtNMultiplierDiagPC cannot invert it. A fluid-core "
                "pressure row has a physical zero diagonal; use pc_type none or "
                "gadopt.DtNMultiplierDenseSchurPC for that saddle system. A "
                "zero DtN or rotation diagonal is a bug upstream."))
        self._d = diag

    def _solve(self, rhs):
        return rhs / self._d


class DtNMultiplierDenseSchurPC(_RealBlockPCBase):
    r"""Form the whole multiplier Schur complement once at setup and factor it.

    **Opt-in, exactly like `DtNMultiplierDiagPC`. This is not any preset's
    default and must not be made one** -- both shipped presets run block 1 at
    `pc_type: none`, and flipping that would silently move every number the
    current campaign is producing. Select it by name on the multiplier block::

        "dtn_fieldsplit_1_pc_type": "python",
        "dtn_fieldsplit_1_pc_python_type": "gadopt.DtNMultiplierDenseSchurPC",

    with a full Schur factorisation above it
    (`dtn_pc_fieldsplit_schur_fact_type: full`), so that the `Amat` this PC is
    handed is PETSc's `MATSCHURCOMPLEMENT`. **One `A.mult(e_k)` is then one
    application of the Schur complement S**, and the block is only 72 columns
    wide at L = 5 (75 with rotation), so that many `MatMult`s build S entirely,
    each costing one block-0 solve. Contract the columns into a dense array on
    every rank, factor it once, and apply the factors from then on. No Firedrake
    assembly is involved anywhere, which is why this works where `jacobi`,
    `selfp`, `AssembledPC` and `-pc_fieldsplit_schur_precondition full` all fail
    on the `Real` block -- the same reason `DtNMultiplierDiagPC` above works.

    Unlike the diagonal PC, this one takes **no data** from the appctx. It needs
    no `dtn_block1_diagonal`, no module global and no per-solver wiring: it forms
    S purely from the operator handed to it. That is the design rule this project
    settled on -- a preconditioner that reads its data off the operator has no
    appctx failure mode, so it works where the diagonal PC raises: inside the
    multiplier path's adjoint solve, whose kwargs pyadjoint strips of `appctx`
    (`firedrake/adjoint_utils/blocks/solving.py:578`). The low-rank path
    supplies the context through `LowRankVariationalSolver`'s `adj_kwargs`,
    and this class reads nothing from it there either.

    What it does read from the appctx is two *markers* that say when the
    complement is stale (see "When the complement is rebuilt" below). Neither
    is data the solve needs: a context that carries neither leaves the
    complement in place, which is a valid preconditioner and a correct solve.

    ## When the complement is rebuilt

    The complement describes the mechanics block, so the question `update` has
    to answer is "has that block moved since the last build". The solver
    already answers it, for itself and for the nested condensation, and this
    class reads the same answer rather than inventing a second one:

    - **`operator_version`**, published by
      `CoupledInternalVariableSolver._refresh_operator_version` at the start of
      every `solve`. It is an integer that changes whenever any coefficient of
      the Jacobian changes - the time step, a viscosity field written between
      steps, a shear modulus or `B_mu` `Constant` assigned - and `None` when
      the rheology makes the Jacobian depend on the state. A changed integer
      means a new mechanics block, so `update` builds a new complement.
      `gadopt.InternalVariableSCPC` keys its own reassembly on this same value,
      which is the argument for using it here: one statement of when the
      operator changed, trusted by everything that caches a piece of it, and no
      second rule to drift out of step with it. It follows that
      `CoupledInternalVariableSolver.invalidate_jacobian` rebuilds this
      complement as well as the condensed operator, which is how a caller
      declares a change the fingerprint cannot see - a coefficient written
      through a view of its data, or one that appears only in a user-supplied
      Jacobian `J`.
    - **`gia_solve_index`**, incremented by `SelfGravitatingGIASolver.solve`
      before each nonlinear solve, read only when `operator_version` is `None`.
      For a power law the block-0 operator moves with the state, so a
      complement built at one Newton iteration describes a Jacobian the later
      iterations no longer have, and no version number can express that. The
      rule is **once per solve**: the complement is built at the first Newton
      iteration of a solve, from the state that solve starts from, and kept as
      a preconditioner through the rest of that solve's Newton iterations.

    Cost rules out the alternative of one build per Newton iteration. One build
    is one block-0 solve per column, 72 columns at L = 5 and 75 with rotation,
    so a build at every Newton iteration roughly triples the block-0 work of a
    three-iteration step. A complement that lags the state costs outer
    iterations and changes no residual, so the cheap rule is the right trade.

    Behaviour under `pyadjoint`, which is not the obvious one. The **forward
    replay** solver is built from the taped constructor kwargs
    (`firedrake/adjoint_utils/variational_solver.py:50, 83-91`), and `appctx`
    is one of them, so the replay solver shares the *live* application context
    of the forward solver. What a replay never does is run
    `SelfGravitatingGIASolver.solve`: replays go through `_forward_solve`
    (`firedrake/adjoint_utils/blocks/solving.py:650`), so neither
    `operator_version` nor `gia_solve_index` moves during a replay and the
    replay solver's own complement is built once at its `initialize` and kept
    across every replay, including every replayed step of a taped march. The
    **adjoint** solve differs by path. On the multiplier path it carries
    Firedrake's default empty context, because pyadjoint pops `appctx` from
    its kwargs (`blocks/solving.py:578`), so both markers are absent, `update`
    returns without comparing anything, and its complement is built once and
    kept. On the low-rank path `LowRankVariationalSolver` supplies the live
    context through `adj_kwargs`, so both markers are present; they are
    frozen during a reverse sweep for the same reason they are frozen during a
    replay (nothing runs `SelfGravitatingGIASolver.solve`), so `update`
    compares equal values and the adjoint's complement is likewise built once,
    at the first adjoint solve, and kept.

    ## Two correctness conditions, and the second is not what the design assumed

    - The complement is a **preconditioner**. A stale one changes no residual,
      only the outer iteration count, which is what makes a per-solve rebuild
      rule a cost decision instead of a correctness one. A new time-step value
      changes the mechanics block and triggers a rebuild anyway, and a
      state-dependent Jacobian triggers one per solve.
    - S must be *linear*, and it is only as linear as the block-0 solve inside
      it. With `dtn_fieldsplit_0_ksp_type: preonly` (a fixed LU) that solve is a
      linear operator and the complement is exact to roundoff. With an iterative
      block-0 KSP -- the configuration that runs at production -- FGMRES to a
      relative tolerance is **not** a linear operator, so each column is built
      with a slightly different effective inverse and S is approximate. Harmless
      for a preconditioner, fatal for "apply it exactly forever": the honest
      description is a very good preconditioner built once.

    ## Transpose

    S is **structurally asymmetric** (relative asymmetry ~0.34): the constraint
    row carries `theta_psi * u_k^T` while the feedback column carries
    `theta_psi * (lam_k - alpha/R) * u_k`, so S is a symmetric matrix times a
    diagonal spanning 0..L/R. `applyTranspose` -- which the adjoint inner solve
    calls -- must therefore solve S^T x = b, not S x = b. `_solve_transpose`
    below does exactly that; the base default (forward = transpose) would be
    silently wrong here.
    """

    _prefix = "dtn_multiplier_dense_schur_"

    def initialize(self, pc):
        """Record the state this build describes, then build the complement.

        Both markers are read before `_build`, so that an `update` inside the
        same solve compares against the state this build describes and does no
        work. Reading them here, rather than lazily at the first `update`, is
        what makes the rule correct for a Newtonian march: `update` is not
        called at all inside a solve that takes one Newton step, so the first
        `update` a fixed-step Newtonian solver sees belongs to the *second*
        solve, and a marker adopted there instead of compared against misses a
        coefficient changed between the two.

        Args:
          pc: PETSc preconditioner.
        """
        self._operator_version, self._solve_index, _ = self._markers(pc)
        # Counts builds over the life of this PC instance, `initialize`
        # included. Tests read it to pin the rebuild rule; nothing in the
        # solve path branches on it.
        self.build_count = 0
        self._build(pc)

    def update(self, pc):
        """Rebuild the complement when the mechanics block it describes moved.

        Two triggers, and the class docstring section "When the complement is
        rebuilt" carries the argument for each:

        1. `operator_version` is an integer and differs from the one this
           complement was built at. The solver bumps it for any change of a
           Jacobian coefficient, the time step included, so this one test
           covers a viscosity field written between steps as well as a
           `dt.assign`.
        2. `operator_version` is `None` - the rheology makes the Jacobian
           depend on the state, so no version can describe it - and
           `gia_solve_index` has moved. The complement is then a preconditioner
           frozen at the state each solve starts from, which is one build per
           time step instead of one per Newton iteration.

        A context carrying neither marker leaves the complement alone. That is
        the multiplier path's adjoint solve, whose kwargs pyadjoint strips of
        `appctx`. The low-rank path's adjoint carries the live context through
        `LowRankVariationalSolver`, with both markers frozen for the sweep, so
        it ends in the same place: one complement, built at the first adjoint
        solve and kept.

        Args:
          pc: PETSc preconditioner.
        """
        version, solve_index, present = self._markers(pc)
        if not present:
            return
        if version is not None:
            if version != self._operator_version:
                self._operator_version = version
                # Carry the solve index across with it, so that a later
                # switch to the state-dependent branch compares against this
                # build and not against an older one.
                self._solve_index = solve_index
                self._build(pc)
            return
        if solve_index is None or self._solve_index is None:
            return
        if solve_index != self._solve_index:
            self._operator_version = version
            self._solve_index = solve_index
            self._build(pc)

    def _appctx(self, pc):
        """The solver's application context, or an empty mapping.

        `PCBase.get_appctx` resolves the context through `pc.getDM()`, and a
        PETSc PC that Firedrake did not build carries no DM: `pc.getDM()`
        returns a wrapper around a NULL handle, and Firedrake's
        `dmhooks.get_appctx` then dereferences it and **crashes the process**
        with a segmentation violation rather than raising (measured on the
        bare dense `Mat` the unit tests of this class drive it over). So the
        handle test comes first and is not tidiness: it is what keeps a
        hand-built PC from killing the run. Reading `dm.handle` is a pure
        Python attribute lookup on the petsc4py wrapper and touches no PETSc
        object, so it is safe where everything else is not.

        Args:
          pc: PETSc preconditioner.

        Returns:
          The application context mapping, or `{}` when there is none to read.
        """
        dm = pc.getDM()
        if dm.handle == 0:
            return {}
        try:
            return self.get_appctx(pc) or {}
        except AttributeError:
            # A Firedrake DM with no solver context pushed onto it:
            # `dmhooks.get_appctx` returns None and the `.appctx` lookup on it
            # raises. Read that as "no markers", exactly like an empty context.
            return {}

    def _markers(self, pc):
        """The solver's statement of when the mechanics block last changed.

        `operator_version` is the same value `gadopt.InternalVariableSCPC`
        keys its reassembly on, so the two caches of pieces of one operator
        agree by construction. `gia_solve_index` covers the case that version
        cannot express, a Jacobian that moves inside one nonlinear solve.

        Both are rank-consistent: the version is bumped by a reduction over
        the communicator in `_refresh_operator_version`, and the solve index is
        incremented by the collective `solve()`. So the rebuild decision this
        returns is identical on every rank and the collective `_build` is
        entered by all ranks together.

        Args:
          pc: PETSc preconditioner.

        Returns:
          `(operator_version, solve_index, present)`. `present` is False when
          the context names neither marker, which is the signal to leave the
          complement alone.
        """
        appctx = self._appctx(pc)
        present = "operator_version" in appctx or "gia_solve_index" in appctx
        solve_index = appctx.get("gia_solve_index")
        return (appctx.get("operator_version"),
                None if solve_index is None else int(solve_index),
                present)

    def _build(self, pc):
        A, _ = pc.getOperators()
        # getSizes returns ((local_rows, GLOBAL_rows), (local_cols, GLOBAL_cols)).
        # Take the GLOBAL sizes. Every Real dof sits on one rank, so the LOCAL
        # column count is the whole block (72) on that rank and 0 on every other.
        # This PC builds S redundantly on every rank -- `_gather` Allreduces a
        # buffer of length `self._n`, and `_solve` runs the dense factor on all
        # ranks -- so `n` must be the global size everywhere. Keying it on the
        # local size makes the off-rank processes build a 0x0 S, skip the
        # (collective) MatMult loop, and raise "zero-size array to reduction" at
        # `np.abs(S).max()`. Measured on a 104-rank Gadi run: an Unhandled Python
        # Exception on every rank but 0, invisible to any serial unit test.
        ((_lm, m), (_ln, n)) = A.getSizes()
        if m != n:
            raise self._loud(ValueError(
                f"DtNMultiplierDenseSchurPC needs a square operator to form the "
                f"Schur complement, but its Amat is {m}x{n}. The block-1 PC must "
                "be handed the MATSCHURCOMPLEMENT of a full Schur factorisation "
                "(dtn_pc_fieldsplit_schur_fact_type: full)."))
        comm = pc.comm.tompi4py()
        e = A.createVecRight()
        col = A.createVecLeft()
        S = np.zeros((n, n))
        for k in range(n):
            e.set(0.0)
            lo, hi = e.owner_range
            if lo <= k < hi:
                e.setValue(k, 1.0)
            e.assemble()
            # One MatMult of the MATSCHURCOMPLEMENT is one application of S, so
            # column k of S is S e_k, gathered redundantly onto every rank.
            A.mult(e, col)
            S[:, k] = self._gather(comm, col, n)
        e.destroy()
        col.destroy()
        self._n = n
        self._S = S
        self._factorise(S)
        # Counted after the factorisation, so the count is the number of usable
        # complements this PC has produced and a build that raised is not one
        # of them.
        self.build_count = getattr(self, "build_count", 0) + 1
        scale = max(np.abs(S).max(), 1e-300)
        PETSc.Sys.Print(
            f"    [dense Schur] {n}x{n} built in {n} block-0 applications; "
            f"cond {np.linalg.cond(S):.3e}  "
            f"relative asymmetry {np.abs(S - S.T).max() / scale:.3e}")

    def _factorise(self, S):
        """Factor S so that both S x = b and S^T x = b are cheap from here on.

        Prefer an LU factorisation (scipy `lu_factor`/`lu_solve`), which does the
        forward and the transpose solve off one factoring; fall back to storing
        the inverse and its transpose when scipy is absent.
        """
        n = S.shape[0]
        if _HAVE_SCIPY:
            lu = _lu_factor(S)
            # lu_factor never raises on a singular matrix -- it returns a U with
            # a zero pivot and a LinAlgWarning. Catch the zero pivot here so the
            # failure is a NAMED ValueError above the eventual PETSc 101, never a
            # silent NaN solve.
            if not np.all(np.abs(np.diag(lu[0])) > 0.0):
                raise self._loud(ValueError(
                    f"the {n}x{n} multiplier Schur complement is singular: a "
                    "zero pivot appeared in its LU factorisation. A mode with "
                    "zero scale or a boundary with zero discrete area would do "
                    "that, and both are bugs upstream."))
            self._lu = lu
        else:  # pragma: no cover - scipy ships in the firedrake venv
            try:
                self._inv = np.linalg.inv(S)
            except np.linalg.LinAlgError as exc:
                raise self._loud(ValueError(
                    f"the {n}x{n} multiplier Schur complement is singular: "
                    f"{exc}")) from None
            self._invT = self._inv.T.copy()

    def _solve(self, rhs):
        if _HAVE_SCIPY:
            return _lu_solve(self._lu, rhs, trans=0)
        return self._inv @ rhs

    def _solve_transpose(self, rhs):
        if _HAVE_SCIPY:
            return _lu_solve(self._lu, rhs, trans=1)
        return self._invT @ rhs


class _LowRankPotentialOperator:
    r"""`A_psipsi + B` on the potential space, with `B` applied in factored form.

    The potential block of `gadopt.CondensedBlockPC`'s nest on the low-rank DtN
    path. `B = theta * sum_b C_b^T W_b C_b` is the exterior condition the
    multiplier representation obtains through its Schur complement; here it is
    the rank-k update `gadopt.dtn_coupled.CoupledLowRankDtN` applies, and it is
    never formed. One application is the assembled matrix-vector product plus,
    per DtN boundary, two small dense products against the boundary entries with
    one `Allreduce` of the `k`-vector in between.

    **The indices are the potential space's own.** The sub-vector of the nest
    carries the potential space's layout, so the rows are
    `BoundaryModeRows.dofs` directly. `CoupledLowRankDtN.rows_mono` shifts the
    same rows onto the monolithic mixed vector by `psi_local_offset` and is
    wrong here by exactly the owned size of the displacement and history fields
    in front of the potential. That error writes the update into the wrong rows
    of a vector it still fits inside, so it neither raises nor changes the
    residual: it degrades the preconditioner and nothing says so.

    **`theta` is read at every application.** It is
    `scaling_factor * B_mu / Lambda`, all three of which can be adjoint
    controls, so it is taken from the operator's own callable rather than
    folded into the weights once.

    `multTranspose` is the same action, because `B` is symmetric and the
    assembled potential block is too.

    Attributes:
      assembled: The assembled `A_psipsi`, as a `PETSc.Mat`. Reassembled in
        place by `CondensedBlockPC._assemble_blocks`, so this reference stays
        valid across every rebuild.
      dtn: The `CoupledLowRankDtN`, for its `mode_rows`, its prefactor and its
        communicator.
      applications: Counts applications, so a test can confirm the update ran
        rather than being bypassed.
    """

    def __init__(self, assembled, dtn_operator):
        self.assembled = assembled
        self.dtn = dtn_operator
        self.applications = 0

    def mult(self, mat, x, y):
        """`y = (A_psipsi + B) x`."""
        self.assembled.mult(x, y)
        theta = self.dtn.theta_value
        x_local = x.array_r
        # `array_w` hands back the same local memory and marks it modified, so
        # the `+=` below accumulates onto what `mult` just wrote. This is the
        # pattern `gadopt.dtn_lowrank.LowRankDtNOperator._add_low_rank` uses.
        y_local = y.array_w
        for rows in self.dtn.mode_rows:
            if rows.rows.size:
                local = rows.rows @ x_local[rows.dofs]
            else:
                local = np.zeros(len(rows.keys))
            total = np.empty_like(local)
            # The modes are global and the boundary dofs are partitioned, so
            # the `k`-vector is summed before it is scattered back. This is the
            # whole of the parallel communication of one application.
            self.dtn.comm.Allreduce(local, total)
            if rows.rows.size:
                y_local[rows.dofs] += theta * (
                    rows.rows.T @ (rows.weights * total))
        self.applications += 1

    # Symmetric, so the transpose action is the same action. Stated rather than
    # left to inheritance: PETSc will use this for a transposed solve, and
    # getting it wrong would show up only in the adjoint.
    multTranspose = mult


class LowRankPotentialPC(fd.preconditioners.base.PCBase):
    r"""GAMG plus a Woodbury correction for `A_psipsi + B`.

    The preconditioner of the potential split of `gadopt.CondensedBlockPC` on
    the low-rank DtN path. Its operator is `_LowRankPotentialOperator`, which
    the class reads off its own preconditioning matrix's Python context; it
    needs no application-context key of its own.

    ## Why this exists

    `AugmentedImplicitMatrixContext` adds `B` to the OUTER Jacobian's action
    only. Firedrake's `createSubMatrix` builds every fieldsplit sub-block as a
    plain `ImplicitMatrixContext`, so block 0 never sees `B`, and
    `CondensedBlockPC` assembles `A_psipsi` from the bilinear form, which has no
    `B` in it either. Measured on the annulus: the outer FGMRES then spends 8 or
    9 iterations where the multiplier representation spends 3
    (`NOTES/FINDING-LOWRANK-ITERATIVE-2026-09-16.md`). This class is what puts
    `B` inside block 0.

    ## The identity

    With `U` the `n_psi x k` matrix whose columns are the mode rows, `W` the
    diagonal of the weights and `theta` the prefactor, the Woodbury identity in
    the form that never inverts `W` is

        (A + theta U W U^T)^-1 = A^-1 - Z Cap^-1 (theta W U^T A^-1)
        Z   = A^-1 U
        Cap = I + theta W (U^T Z)

    so one application costs one solve against `A` alone, one `Allreduce` of a
    `k`-vector, one dense solve of size `k`, and one scatter back.

    **`W` is singular in every default configuration, which is why the identity
    is written this way.** A weight is `(lam_k - alpha/R) / (scale_k * A_h)`,
    and `gadopt.dtn_form` builds `lam = (l+1)/R` on an exterior boundary and
    `l/R` on an interior one with `alpha = 1`, so the weight is exactly zero for
    the exterior `l = 0` mode and the interior `l = 1` modes: four modes at
    every truncation, in 2-D and in 3-D alike. The textbook form of the identity
    carries `W^-1/theta` in the capacitance and does not exist for any of them.
    This form carries `theta W` instead, contributes nothing for a zero weight,
    which is what a zero weight means, and stays well conditioned for a small
    one. `Cap` is non-singular exactly when `A + B` is, by the determinant
    lemma.

    `Z = A^-1 U` is built by `k` accurate solves of `A_psipsi`, to the fixed
    `_column_rtol` of 1e-10, because an error in a column enters `Cap` and is not
    corrected by the outer Krylov method. The application's own `A^-1` may be as
    inexact as one V-cycle: it makes the identity approximate, which is what a
    preconditioner is allowed to be.

    ## What is rebuilt, and when

    Within one forward run nothing is. The potential block is
    `theta * (grad psi . grad v dx + sum_b (alpha/R_b) psi v ds_b)`, so neither
    `dt` nor the stress nor the rheology enters it, and `B = theta * B0` with
    `theta = scaling_factor * B_mu / Lambda` does not either. `Z`, `Cap` and the
    multigrid hierarchy therefore survive every time-step change and every
    Newton iteration, which the multiplier representation's dense complement
    cannot do, because that complement goes through the mechanics block.

    **`theta` is not an independent input.** It multiplies the whole potential
    row, so it scales `A_psipsi` and `B` together: a replay with `Lambda`,
    `B_mu` or `scaling_factor` as a control moves `A_psipsi` by the same factor
    it moves `B`. Refactoring `Cap` at a new `theta` while keeping `Z` and the
    hierarchy of the old matrix would build a capacitance belonging to no
    operator at all. So there is ONE reuse test, the Frobenius norm of
    `A_psipsi`, and everything is rebuilt together when it moves.

    The inner Krylov solve works on a private copy of `A_psipsi`, refreshed from
    the same test. Without the copy, `CondensedBlockPC.update` reassembling the
    block in place and calling `assemble()` on the enclosing nest bumps the
    matrix state and makes PETSc re-run the multigrid setup on every `dt` change
    and every Newton iteration, which is a recomputation of the coarse operators
    for entries that did not change.

    ## Options

    Under this preconditioner's own prefix plus `lowrank_`: `ksp_*` and `pc_*`
    for the solve against `A_psipsi` that the application uses. The default the
    preset writes is one GAMG V-cycle (`ksp_type preonly`). The column build
    uses the same preconditioner and hierarchy with CG and a tight tolerance,
    so a second hierarchy is never built.

    ## What is refused

    A preconditioning matrix that is not `_LowRankPotentialOperator`, which
    means the class was selected on the multiplier path or outside
    `CondensedBlockPC`'s nest. A transpose application is refused for the same
    reason `CondensedBlockPC` refuses one, and because this approximate inverse
    is not symmetric anyway: the columns `Z` come from an accurate solve and the
    leading term from one V-cycle, so the two halves of the correction are built
    to different accuracies.
    """

    needs_python_pmat = True

    #: Option prefix of the solve against `A_psipsi`, under this
    #: preconditioner's own prefix.
    _prefix = "lowrank_"

    #: Relative tolerance of the `k` solves that build the columns `Z`. Tight
    #: because an error in a column enters the capacitance, which no outer
    #: Krylov method corrects; the application's own solve is separate and may
    #: be one V-cycle.
    _column_rtol = 1e-10

    def initialize(self, pc):
        """Build the hierarchy, the columns `Z` and the capacitance."""
        _, P = pc.getOperators()
        # The type is tested before the context is asked for: `getPythonContext`
        # on a matrix of any other type raises inside PETSc, and this method
        # runs inside a PETSc callback, where a raised error is reported as an
        # error code with the Python cause some distance away.
        context = (P.getPythonContext()
                   if P.getType() == PETSc.Mat.Type.PYTHON else None)
        if not isinstance(context, _LowRankPotentialOperator):
            raise ValueError(
                "gadopt.LowRankPotentialPC preconditions the potential block "
                "of gadopt.CondensedBlockPC on the low-rank DtN path, so its "
                "preconditioning matrix must carry a "
                "_LowRankPotentialOperator Python context; it got "
                f"{type(context).__name__}. On the multiplier representation "
                "the potential split has no low-rank update and its "
                "preconditioner is plain GAMG: select this class only through "
                "selfgrav_dtn_iterative_solver_parameters("
                "dtn_representation='lowrank').")
        self.context = context
        self.assembled = context.assembled
        self.dtn = context.dtn
        self.comm = self.dtn.comm

        # The columns of `U`, flattened across the DtN boundaries: one entry
        # per mode, holding the owned local indices it touches, the row itself
        # and its weight. The flattening is what makes the capacitance one
        # dense `k x k` system instead of one per boundary, which it must be:
        # two boundaries share the potential space, so their columns are not
        # orthogonal and a per-boundary capacitance would ignore the coupling.
        # A zero weight is kept, not refused and not dropped: it contributes
        # nothing to `B`, and the capacitance below carries `theta W` instead
        # of `W^-1/theta`, so a zero costs a row of the identity and nothing
        # else. Four modes are exactly zero in every default configuration.
        self._columns = []
        for rows in self.dtn.mode_rows:
            for mode in range(len(rows.keys)):
                self._columns.append((rows.dofs,
                                      np.ascontiguousarray(rows.rows[mode]),
                                      float(rows.weights[mode])))
        self._n_modes = len(self._columns)
        self._weights = np.array([w for _, _, w in self._columns])

        # A private copy, so that the multigrid hierarchy is not torn down and
        # rebuilt every time `CondensedBlockPC.update` reassembles the block in
        # place and calls `assemble()` on the enclosing nest. Those two bump the
        # matrix state, and PETSc then re-runs the setup of any preconditioner
        # built on it, whether or not an entry moved. The copy is refreshed from
        # the same reuse test that governs `Z`.
        self._matrix = self.assembled.copy()

        prefix = (pc.getOptionsPrefix() or "") + self._prefix
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(self._matrix, self._matrix)
        ksp.setFromOptions()
        self.ksp = ksp

        #: Builds of the columns `Z` and of everything keyed with them. A
        #: forward march at any number of time steps must leave this at 1.
        self.column_builds = 0
        self._rebuild()

    def _mark_spd(self):
        """Claim the private copy symmetric positive definite, as the block is.

        `CondensedBlockPC._assemble_blocks` makes the same claim on the block
        itself after every assembly, and a copy does not inherit it. Without it
        the multigrid setup takes its non-symmetric path on a Laplacian.
        """
        self._matrix.setOption(PETSc.Mat.Option.SPD, True)
        self._matrix.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

    def _rebuild(self):
        """Refresh the private copy, the columns `Z` and the capacitance.

        `_columns`, `_weights` and `_n_modes` are read once in `initialize`
        and are not refreshed here, because the mode rows are built once in
        the operator's constructor and a change of truncation needs a new
        solver. A rebuild answers a change of `A_psipsi` or of the prefactor,
        which moves `Z`, `U^T Z` and the capacitance and nothing else.
        """
        self.assembled.copy(self._matrix,
                            PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        self._mark_spd()
        self._build_columns()
        self._factorise_capacitance()
        self._fingerprint = self._matrix_fingerprint()
        self.column_builds += 1

    def _matrix_fingerprint(self) -> float:
        """The Frobenius norm of `A_psipsi`, as the reuse test for `Z`.

        A norm is a weak fingerprint, and it is the right strength here: the
        columns are expected never to need rebuilding, so this exists to catch
        a potential block that moves against expectation, and a move that
        preserved the norm exactly would have to be a reflection of the
        entries. The cost is one pass over the non-zeros per linear solve,
        against a multigrid V-cycle on the same matrix.
        """
        return float(self.assembled.norm(PETSc.NormType.FROBENIUS))

    def _build_columns(self):
        """`Z = A^-1 U`, by one accurate solve per mode.

        The solves borrow this class's KSP, and so its multigrid hierarchy,
        with the method and tolerance temporarily replaced: building a second
        KSP would build a second hierarchy on the same matrix.
        """
        saved_type = self.ksp.getType()
        saved_rtol, saved_atol, saved_dtol, saved_max = self.ksp.getTolerances()
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.setTolerances(rtol=self._column_rtol, max_it=1000)
        try:
            rhs = self.assembled.createVecRight()
            self._Z = []
            for dofs, row, _ in self._columns:
                # `set` then `array_w`, and not two writes through one view:
                # the cached norm `KSPSolve` tests `rtol` against is
                # invalidated when the array view is released, and a single
                # view would leave that release to the moment CPython drops the
                # previous iteration's reference. Two explicit acquisitions
                # cost one call and depend on nothing.
                rhs.set(0.0)
                rhs.array_w[dofs] = row
                column = self.assembled.createVecRight()
                self.ksp.solve(rhs, column)
                reason = self.ksp.getConvergedReason()
                if reason < 0:
                    raise ValueError(
                        "gadopt.LowRankPotentialPC could not solve for a "
                        "column of Z: the potential block's Krylov solve "
                        f"stopped with PETSc reason {reason}. The columns "
                        "enter the capacitance directly, so an inaccurate one "
                        "is a wrong preconditioner and not a slow one.")
                self._Z.append(column)
        finally:
            self.ksp.setType(saved_type)
            self.ksp.setTolerances(rtol=saved_rtol, atol=saved_atol,
                                   divtol=saved_dtol, max_it=saved_max)
        # `U^T Z`, the part of the capacitance that does not carry the
        # prefactor. Built rank-locally against the sparse rows and summed
        # once, instead of by `k^2` global dot products.
        local = np.zeros((self._n_modes, self._n_modes))
        for j, column in enumerate(self._Z):
            column_local = column.array_r
            for i, (dofs, row, _) in enumerate(self._columns):
                local[i, j] = row @ column_local[dofs]
        self._UtZ = np.empty_like(local)
        self.comm.Allreduce(local, self._UtZ)

    def _factorise_capacitance(self):
        """`Cap = I + theta W (U^T Z)`, factored for the application.

        This form of the identity never inverts `W`, which is what lets the
        four zero-weight modes of every default configuration through. The
        prefactor is read here and enters as a row scaling.
        """
        self._theta = self.dtn.theta_value
        cap = np.eye(self._n_modes) + (
            (self._theta * self._weights)[:, np.newaxis] * self._UtZ)
        if _HAVE_SCIPY:
            lu = _lu_factor(cap)
            # `lu_factor` never raises on a singular matrix: it returns a `U`
            # with a zero pivot and a warning. Catching it here makes the
            # failure a named error instead of a silent NaN application that
            # the outer Krylov method reports as stagnation.
            if not np.all(np.abs(np.diag(lu[0])) > 0.0):
                raise ValueError(
                    f"gadopt.LowRankPotentialPC's {self._n_modes}x"
                    f"{self._n_modes} capacitance matrix is singular: a zero "
                    "pivot appeared in its LU factorisation.")
            self._cap_lu = lu
            self._cap_inv = None
        else:  # pragma: no cover - scipy ships in the firedrake venv
            self._cap_lu = None
            self._cap_inv = np.linalg.inv(cap)

    def _cap_solve(self, rhs):
        """`Cap^-1 rhs` through the stored factorisation.

        `check_finite=False`: a non-finite residual arriving from a diverged
        displacement split must propagate as a NaN, which the outer Krylov
        method reports as `DIVERGED_NANORINF`, the same way the multiplier
        path reports it. With the check on, scipy raises `ValueError: array
        must not contain infs or NaNs` here, and the traceback points at the
        capacitance instead of at the split that diverged.
        """
        if self._cap_lu is not None:
            return _lu_solve(self._cap_lu, rhs, check_finite=False)
        return self._cap_inv @ rhs

    def update(self, pc):
        """Rebuild everything, or nothing, on one test of `A_psipsi`.

        PETSc calls this on every linear solve. The prefactor is NOT a second
        input to test: it scales the assembled block and the update together,
        so a change in it moves this norm.
        """
        # The rebuild below is COLLECTIVE (it solves, and it reduces), so every
        # rank must reach the same decision or the ones that rebuild wait on an
        # `Allreduce` the others never enter. `Mat.norm` reduces and MPI
        # guarantees an identical result everywhere, so comparing the floats
        # already agrees; this one-flag reduction removes the dependence on
        # that guarantee for the price of one boolean. It is the pattern
        # `_refresh_operator_version` uses for the same reason.
        moved = self._matrix_fingerprint() != self._fingerprint
        if self.comm.allreduce(moved, MPI.LOR):
            self._rebuild()

    @PETSc.Log.EventDecorator("LowRankPotentialPCApply")
    def apply(self, pc, x, y):
        """`y = (A + theta U W U^T)^-1 x`, to the accuracy of the inner solve."""
        self.ksp.solve(x, y)
        y_local = y.array_r
        local = np.array([row @ y_local[dofs]
                          for dofs, row, _ in self._columns])
        total = np.empty_like(local)
        # The modes are global and the boundary dofs are partitioned, so the
        # `k`-vector is summed before the dense solve. This is the whole of the
        # parallel communication of one application.
        self.comm.Allreduce(local, total)
        correction = self._cap_solve(self._theta * self._weights * total)
        # One fused pass over the potential vector instead of `k` separate
        # ones. At the truncations this exists for, `k` is in the hundreds per
        # boundary, so `k` calls to `axpy` would read `k` potential vectors per
        # block-0 Krylov iteration.
        y.maxpy(-correction, self._Z)

    def applyTranspose(self, pc, x, y):
        """Refused; the transpose application is not implemented.

        Two reasons, either sufficient. pyadjoint solves `adjoint(J)` with the
        forward options, so an adjoint solve reaches `apply` on the adjoint
        operator and never this method, which is why `CondensedBlockPC` refuses
        one as well. And the map this class applies is not symmetric even
        though its operator is: the leading term comes from one multigrid
        V-cycle and the columns `Z` from an accurate solve, so the two halves of
        the correction are built to different accuracies. A silent `apply` here
        would precondition a transposed solve with a map that is close to the
        right one and not equal to it.
        """
        raise NotImplementedError(
            "gadopt.LowRankPotentialPC has no transpose application. An "
            "adjoint solve reaches the forward apply, because pyadjoint solves "
            "adjoint(J) with the forward solver options.")

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        # The base class quietly returns on a missing or non-ASCII viewer, so
        # repeat its test before writing anything of our own.
        if viewer is None or viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            f"Low-rank potential preconditioner, {self._n_modes} modes, "
            f"{self.column_builds} build(s)\n")
        self.ksp.view(viewer)


class SubstitutedDisplacementPC(fd.AuxiliaryOperatorPC):
    r"""Precondition the displacement Schur complement of a coupled GIA solve.

    `CoupledInternalVariableSolver` keeps the internal variables `m_i` as
    unknowns next to the displacement `u`. Eliminating them exactly from the
    coupled Jacobian gives the displacement operator of the substituted solver
    (`InternalVariableSolver`), because the backward-Euler update

    $$ m_i^{new} = \frac{m_i^{old} + (dt/\tau_i)\,d(u)}{1 + dt/\tau_i} $$

    is linear in the deviatoric strain `d(u)` and the internal-variable block
    is a cell-local DG mass matrix. The eliminated operator is therefore the
    elastic displacement block minus the viscous part of the shear response:

    $$ S = A_{uu} - \sum_i 2\mu_i \frac{dt}{\tau_i + dt}
           \int d(v) : d(u) \, dx $$

    which has shear coefficient `eta_eff = sum_i mu_i tau_i / (tau_i + dt)`.

    This class builds that operator as the preconditioning matrix for the
    displacement split of a Schur fieldsplit that eliminates the internal
    variables first. It reuses the elastic block `A_uu` handed to it by the
    fieldsplit, which already carries the Nitsche boundary terms and the
    free-surface prestress term, and subtracts the volume correction above.
    It differs from the exact Schur complement in two places. The exact
    complement carries the L2 projection of `d(u)` onto the DQ1
    internal-variable space, which for a Q2 displacement on hexahedra is not
    the identity, while the correction here uses `d(u)` itself; that is a
    volume difference. And the boundary terms of `A_uu` keep the elastic
    coefficient `mu0`, so on a boundary with a weak normal condition the two
    differ by a boundary term as well. GAMG then runs on a sparse matrix that
    is spectrally close to the substituted operator.

    The application context must carry:

    - `approximation`: the `InternalVariableApproximation` of the solve, for
      `shear_modulus`, `maxwell_times` and `deviatoric_strain`.
    - `dt`: the time step.
    - `scaling_factor` (optional, default 1): the factor the coupled solver
      applies to the displacement residual.
    - `displacement_near_nullspace` (optional): a callable that takes the
      displacement space and returns a `VectorSpaceBasis`. It is used when the
      fieldsplit hands over no near-nullspace, which is what GAMG needs for
      the rigid-body modes.

    Options for the inner solver use the `aux_` prefix, for example
    `fieldsplit_1_aux_pc_type: gamg`. The matrix is flagged SPD.
    """

    def form(self, pc: PETSc.PC, test, trial):
        _, P = pc.getOperators()
        if P.getType() != "python":
            raise ValueError(
                f"{type(self).__name__} needs a matrix-free operator "
                "(mat_type matfree) so that it can read the displacement "
                "block's bilinear form."
            )
        context = P.getPythonContext()
        if not context.on_diag:
            raise ValueError("Only makes sense to precondition a diagonal block")
        elastic_block = context.a
        bcs = context.row_bcs

        appctx = self.get_appctx(pc)
        missing = [key for key in ("approximation", "dt") if key not in appctx]
        if missing:
            raise KeyError(
                f"{type(self).__name__} needs {missing} in the application "
                "context; CoupledInternalVariableSolver.set_solver_options sets them."
            )
        approximation = appctx["approximation"]
        dt = appctx["dt"]
        scaling_factor = appctx.get("scaling_factor", 1)

        # Use the arguments of the extracted block so that the correction lives
        # on the same (sub)space as the elastic block it is added to.
        block_test, block_trial = elastic_block.arguments()
        # Reuse the quadrature degree of the elastic block's cell integrals so
        # that the two parts of the form are integrated consistently.
        degrees = {
            integral.metadata().get("quadrature_degree")
            for integral in elastic_block.integrals()
            if integral.integral_type() == "cell"
        }
        degrees.discard(None)
        dx = fd.dx(degree=max(degrees)) if degrees else fd.dx

        # Viscous part of the shear response that the eliminated internal
        # variables remove from the elastic block. Each Maxwell element with
        # shear modulus mu_i and Maxwell time tau_i contributes the fraction
        # dt/(tau_i + dt) of its elastic stiffness after one implicit step.
        # These are the unscaled Maxwell times: for a power-law rheology the
        # residual scales them by the power-law factor, so this operator is
        # right for a Newtonian rheology only.
        strain = approximation.deviatoric_strain(block_trial)
        correction = 0
        for mu_i, tau_i in zip(
            approximation.shear_modulus, approximation.maxwell_times
        ):
            correction += 2 * mu_i * dt / (tau_i + dt)
        viscous_part = (
            scaling_factor
            * fd.inner(fd.nabla_grad(block_test), correction * strain)
            * dx
        )
        return elastic_block - viscous_part, bcs

    def set_nullspaces(self, pc: PETSc.PC):
        """Copy the parent nullspaces, and build the near-nullspace if absent."""
        super().set_nullspaces(pc)
        Pmat = self.P.petscmat
        if Pmat.getNearNullSpace().handle != 0:
            return
        builder = self.get_appctx(pc).get("displacement_near_nullspace")
        if builder is None:
            return
        V = get_function_space(pc.getDM()).collapse()
        Pmat.setNearNullSpace(builder(V).nullspace())

    def initialize(self, pc: PETSc.PC):
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


def _restrict_to_mesh(form, mesh):
    """Keep only the integrals of `form` written on `mesh`.

    Slate compiles an expression whose integrals are on ONE mesh, and the
    self-gravity block-0 form spans two: the potential on the parent mesh and
    the displacement and internal variables on the mantle submesh. Extracting
    the `(u, M)` sub-block is enough on the forward path, where every surviving
    integral is on the submesh. It is NOT enough on an adjoint operator: one
    cell integral on the PARENT mesh survives the same extraction, and Slate
    then fails with `too many values to unpack (expected 1)` out of
    `slate/slac/compiler.py:_compile_expression_comm`.

    Measured on the annulus with a fluid core
    (`NOTES/team/lowrank-block0/diag_adj.py`): the forward `(u, M)` block has
    15 integrals, all on the submesh; the adjoint's has one more, on the
    parent mesh.

    **That integral is identically zero**, and this function exists because
    the splitter does not reduce it. Its integrand is

        conj(theta * sum_i grad([v_0[0..5], 0, ..., 0])[6, i]
                    * grad([v_1[0..5], 0, ..., 0])[6, i])

    the potential Laplacian with the potential slot, component 6, already set
    to `Zero` by `ExtractSubBlock.argument`. UFL folds `Grad(Zero)` and does
    not fold `Indexed(Grad(ListTensor), 6)`, so the integrand survives as a
    symbol with no value: `expand_derivatives` on it returns a form with no
    integrals. It appears only on an adjoint operator because `ufl.adjoint`
    reorders the arguments after `derivative` has run, and a forward block
    reaches this class already expanded.

    So this filter changes no number, on either path, and it proves that on
    every call: the dropped integrals are expanded, and a form that keeps any
    integral after `expand_derivatives` is refused. A genuine cross-mesh
    `(u, M)` coupling on the parent mesh would belong in `A_uM`, and dropping
    it would change `S_uu` with nothing in any log to say so. The proof is one
    `expand_derivatives` on one integral, once per `initialize`.

    Args:
      form: the extracted `(u, M)` sub-block.
      mesh: the mesh the internal-variable elimination runs on.

    Returns:
      The form with the integrals on other meshes removed, or the form itself
      when every integral is already on `mesh`, which is every forward call.
    """
    kept = [integral for integral in form.integrals()
            if integral.ufl_domain() is mesh]
    if len(kept) == len(form.integrals()):
        return form
    if not kept:
        raise ValueError(
            "gadopt.CondensedBlockPC found no integral of the "
            "(displacement, internal variable) block on the displacement's "
            f"own mesh {mesh}. There is then nothing to eliminate and the "
            "block-0 form is not the one this class was written for.")
    dropped = [integral for integral in form.integrals()
               if integral.ufl_domain() is not mesh]
    # The proof that the filter changes no number: every dropped integral
    # expands to nothing.
    if not expand_derivatives(ufl_Form(dropped)).empty():
        raise ValueError(
            "gadopt.CondensedBlockPC found a (displacement, internal "
            "variable) coupling on a mesh other than the displacement's own "
            f"mesh {mesh} that does not expand to zero. Dropping it would "
            "change the eliminated block silently, so it is refused; the term "
            "belongs in the displacement-internal-variable coupling block.")
    return ufl_Form(kept)


def _split_mixed_coefficients(form):
    """Replace every mixed-space coefficient of `form` by its components.

    Slate compiles an expression whose arguments and coefficients live on ONE
    mesh. The self-gravity block-0 space spans two: the potential on the parent
    mesh and the displacement on the mantle submesh. Extracting the `(u, M)`
    sub-block makes the ARGUMENTS single-mesh, and a coefficient on the whole
    mixed space still spans both, so Slate refuses the form with
    `too many values to unpack (expected 1)` out of
    `slate/slac/compiler.py:_compile_expression_comm`.

    On the forward path the question never arises. Firedrake reaches this class
    through `createSubMatrix`, which runs `solving_utils.split`, and that
    replaces the mixed solution by its per-field components before the form
    ever gets here. An adjoint solve builds its operator from
    `adjoint(dFdu)` and hands over a form whose coefficients are still on the
    mixed space, so the same class meets a form the forward path never
    produces.

    This does what `solving_utils.split` does to `J`: rebuild each mixed
    coefficient as a `ufl.as_vector` of the scalar components of its
    sub-functions, in field order, and substitute it. A form with no mixed
    coefficient is returned unchanged, which is every forward call.

    Args:
      form: the extracted sub-block, a UFL form.

    Returns:
      The same form with every mixed-space coefficient replaced.
    """
    replacements = {}
    for coefficient in form.coefficients():
        space = coefficient.function_space()
        if space is None or len(space) <= 1:
            continue
        components = []
        for piece in fd.split(coefficient):
            if piece.ufl_shape == ():
                components.append(piece)
            else:
                # A tensor-valued field (the combined internal variables are
                # `(n, d, d)`) contributes every scalar component, in the
                # order `as_vector` expects.
                components.extend(piece[index]
                                  for index in np.ndindex(piece.ufl_shape))
        replacements[coefficient] = ufl_as_vector(components)
    if not replacements:
        return form
    return fd.replace(form, replacements)


def internal_variable_condensation(A, keep: int = 0, eliminate: int = 1):
    r"""The Slate Schur complement of the cell-local internal-variable block.

    The coupled internal-variable Jacobian couples the displacement `u` to the
    combined history field `M` through the two off-diagonal blocks, and the
    `M` block itself is block-diagonal per cell: a DG mass matrix scaled by
    `1/dt + 1/tau_i` per Maxwell element, plus the coupling between elements
    that a power-law rheology introduces. Its inverse is therefore cell-local
    and exact, and eliminating `M` gives the exact Schur complement

    $$ S_{uu} = A_{uu} - A_{uM} A_{MM}^{-1} A_{Mu} $$

    on the displacement space. All Maxwell elements live in the one field, so
    the number of elements never appears here and the expression is the same
    for `n = 1` and for `n = 4`.

    This is written once and used by both routes that eliminate `M`:
    `InternalVariableSCPC`, which eliminates it on every inner iteration of a
    Krylov solve over the pair `(u, M)`, and `CondensedBlockPC`, which
    eliminates it once per block-0 application. Two copies of one Slate
    expression would be two chances for a sign or a transpose to drift apart,
    and a preconditioner built from a slightly different operator still
    converges, only more slowly.

    Args:
      A: a Slate `Tensor` of the mixed bilinear form. It may hold more fields
        than the two named here (the block-0 form of the self-gravity solver
        holds `(u, M, psi)`); only the two named blocks are read.
      keep: the field index of the displacement, which stays in the condensed
        system.
      eliminate: the field index of the combined internal-variable field.

    Returns:
      `(S, inverse)`, the Slate expression for the condensed displacement
      operator and the Slate inverse of the `M` block. The inverse is returned
      because the caller needs the same expression for the right-hand side
      elimination and for the back-substitution.
    """
    blocks = A.blocks
    # One Slate inverse of the cell-local internal-variable block, which for a
    # power-law rheology also holds the coupling between elements, so this is
    # the exact tangent elimination and not an approximation of it.
    inverse = blocks[eliminate, eliminate].inv
    condensed_operator = (blocks[keep, keep]
                          - blocks[keep, eliminate] * inverse
                          * blocks[eliminate, keep])
    return condensed_operator, inverse


class InternalVariableSCPC(fd.SCPC):
    r"""Static condensation of the DG internal variables of a coupled GIA solve.

    The coupled internal-variable Jacobian has a displacement block and one
    cell-local block for the internal-variable field (a DG mass matrix scaled
    by `1/dt + 1/tau_i` per Maxwell element, plus the cross coupling between
    elements that a power-law rheology introduces). Slate eliminates that
    block cell by cell and assembles the condensed displacement operator

    $$ S = A_{uu} - A_{uM} A_{MM}^{-1} A_{Mu} $$

    as a sparse matrix on the displacement space, which is the exact Schur
    complement of the coupled system. The condensed system is solved with the
    options under the `condensed_field_` prefix, and the internal variables
    are recovered locally afterwards. Every Maxwell element lives in the one
    field, so the elimination is always of field 1 and the number of elements
    never appears here. For a power-law rheology the cross coupling between
    elements sits inside the `M` block, so its Slate inverse is the exact
    tangent elimination.

    `SCPC`'s own Schur builder assumes the eliminated fields come before the
    kept one (its default keeps the last field), and fails on the `(u, M)`
    layout where the displacement comes first. This class therefore writes
    the condensed system and the back-substitution itself, in
    `condensed_system` and `local_solver_calls`, as one Slate inverse of the
    `M` block. On top of that it adds two things to `SCPC`.

    **Nullspaces.** `SCPC` reads `condensed_field_nullspace` from the
    application context and sets it on the condensed operator. GAMG on the
    displacement operator also needs the rigid-body modes as a near-nullspace,
    and CG on a singular operator needs the transpose nullspace to make the
    right-hand side consistent, so this class also reads
    `condensed_field_near_nullspace` and `condensed_field_transpose_nullspace`.
    Each is a callable taking the condensed space and returning a
    `VectorSpaceBasis`; `CoupledInternalVariableSolver` sets them from its
    `nullspace`, `transpose_nullspace` and `near_nullspace` arguments.

    **Operator reuse.** `SCPC.update` reassembles the condensed matrix on every
    preconditioner setup, which PETSc requests on every linear solve. A
    Newtonian march with a fixed time step has a constant Jacobian, so the
    condensed matrix and the GAMG hierarchy built on it can serve the whole
    march. This class reads `operator_version` from the application context
    and reassembles only when the version differs from the one the current
    matrix was built with. `None` means "always rebuild", which the solver
    publishes when the Jacobian depends on the solution (a power-law
    rheology, where the operator changes within one Newton solve). Without
    the key the class behaves as `SCPC` does. `assembly_count` records how
    many times the condensed operator has been assembled.

    This class reads the attributes `cxt`, `weight`, `S` and `S_pc` of `SCPC`,
    and `CoupledInternalVariableSolver` reads `_bases` of Firedrake's
    `MixedVectorSpaceBasis` to hand over the displacement nullspaces. Both
    were written against Firedrake main of 2026-08-06 (450aa7905); a
    Firedrake update can change any of these.

    Select with `pc_type: python`, `pc_python_type: gadopt.InternalVariableSCPC`
    and `pc_sc_eliminate_fields: "1"`. The operator must be matrix-free
    (`mat_type: matfree`).
    """

    def condensed_system(self, A, rhs, elim_fields, prefix, pc):
        blocks = A.blocks
        vectors = AssembledVector(rhs).blocks
        if list(elim_fields) != [1] or len(rhs.subfunctions) != 2:
            raise ValueError(
                "InternalVariableSCPC keeps the displacement (field 0) and "
                "eliminates the internal-variable field (field 1) of a two-field "
                f"space; got eliminated fields {list(elim_fields)} of "
                f"{len(rhs.subfunctions)}."
            )
        condensed_operator, inverse = internal_variable_condensation(A)
        condensed_rhs = vectors[0] - blocks[0, 1] * inverse * vectors[1]
        return LAContext(condensed_operator, condensed_rhs, (0,)), inverse

    def local_solver_calls(self, A, rhs, solution, elim_fields, inverse):
        displacement = AssembledVector(solution.subfunctions[0])
        field_rhs = AssembledVector(rhs.subfunctions[1])
        # Back-substitute: M = A_MM^-1 (b_M - A_Mu u), cell by cell.
        local_solution = inverse * (field_rhs - A.blocks[1, 0] * displacement)
        return [
            functools.partial(
                get_assembler(
                    local_solution, form_compiler_parameters=self.cxt.fc_params
                ).assemble,
                tensor=solution.subfunctions[1],
            )
        ]

    @PETSc.Log.EventDecorator("SCPCInit")
    def initialize(self, pc):
        """Build the condensed system, its solver and the local recovery.

        This is `SCPC.initialize` of Firedrake main (2026-08-06) with one
        change: a strong boundary condition on a *component* of the
        displacement (`ux`, `uy`, `uz` in the G-ADOPT boundary dictionary)
        is carried over to the same component of the condensed space, where
        `SCPC` refuses it because the component subspace has no field index
        of its own. `SCPC`'s guard against more than three fields is dropped
        on purpose: `condensed_system` enforces the two-field layout. Everything
        else is unchanged, so the attributes the rest of the class reads
        (`cxt`, `weight`, `S`, `S_pc`, `condensed_ksp`) are the ones `SCPC`
        defines.
        """
        from firedrake.bcs import DirichletBC
        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        from firedrake.functionspace import FunctionSpace
        from firedrake.matrix_free.operators import ImplicitMatrixContext
        from firedrake.parloops import par_loop, INC
        from firedrake.slate.slate import Tensor
        from ufl import dx
        import numpy as np

        prefix = (pc.getOptionsPrefix() or "") + "condensed_field_"
        A, P = pc.getOperators()
        self.cxt = A.getPythonContext()
        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        self.bilinear_form = self.cxt.a

        # Retrieve the mixed function space
        W = self.bilinear_form.arguments()[0].function_space()

        elim_option = (pc.getOptionsPrefix() or "") + "pc_sc_eliminate_fields"
        # By default, we condense down to the last field in the mixed space.
        elim_fields = PETSc.Options().getIntArray(elim_option, range(len(W) - 1))
        elim_fields = list(map(int, elim_fields))

        condensed_fields = list(set(range(len(W))) - set(elim_fields))
        if len(condensed_fields) != 1:
            raise NotImplementedError("Cannot condense to more than one field")

        c_field, = condensed_fields

        # Need to duplicate a space which is NOT
        # associated with a subspace of a mixed space.
        Vc = FunctionSpace(W.mesh()[c_field], W[c_field].ufl_element())
        bcs = []
        for bc in self.cxt.row_bcs:
            space = bc.function_space()
            # A component subspace (`Z.sub(0).sub(1)`) has no index of its
            # own; its field is the index of its parent and its component
            # selects the same component of the condensed space.
            component = space.component
            field = space.parent.index if component is not None else space.index
            if field != c_field:
                raise NotImplementedError("Strong BC set on unsupported space")
            target = Vc if component is None else Vc.sub(component)
            bcs.append(DirichletBC(target, 0, bc.sub_domain))

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        self.c_field = c_field
        self.condensed_rhs = Cofunction(Vc.dual())
        self.residual = Cofunction(W.dual())
        self.solution = Function(W)

        shapes = (Vc.finat_element.space_dimension(), np.prod(Vc.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.weight = Function(Vc)
        par_loop((domain, instructions), dx, {"w": (self.weight, INC)})
        with self.weight.dat.vec as wc:
            wc.reciprocal()

        # Get expressions for the condensed linear system
        A_tensor = Tensor(self.bilinear_form)
        reduced_sys, schur_builder = self.condensed_system(
            A_tensor, self.residual, elim_fields, prefix, pc
        )
        S_expr = reduced_sys.lhs
        r_expr = reduced_sys.rhs

        # Construct the condensed right-hand side
        self._assemble_Srhs = get_assembler(
            r_expr, bcs=bcs, form_compiler_parameters=self.cxt.fc_params
        ).assemble

        # Allocate and set the condensed operator
        form_assembler = get_assembler(
            S_expr,
            bcs=bcs,
            form_compiler_parameters=self.cxt.fc_params,
            mat_type=mat_type,
            options_prefix=prefix,
            appctx=self.get_appctx(pc),
        )
        self.S = form_assembler.allocate()
        self._assemble_S = form_assembler.assemble

        self._assemble_S(tensor=self.S)
        Smat = self.S.petscmat

        # If a different matrix is used for preconditioning,
        # assemble this as well
        if A != P:
            self.cxt_pc = P.getPythonContext()
            P_tensor = Tensor(self.cxt_pc.a)
            P_reduced_sys, _ = self.condensed_system(
                P_tensor, self.residual, elim_fields, prefix, pc
            )
            S_pc_expr = P_reduced_sys.lhs
            self.S_pc_expr = S_pc_expr

            # Allocate and set the condensed operator
            form_assembler = get_assembler(
                S_pc_expr,
                bcs=bcs,
                form_compiler_parameters=self.cxt.fc_params,
                mat_type=mat_type,
                options_prefix=prefix,
                appctx=self.get_appctx(pc),
            )
            self.S_pc = form_assembler.allocate()
            self._assemble_S_pc = form_assembler.assemble

            self._assemble_S_pc(tensor=self.S_pc)
            Smat_pc = self.S_pc.petscmat

        else:
            self.S_pc_expr = S_expr
            Smat_pc = Smat

        # Get nullspace for the condensed operator (if any).
        # This is provided as a user-specified callback which
        # returns the basis for the nullspace.
        nullspace = self.cxt.appctx.get("condensed_field_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(Vc)
            Smat.setNullSpace(nsp.nullspace())

        # Create a SNESContext for the DM associated with the trace problem
        self._ctx_ref = self.new_snes_ctx(
            pc, S_expr, bcs, mat_type, self.cxt.fc_params, options_prefix=prefix
        )

        # Push new context onto the dm associated with the condensed problem
        c_dm = Vc.dm

        # Set up ksp for the condensed problem
        c_ksp = PETSc.KSP().create(comm=pc.comm)
        c_ksp.incrementTabLevel(1, parent=pc)

        # Set the dm for the condensed solver
        c_ksp.setDM(c_dm)
        c_ksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
        c_ksp.setOptionsPrefix(prefix)
        c_ksp.setOperators(A=Smat, P=Smat_pc)
        self.condensed_ksp = c_ksp

        with dmhooks.add_hooks(c_dm, self, appctx=self._ctx_ref, save=False):
            c_ksp.setFromOptions()

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_calls(
            A_tensor, self.residual, self.solution, elim_fields, schur_builder
        )

        # Additions to `SCPC.initialize` start here.
        self.assembly_count = 1
        self._operator_version = self._published_version()

        appctx = self.cxt.appctx
        matrices = [Smat]
        if hasattr(self, "S_pc"):
            matrices.append(Smat_pc)
        transpose_nullspace = appctx.get("condensed_field_transpose_nullspace")
        if transpose_nullspace is not None:
            basis = transpose_nullspace(Vc).nullspace()
            for matrix in matrices:
                matrix.setTransposeNullSpace(basis)
        near_nullspace = appctx.get("condensed_field_near_nullspace")
        if near_nullspace is not None:
            basis = near_nullspace(Vc).nullspace()
            for matrix in matrices:
                matrix.setNearNullSpace(basis)

    def _published_version(self):
        """The operator version the solver publishes, or a sentinel.

        The sentinel is a fresh object, so a context without the key never
        compares equal to a stored version and the operator is rebuilt on
        every update, which is what `SCPC` does.
        """
        return self.cxt.appctx.get("operator_version", object())

    def update(self, pc):
        version = self._published_version()
        if version is not None and version == self._operator_version:
            return
        self._operator_version = version
        self.assembly_count += 1
        super().update(pc)


class CondensedBlockPC(fd.preconditioners.base.PCBase):
    r"""Block-0 preconditioner that eliminates the internal variables once.

    `SelfGravitatingGIASolver` on the full layout has a block 0 over the three
    fields `(u, M, psi)`: the displacement, the combined internal-variable
    history field and the gravitational potential. `M` is a DG field of shape
    `(n, d, d)` and carries about 85 percent of the block-0 unknowns, and its
    Jacobian block is block-diagonal per cell. This class does the whole
    block-0 solve, with `M` eliminated exactly once:

        r_u' = r_u - A_uM A_MM^-1 r_M                 Slate, cell-local
        solve [[S_uu, A_upsi], [A_psiu, A_psipsi]] (u, psi) = (r_u', r_psi)
        M    = A_MM^-1 (r_M - A_Mu u)                 Slate, cell-local

    with `S_uu = A_uu - A_uM A_MM^-1 A_Mu` the exact Schur complement that
    `internal_variable_condensation` writes. Block 0's own KSP is therefore
    `preonly`: one application of this class is one elimination, one Krylov
    solve on the `(u, psi)` system and one back-substitution.

    The alternative route, `InternalVariableSCPC` inside a two-split sweep over
    `(u, M)` and `psi`, is kept (`selfgrav_dtn_iterative_solver_parameters(
    condensed=False, block0="pair")`). There `M` rides in every block-0 Krylov
    vector and the elimination runs on every block-0 inner iteration, about
    900 times per production time step.

    **Everything here is preconditioner-side.** The residual, the mixed space,
    the tape and the adjoint are untouched, so the answer is the answer the
    direct preset gives and the only quantities that move are iteration counts
    and wall clock.

    ## The operator

    The four blocks are assembled sparse matrices held in a `PETSc.Mat` of
    type nest, in the order `(u, psi)`. The order is not cosmetic: the
    fieldsplit takes its index sets from the nest, so split 0 is the
    displacement block and split 1 the potential block, and a nest built the
    other way round would apply the displacement split's truncated CG and the
    near-incompressible modes to the potential Laplacian without raising
    anything.

    `S_uu` comes from the Slate expression; `A_upsi`, `A_psiu` and `A_psipsi`
    are sub-blocks of the block-0 bilinear form, extracted with
    `ExtractSubBlock` and assembled as `aij`. `A_psipsi` is the potential
    Laplacian with the DtN boundary terms, and it is marked SPD so that GAMG
    uses its symmetric path on it.

    ## Options

    All under this preconditioner's own prefix plus `condensed_` (under the
    shipped preset, `dtn_fieldsplit_0_condensed_`): `ksp_*` for the `(u, psi)`
    Krylov solve, `pc_type fieldsplit` with `pc_fieldsplit_type multiplicative`
    for the sweep, and `fieldsplit_0_*` / `fieldsplit_1_*` for the
    displacement and potential splits.

    ## The application context

    The key names are the ones `InternalVariableSCPC` reads, so
    `SelfGravitatingGIASolver._attach_condensation_context` publishes one set
    for both routes:

    - `operator_version`: the reuse fingerprint. A Newtonian march at fixed
      `dt` has a constant Jacobian, so the four blocks and the GAMG
      hierarchies built on them serve the whole march; the class reassembles
      only when the version moves. `None` means "rebuild on every update",
      which the solver publishes for a power law, where the Jacobian moves
      inside one Newton solve.
    - `condensed_field_nullspace`, `condensed_field_transpose_nullspace`,
      `condensed_field_near_nullspace`: each a callable taking the
      displacement space and returning a `VectorSpaceBasis`. The first two are
      set on `S_uu` so that the Krylov solve works on a consistent system (on
      a nonsymmetric condensed operator the left kernel is not the right one,
      which is why both are read); the third is the near-nullspace GAMG
      coarsens onto, the rigid-body modes plus the low-degree divergence-free
      fields by default.

    ## What is refused

    A block-0 space that does not have exactly three fields, which is what
    `condense_internal_variables=True` produces, and a strong boundary
    condition on anything but the displacement. A strong condition on the
    displacement is supported: it is carried into `S_uu` by the assembler, as
    `firedrake.SCPC` carries it into its condensed operator, and the rows it
    owns are emptied in the rectangular `A_upsi`. Production 3-D and every
    annulus test use the weak forms (`un`, `normal_stress`) instead.

    `applyTranspose` raises: pyadjoint solves `adjoint(J)` with the forward
    options, so an adjoint solve of this system reaches the forward `apply`,
    and a caller that does arrive at the transpose has reached a path this
    class does not cover.

    Select with `pc_type: python`, `pc_python_type: gadopt.CondensedBlockPC`
    and `ksp_type: preonly` on block 0. The block-0 operator must be
    matrix-free.
    """

    needs_python_pmat = True

    #: The field indices of the block-0 mixed space. Fixed by
    #: `self_gravitating_gia_space` on the full layout: displacement first,
    #: the combined history field second, the potential third.
    DISPLACEMENT, INTERNAL_VARIABLE, POTENTIAL = 0, 1, 2

    def initialize(self, pc):
        """Assemble the four blocks, build the nest and its Krylov solve."""
        from firedrake.bcs import DirichletBC
        from firedrake.cofunction import Cofunction
        from firedrake.function import Function
        from firedrake.functionspace import FunctionSpace
        from firedrake.formmanipulation import ExtractSubBlock
        from firedrake.matrix_free.operators import ImplicitMatrixContext
        from firedrake.parloops import par_loop, INC
        from firedrake.slate.slate import Tensor
        from ufl import dx as ufl_dx

        prefix = (pc.getOptionsPrefix() or "") + "condensed_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()
        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError(
                "gadopt.CondensedBlockPC needs a matrix-free block-0 operator: "
                "it reads the block-0 bilinear form off the operator's Python "
                f"context, and got {type(self.cxt).__name__}.")

        self.bilinear_form = self.cxt.a
        W = self.bilinear_form.arguments()[0].function_space()
        # The field count is the guard that catches the one mismatch nothing
        # else notices: the space and the preset are independent arguments,
        # and `condense_internal_variables=True` gives a block 0 of two fields
        # with no history field to eliminate.
        if len(W) != 3:
            raise ValueError(
                "gadopt.CondensedBlockPC eliminates the internal-variable "
                "field (field 1) of a three-field block-0 space "
                "(displacement, internal variables, potential); the space it "
                f"was given has {len(W)} field(s): "
                f"{[V.ufl_element() for V in W]}. A two-field block 0 is what "
                "self_gravitating_gia_space(condense_internal_variables=True) "
                "builds, and that layout has no internal variables to "
                "eliminate: either build the space with "
                "condense_internal_variables=False, or use "
                "selfgrav_dtn_iterative_solver_parameters(condensed=True), "
                "whose block 0 is a two-way sweep.")
        # Standalone copies of the two kept fields. The sub-spaces of a mixed
        # space carry the mixed dof layout, and the assembled blocks act on
        # the individual spaces. `W.mesh()[field]` rather than `W[field].mesh()`
        # because the self-gravity space spans two meshes: the potential lives
        # on the parent mesh and the displacement on the mantle submesh.
        self.displacement_space = FunctionSpace(
            W.mesh()[self.DISPLACEMENT],
            W[self.DISPLACEMENT].ufl_element())
        self.potential_space = FunctionSpace(
            W.mesh()[self.POTENTIAL], W[self.POTENTIAL].ufl_element())

        # A strong condition on the displacement is carried into the blocks
        # the preconditioner solves on, the way `firedrake.SCPC` carries it
        # into its condensed operator. Applying it to the residual and not to
        # these blocks would precondition a system the residual does not
        # describe: the constrained rows of `S_uu` would stay coupled to the
        # interior and the solve would either stall or converge to the wrong
        # fixed point, with nothing in the log to say so. Only a condition on
        # the displacement is meaningful here; the potential rows are
        # untouched by the elimination and the history field is cell-local.
        bcs = []
        for bc in self.cxt.row_bcs:
            space = bc.function_space()
            # A component subspace (`Z.sub(0).sub(1)`) has no index of its
            # own; its field is its parent's and its component selects the
            # same component of the standalone displacement space.
            component = space.component
            field = space.parent.index if component is not None else space.index
            if field != self.DISPLACEMENT:
                raise NotImplementedError(
                    "gadopt.CondensedBlockPC supports a strong boundary "
                    "condition on the displacement (field 0) only; got one on "
                    f"field {field}.")
            target = (self.displacement_space if component is None
                      else self.displacement_space.sub(component))
            bcs.append(DirichletBC(target, 0, bc.sub_domain))
        self._bcs = bcs
        # The rows of the rectangular `u`-`psi` coupling block that the
        # condition owns. `S_uu` gets an identity row there from the assembler,
        # so the coupling row must be empty for the two to describe one
        # equation `u = 0` on that degree of freedom. The columns of `A_psiu`
        # need no such treatment: the constrained entries of the incoming
        # residual are zero (the matrix-free block-0 operator eliminates them),
        # so the displacement this class returns is zero there and those
        # columns are multiplied by zero.
        self._constrained_rows = self._bc_rows(bcs)

        self.residual = Cofunction(W.dual())
        self.solution = Function(W)
        self.condensed_rhs = Cofunction(self.displacement_space.dual())

        # The multiplicity of every displacement degree of freedom: the number
        # of cells that hold it. A Slate vector expression is assembled cell by
        # cell and the contributions are summed, so a continuous field read
        # back through `AssembledVector` arrives multiplied by that count. The
        # incoming residual is therefore divided by it first, and the sum
        # reproduces it exactly. The internal-variable field is discontinuous,
        # so its multiplicity is one everywhere and it needs no such scaling.
        # This is what `firedrake.SCPC` does with the same name.
        shapes = (self.displacement_space.finat_element.space_dimension(),
                  np.prod(self.displacement_space.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.weight = Function(self.displacement_space)
        par_loop((domain, instructions), ufl_dx, {"w": (self.weight, INC)})
        with self.weight.dat.vec as weight:
            weight.reciprocal()

        # -- the Slate half: the elimination, the complement, the recovery --
        # Slate compiles an expression whose arguments live on ONE mesh, and
        # the block-0 space spans two (the potential on the parent mesh, the
        # displacement on the mantle submesh). So the Slate work is done on
        # the `(u, M)` sub-form, which is single-mesh on every configuration
        # because the history field lives on the displacement's mesh. That is
        # also exactly the form `InternalVariableSCPC` is handed by the
        # `"pair"` route's fieldsplit, so both routes condense the same two by
        # two system.
        splitter = ExtractSubBlock()
        pair_indices = (self.DISPLACEMENT, self.INTERNAL_VARIABLE)
        self._pair_form = _restrict_to_mesh(
            _split_mixed_coefficients(
                splitter.split(self.bilinear_form,
                               (pair_indices, pair_indices))),
            self.displacement_space.mesh())
        W_pair = self._pair_form.arguments()[0].function_space()
        # The Slate right-hand side and solution live on the `(u, M)` space;
        # the mixed `(u, M, psi)` residual and solution are the vectors PETSc
        # hands over. One field-wise copy each way per application keeps the
        # two layouts apart, and the copies are rank-local memory traffic on
        # fields the elimination touches anyway.
        self._pair_residual = Cofunction(W_pair.dual())
        self._pair_solution = Function(W_pair)

        A = Tensor(self._pair_form)
        blocks = A.blocks
        S_expr, inverse = internal_variable_condensation(A, keep=0,
                                                         eliminate=1)
        vectors = AssembledVector(self._pair_residual).blocks
        # r_u' = r_u - A_uM A_MM^-1 r_M, assembled once per application.
        rhs_expr = vectors[0] - blocks[0, 1] * inverse * vectors[1]
        # M = A_MM^-1 (r_M - A_Mu u), cell by cell, once per application.
        recovery_expr = inverse * (
            AssembledVector(self._pair_residual.subfunctions[1])
            - blocks[1, 0]
            * AssembledVector(self._pair_solution.subfunctions[0]))

        fcp = self.cxt.fc_params
        self._assemble_condensed_rhs = get_assembler(
            rhs_expr, bcs=self._bcs, form_compiler_parameters=fcp).assemble
        self._assemble_internal_variables = get_assembler(
            recovery_expr, form_compiler_parameters=fcp).assemble

        displacement_assembler = get_assembler(
            S_expr, bcs=self._bcs, form_compiler_parameters=fcp, mat_type="aij",
            options_prefix=prefix, appctx=self.cxt.appctx)
        self.S_uu = displacement_assembler.allocate()
        self._assemble_S_uu = displacement_assembler.assemble

        # -- the three blocks that carry no internal variable --------------
        # The potential rows of the block-0 Jacobian hold no `M` at all (the
        # history field enters the mechanics rows only), so these three are
        # plain sub-blocks of the bilinear form with no Schur correction.
        self._sub_forms = {
            name: splitter.split(self.bilinear_form, indices)
            for name, indices in (
                ("A_upsi", (self.DISPLACEMENT, self.POTENTIAL)),
                ("A_psiu", (self.POTENTIAL, self.DISPLACEMENT)),
                ("A_psipsi", (self.POTENTIAL, self.POTENTIAL)))}
        for name, form in self._sub_forms.items():
            setattr(self, name, fd.assemble(
                form, mat_type="aij", form_compiler_parameters=fcp))
        self._assemble_blocks()

        self._set_displacement_nullspaces()

        # -- the low-rank DtN update, when the solver carries one -----------
        # `SelfGravitatingGIASolver.build_dtn_operator` publishes this key on
        # the low-rank representation and never on the multiplier one, so its
        # absence keeps every existing configuration byte for byte.
        #
        # The update lives in the potential rows alone, so it lives in the
        # potential block of the nest and nothing else here changes. It has to
        # go into the PRECONDITIONING matrix and not into a separate operator
        # matrix: PCFieldSplit extracts both the split operator and the split
        # preconditioning matrix from `P` and ignores `A` entirely (measured,
        # `NOTES/team/lowrank-block0/nest_probe2.py`), so a nest handed over as
        # the operator alone would leave the potential split solving a system
        # with no update in it, converging, with nothing in any log to say so.
        self.dtn_operator = self.cxt.appctx.get("dtn_operator")
        if self.dtn_operator is not None:
            self.potential_operator = _LowRankPotentialOperator(
                self.A_psipsi.petscmat, self.dtn_operator)
            potential_block = PETSc.Mat().createPython(
                self.A_psipsi.petscmat.getSizes(), self.potential_operator,
                comm=pc.comm)
            potential_block.setUp()
        else:
            self.potential_operator = None
            potential_block = self.A_psipsi.petscmat

        # -- the nest and its Krylov solve ---------------------------------
        # The fieldsplit takes its index sets from the nest, so this order is
        # the split order: split 0 is the displacement block.
        self.condensed_operator = PETSc.Mat().createNest(
            [[self.S_uu.petscmat, self.A_upsi.petscmat],
             [self.A_psiu.petscmat, potential_block]], comm=pc.comm)
        self.condensed_operator.setUp()
        # The two work vectors are nest vectors over the blocks' own vectors,
        # built explicitly so that the sub-vectors stay reachable: `apply`
        # writes the eliminated right-hand side into them and reads the
        # solution back out of them, field by field.
        self._rhs_blocks = (self.S_uu.petscmat.createVecRight(),
                            self.A_psipsi.petscmat.createVecRight())
        self._solution_blocks = (self.S_uu.petscmat.createVecRight(),
                                 self.A_psipsi.petscmat.createVecRight())
        self._rhs = PETSc.Vec().createNest(list(self._rhs_blocks),
                                           comm=pc.comm)
        self._solution_vec = PETSc.Vec().createNest(
            list(self._solution_blocks), comm=pc.comm)

        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(self.condensed_operator, self.condensed_operator)
        ksp.setFromOptions()
        self.condensed_ksp = ksp

        #: Assemblies of the four blocks, including the one above: a Newtonian
        #: march at fixed `dt` must stay at 1, and a power law must reach the
        #: Newton iteration count.
        self.assembly_count = 1
        #: Eliminations of `M`, one per `apply`. The whole point of the route
        #: is that this counts applications and not inner iterations.
        self.elimination_count = 0
        self._operator_version = self._published_version()

    def _assemble_blocks(self):
        """Reassemble the four blocks in place, keeping their `PETSc.Mat`s.

        The nest and the KSP hold references to these matrices, so the
        assembly must write into them rather than build new ones.
        """
        self._assemble_S_uu(tensor=self.S_uu)
        for name, form in self._sub_forms.items():
            fd.assemble(form, tensor=getattr(self, name), mat_type="aij",
                        form_compiler_parameters=self.cxt.fc_params)
        if len(self._constrained_rows):
            # `diag=0.0`: the block is rectangular, so there is no diagonal to
            # write, and the constrained equation lives in `S_uu`'s identity
            # row alone.
            self.A_upsi.petscmat.zeroRows(self._constrained_rows, diag=0.0)
        # The potential block is a symmetric positive-definite Laplacian with
        # the DtN boundary terms; telling PETSc so lets GAMG use its symmetric
        # path (Chebyshev/Jacobi smoothing, no extra transpose products).
        # Both an assembly into the block and an assembly of the enclosing
        # nest clear a `Mat`'s symmetry flags, so the claim is made here,
        # after every reassembly, and not once in `initialize`. It is also
        # marked eternal: the sparsity and the sign structure of the potential
        # Laplacian do not depend on the Jacobian's values, so the claim holds
        # for every rebuild, and PETSc then keeps it across the assemblies
        # that follow. Without this, the first rebuild (which is every Newton
        # iteration of a power law) would silently move GAMG onto its
        # nonsymmetric path.
        self.A_psipsi.petscmat.setOption(PETSc.Mat.Option.SPD, True)
        self.A_psipsi.petscmat.setOption(
            PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

    def _bc_rows(self, bcs):
        """The global row indices the strong conditions own, as a PETSc array.

        Read off a marked `Function` rather than from the boundary condition's
        node lists, because a condition may act on one component of a vector
        space and the node list is then in nodes rather than in degrees of
        freedom. Marking and reading the vector gives the degree-of-freedom
        indices in the matrix's own numbering, with the local offset added.
        """
        if not bcs:
            return np.zeros(0, dtype=PETSc.IntType)
        marker = fd.Function(self.displacement_space)
        for bc in bcs:
            bc.set(marker, 1.0)
        with marker.dat.vec_ro as vec:
            low, _ = vec.getOwnershipRange()
            rows = low + np.flatnonzero(vec.array_r != 0.0)
        return rows.astype(PETSc.IntType)

    def _set_displacement_nullspaces(self):
        """Publish the kernels and the near-nullspace on `S_uu`.

        The condensed displacement operator is the matrix both the Krylov
        solve and GAMG see, so a basis that never reaches it is a basis that
        does nothing. The near-nullspace is what GAMG builds its coarse spaces
        to reproduce: the condensed operator carries the volumetric penalty of
        the internal-variable stress, whose slow modes sit in the
        divergence-free space once the effective bulk/shear ratio
        `bulk_shear_ratio * (1 + dt/tau)` is large, so the rigid modes alone
        leave the smoother nothing to coarsen those modes onto.
        """
        appctx = self.cxt.appctx
        matrix = self.S_uu.petscmat
        for key, setter in (
            ("condensed_field_nullspace", matrix.setNullSpace),
            ("condensed_field_transpose_nullspace",
             matrix.setTransposeNullSpace),
            ("condensed_field_near_nullspace", matrix.setNearNullSpace),
        ):
            provider = appctx.get(key)
            if provider is not None:
                setter(provider(self.displacement_space).nullspace())

    def _published_version(self):
        """The operator version the solver publishes, or a sentinel.

        The sentinel is a fresh object, so an application context without the
        key never compares equal to a stored version and the blocks are
        rebuilt on every update, which is the safe default.
        """
        return self.cxt.appctx.get("operator_version", object())

    def update(self, pc):
        """Reassemble the four blocks when the Jacobian has moved.

        PETSc calls this on every linear solve. `operator_version` says
        whether anything in the Jacobian changed since the blocks were built;
        `None` (a power law) means it changed and cannot be described, so
        rebuild.
        """
        version = self._published_version()
        if version is not None and version == self._operator_version:
            return
        self._operator_version = version
        self._assemble_blocks()
        self.assembly_count += 1
        # The nullspaces are attached to the matrix object and survive the
        # reassembly, but the near-nullspace providers may build their basis
        # from the space alone, so setting them again costs one interpolation
        # per rebuild and removes a dependence on PETSc's retention rules.
        self._set_displacement_nullspaces()
        # Bump the nest's object state and hand the operators over again, so
        # that PETSc rebuilds the GAMG hierarchies on the new entries instead
        # of preconditioning with the ones of the previous Jacobian.
        self.condensed_operator.assemble()
        self.condensed_ksp.setOperators(self.condensed_operator,
                                        self.condensed_operator)

    @PETSc.Log.EventDecorator("CondensedBlockPCApply")
    def apply(self, pc, x, y):
        """One block-0 application: eliminate, solve, back-substitute.

        Args:
          pc: the PETSc preconditioner object.
          x: the block-0 residual, on the mixed `(u, M, psi)` layout.
          y: the output, on the same layout. Not zero on entry.
        """
        with self.residual.dat.vec_wo as residual:
            x.copy(residual)
        # The Slate half works on the `(u, M)` space; move the two fields it
        # reads across, field by field, because the two spaces have the same
        # per-field dof layout and a different mixed one.
        for pair_field, mixed_field in enumerate(
                (self.DISPLACEMENT, self.INTERNAL_VARIABLE)):
            self._pair_residual.subfunctions[pair_field].dat.data_wo[...] = (
                self.residual.subfunctions[mixed_field].dat.data_ro)
        # Undo the multiplicity the cell-wise Slate assembly will reintroduce
        # on the continuous displacement field; see `self.weight`.
        with self._pair_residual.subfunctions[0].dat.vec as r_u, \
                self.weight.dat.vec_ro as weight:
            r_u.pointwiseMult(r_u, weight)

        # 1. The elimination, cell-local and exact.
        self._assemble_condensed_rhs(tensor=self.condensed_rhs)
        self.elimination_count += 1

        u_rhs, psi_rhs = self._rhs_blocks
        with self.condensed_rhs.dat.vec_ro as vec:
            vec.copy(u_rhs)
        with self.residual.subfunctions[self.POTENTIAL].dat.vec_ro as vec:
            vec.copy(psi_rhs)

        # 2. The `(u, psi)` Krylov solve on the assembled condensed system.
        self._solution_vec.set(0.0)
        self.condensed_ksp.solve(self._rhs, self._solution_vec)

        u_out, psi_out = self._solution_blocks
        with self.solution.subfunctions[self.DISPLACEMENT].dat.vec_wo as vec:
            u_out.copy(vec)
        with self.solution.subfunctions[self.POTENTIAL].dat.vec_wo as vec:
            psi_out.copy(vec)

        # 3. The back-substitution, once, on the displacement just computed.
        self._pair_solution.subfunctions[0].dat.data_wo[...] = (
            self.solution.subfunctions[self.DISPLACEMENT].dat.data_ro)
        self._assemble_internal_variables(
            tensor=self._pair_solution.subfunctions[1])
        self.solution.subfunctions[self.INTERNAL_VARIABLE].dat.data_wo[...] = (
            self._pair_solution.subfunctions[1].dat.data_ro)

        with self.solution.dat.vec_ro as solution:
            solution.copy(y)

    def applyTranspose(self, pc, x, y):
        """Refused; the transpose application is not implemented.

        pyadjoint solves `adjoint(J)` with the forward options, so an adjoint
        solve of this system reaches `apply` on the adjoint operator and never
        this method. A silent no-op here would write nothing into the output
        vector and leave PETSc preconditioning with whatever was in that
        memory, so the refusal is explicit.
        """
        raise NotImplementedError(
            "gadopt.CondensedBlockPC has no transpose application. An adjoint "
            "solve reaches the forward apply, because pyadjoint solves "
            "adjoint(J) with the forward solver options.")
