r"""This module contains classes that augment default Firedrake preconditioners.

"""

import sys

import firedrake as fd
import numpy as np

try:
    from scipy.linalg import lu_factor as _lu_factor, lu_solve as _lu_solve
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy ships in the firedrake venv
    _HAVE_SCIPY = False
from ufl.indexed import Indexed
from firedrake.dmhooks import get_function_space
from firedrake.petsc import PETSc
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
    drops from the kwargs of the adjoint solve.

    Options for the inner fieldsplit are read under a `dtn_` prefix, e.g.
    `Gravity_dtn_fieldsplit_0_ksp_type`. Never supply `pc_fieldsplit_%d_fields`
    options there: the splits are already defined and the field lists would be
    silently ignored.

    Users can provide this class as a `pc_python_type` entry to a PETSc solver
    option dictionary; the preconditioning operator must be matrix-free.

    `update` is a no-op, which is correct for the gravitational Poisson solver
    because its Jacobian is constant by construction (see `update`). The class
    is therefore not intended for problems with a state-dependent Jacobian: on
    those, the assembled potential block would go stale silently rather than
    fail, and the inner preconditioner would need rebuilding here.
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

        The gravitational Poisson Jacobian is constant by construction: the
        density and the gravitational constant enter the residual only through
        terms linear in the test function, so they vanish under
        differentiation, and every remaining coefficient (the Robin shift, the
        DtN eigenvalues and the constraint-row scalings) is fixed when the form
        is built. A repeated setup would therefore only rebuild the index sets
        the preconditioner already holds.

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
        """Nothing to rebuild: the coupled Jacobian is constant by construction.

        Same argument as `DtNTwoBlockSchurPC.update`.
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
       the kwargs of the adjoint solve**. That is precisely why that class
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
                "  * this is an ADJOINT solve. pyadjoint drops appctx from the "
                "kwargs of the adjoint solve (see DtNTwoBlockSchurPC), so a "
                "preconditioner that reads appctx cannot be used on a taped "
                "replay. This is a loud failure by design; it must never "
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

    Unlike the diagonal PC, this one reads **nothing** from the appctx. It needs
    no `dtn_block1_diagonal`, no module global and no per-solver wiring: it forms
    S purely from the operator handed to it. That is the design rule this project
    settled on -- a preconditioner that reads its data off the operator has no
    appctx failure mode, so it survives a `pyadjoint` replay (which drops appctx)
    where the diagonal PC does not.

    ## Build once per time-step value

    The first build runs in `initialize`. Fixed-step solves keep that factor.
    `SelfGravitatingGIASolver` puts its live `dt` in the application context.
    If `dt.assign(...)` changes the value, `update` builds a new complement.
    A normal matrix reassembly with unchanged `dt` does not rebuild it.

    ## Two correctness conditions, and the second is not what the design assumed

    - The coupled Jacobian must stay constant while the time step stays fixed.
      A new time-step value changes the mechanics block and triggers a rebuild.
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
        self._time_step = self._current_time_step(pc)
        self._build(pc)

    def update(self, pc):
        """Rebuild the complement only after the time-step value changes."""
        time_step = self._current_time_step(pc)
        if time_step is None or self._time_step is None:
            return
        if time_step != self._time_step:
            self._time_step = time_step
            self._build(pc)

    def _current_time_step(self, pc):
        """Return the live GIA time-step value, if the solver supplies it."""
        value = self.get_appctx(pc).get("gia_time_step")
        return None if value is None else float(value)

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
