"""Riesz-map preconditioner for the TAO bounded Newton types, under bound constraints.

Why this file exists
--------------------

``pyadjoint.TAOSolver`` gives the reduced-functional Hessian to TAO as a Python-context
``Mat`` and the gradient in the dual space. For the bounded Newton types (``bnls``,
``bntr``, ``bntl``) the Krylov solve therefore runs on a dual-space right-hand side in
nodal coordinates, and the only place the control inner product can enter is the Krylov
preconditioner. pyadjoint has ``RieszMapPC`` for that, but it cannot be used as soon as
a bound is active:

- ``bnk.c:19`` calls ``MatCreateSubMatrix(tao->hessian, inactive, inactive, ...)`` and
  the same on ``tao->hessian_pre``, so that the Newton step is taken in the inactive
  set only.
- A Python ``Mat`` with no ``createSubMatrix`` method makes PETSc fall back to
  ``MatCreateSubMatrixVirtual``. The virtual sub-matrix has a different Python context,
  so ``RieszMapPC.initialize`` raises ``TypeError: RieszMapPC needs a
  ReducedFunctionalHessianMat``.

The control of this benchmark is a temperature field with bounds [0, 1], so bounds do
become active and the fallback is always reached.

What this file does
-------------------

It supplies a separate preconditioning matrix, passed to ``TAOSolver`` through its
``Pmat`` argument, and a Python preconditioner that inverts it.

- ``InnerProductMat`` is the Python context of that matrix. Its action is the control
  inner product ``A``, that is the mass matrix for ``--riesz L2`` and
  ``2 c_d M + 2 c_s K`` for ``--riesz regularisation``. It implements
  ``createSubMatrix``, so the matrix that BNK hands to the Krylov solver is again a
  Python matrix with a context this file understands.
- ``MaskedInnerProductMat`` is the context of that sub-matrix. It masks the active
  dofs: it scatters the reduced vector into a full vector with zeros in the active
  set, applies the full operation, and scatters back.
- ``RieszPC`` is the preconditioner. It applies ``A^{-1}``, that is the Riesz map of
  the control, restricted to the inactive set in the same way.

``Z^T A^{-1} Z``, with ``Z`` the injection of the inactive set, is symmetric and
positive definite whenever ``A`` is, so it is a valid preconditioner for the reduced
Hessian ``Z^T H Z``. It is not the inverse of ``Z^T A Z``; a preconditioner does not
have to be.

The Newton types regularise an indefinite Hessian with ``MatShift`` (``bnk.c:414-421``
at PETSc v3.25.0): before each Krylov solve with a positive perturbation ``pert``, BNK
calls ``MatShift(H_inactive, pert)`` and, because the preconditioning matrix is a
different matrix here, ``MatShift(Hpre_inactive, pert)`` as well. Both contexts here
leave their operator unchanged by that call: the only operation PETSc ever performs
on this matrix is the preconditioner, which applies ``A^{-1}``, and ``A^{-1}``
preconditions ``H + pert I`` as well as it preconditions ``H``. They do not
accumulate the value either, because BNK shifts the preconditioning matrix once per
solve while pyadjoint's Hessian ``update`` resets the shift of the Hessian matrix
only (``tao_solver.py:227-232``), so an accumulated value would never be cleared.

What the contexts do with the value is report it. ``MatShift`` on a Python matrix
calls the ``shift`` method of its context with the exact ``pert`` of BNK, and BNK
keeps that number in a private struct that petsc4py cannot read. So the call is the
one exact view of the perturbation from Python: ``shift`` hands ``pert`` to the
callable ``shift_observer`` of the full-space context, if one is set.
``recording.KrylovRecorder`` sets it and stores the value with the next Krylov
solve. No ``shift`` call before a solve means that BNK used no perturbation for it.

Everything here is a local prototype on this branch. The upstream form of the same
work is ``createSubMatrix`` on ``ReducedFunctionalHessianMat`` in
``pyadjoint/optimization/tao_solver.py``.
"""

from firedrake import Cofunction, Function, TestFunction, assemble
from petsc4py import PETSc

from pyadjoint.optimization.tao_solver import PETScVecInterface


class InnerProductMat:
    """Python ``Mat`` context whose action is the control inner product.

    This is the preconditioning matrix of the Newton-Krylov solve. Its action maps a
    primal control field to a dual one, ``x -> A x``, which is the same direction as
    the Hessian in the coordinates TAO works in, so it is a valid stand-in for the
    Hessian as a preconditioning matrix.

    Args:
        inner_product: the ``InnerProduct`` of ``problem.py``. Its ``form`` is the
            bilinear form of the metric and its ``riesz_map`` inverts it.
        control_space: the control ``FunctionSpace``.
        comm: MPI communicator.

    """

    def __init__(self, inner_product, control_space, comm):
        self.inner_product = inner_product
        self.control_space = control_space
        self.comm = comm
        # One vector interface for the primal side and one for the dual side. Both
        # spaces have the same dof layout, so one interface serves for both; the two
        # names are kept apart only for readability at the call sites.
        self.vec_interface = PETScVecInterface(Function(control_space), comm=comm)
        # A callable ``observer(r, z)`` that ``RieszPC.apply`` calls after each
        # application with the input and the output, or None. ``recording.py`` sets
        # it to rebuild the conjugate gradient coefficients. It only reads the two
        # vectors, so the solve is the same with or without it.
        self.observer = None
        # A callable ``shift_observer(alpha)`` that ``shift`` calls with the value
        # of each ``MatShift`` that BNK applies to this matrix or to its masked
        # sub-matrix, or None. ``recording.py`` sets it to record BNK's Hessian
        # perturbation. It only receives a number, so the solve is the same with
        # or without it.
        self.shift_observer = None

    # -- sizes ---------------------------------------------------------------------

    @property
    def sizes(self):
        """``((n, N), (n, N))``: the local and global dof counts, square."""
        n, N = self.vec_interface.n, self.vec_interface.N
        return ((n, N), (n, N))

    def create_mat(self):
        """Build the PETSc ``Mat`` that carries this context."""
        mat = PETSc.Mat().createPython(self.sizes, self, comm=self.comm)
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        mat.setUp()
        return mat

    # -- matrix operations ---------------------------------------------------------

    def shift(self, A, alpha):
        """Report ``A <- A + alpha I`` from the Newton regularisation, and do not apply it.

        See the module docstring: only ``A^{-1}`` is ever applied, and it is a
        preconditioner for the shifted Hessian as much as for the unshifted one.
        BNK calls this with its perturbation ``pert`` just before a Krylov solve
        when no bound is active, because the full matrix is then the
        preconditioning matrix of the solve. The value goes to ``shift_observer``.

        Args:
            A: the PETSc ``Mat`` that carries this context.
            alpha: the shift, BNK's ``pert``, in the nodal (Euclidean) units of the
                Hessian matrix: ``MatShift`` adds ``alpha`` to every diagonal entry.
        """
        if self.shift_observer is not None:
            self.shift_observer(float(alpha))

    def mult(self, A, x, y):
        """``y = A x``, with ``x`` a primal control vector and ``y`` a dual one.

        The Newton shift is not part of this action, as ``shift`` says. With
        Dirichlet conditions in the inner product (``--prior-wall-bc dirichlet``)
        the action is that of the operator the Riesz map inverts: ``A_ff x_f`` on
        the free nodes and zero on the constrained ones, with the entries of ``x``
        on the constrained nodes ignored. PETSc never calls this method for the
        preconditioner, which applies only ``A^{-1}``. ``ShiftedHessianMat`` calls
        it to add ``pert * A x`` to the Hessian action.
        """
        u = Function(self.control_space)
        self.vec_interface.from_petsc(x, u)
        bcs = getattr(self.inner_product, "bcs", None) or []
        for bc in bcs:
            # Homogeneous conditions: the constrained entries of x take no part.
            bc.zero(u)
        # ``form(u, v)`` with u a Function and v a TestFunction is a linear form, so
        # its assembly is the dual object A u.
        v = assemble(self.inner_product.form(u, TestFunction(self.control_space)))
        for bc in bcs:
            # Zero rows on the constrained nodes, so that A x lies in the range of
            # the operator that the Riesz map inverts.
            v.dat.data_wo_with_halos[bc.nodes] = 0.0
        self.vec_interface.to_petsc(y, v)

    def multHermitian(self, A, x, y):
        """The metric is symmetric, so the Hermitian action is the same action."""
        self.mult(A, x, y)

    def apply_riesz(self, x, y):
        """``y = A^{-1} x``: the Riesz map, from a dual vector to a primal one.

        The Newton shift plays no part here, as the module docstring explains.
        """
        dual = Cofunction(self.control_space.dual())
        self.vec_interface.from_petsc(x, dual)
        primal = self.inner_product.riesz_map(dual)
        self.vec_interface.to_petsc(y, primal)

    def createSubMatrix(self, A, isrow, iscol, submat):
        """Return the sub-matrix on the inactive set as a Python ``Mat``.

        BNK asks for the square sub-matrix on the inactive dofs at every iteration
        where the active set changes. Returning a Python ``Mat`` with a context this
        module understands is what lets ``RieszPC`` still find the Riesz map; the
        PETSc fallback would return a virtual matrix with an opaque context.

        Args:
            A: the parent ``Mat``.
            isrow: the index set of the inactive dofs (rows).
            iscol: the same for columns. BNK always passes the same set.
            submat: an existing sub-matrix to reuse, or None.

        Returns:
            PETSc.Mat
        """
        if submat is not None:
            context = submat.getPythonContext()
            context.reset(isrow)
            return submat

        context = MaskedInnerProductMat(self, isrow)
        return context.create_mat()


class MaskedInnerProductMat:
    """Python ``Mat`` context for the inner product restricted to the inactive dofs.

    The action is ``Z^T A Z`` with ``Z`` the injection of the inactive set into the
    full control space: scatter the reduced vector into a full vector that is zero on
    the active set, apply the full operation, and scatter the result back.

    Args:
        parent: the ``InnerProductMat`` of the full space.
        iset: the index set of the inactive dofs.
    """

    def __init__(self, parent, iset):
        self.parent = parent
        self.x_full = parent.vec_interface.new_petsc()
        self.y_full = parent.vec_interface.new_petsc()
        self.iset = None
        self.x_reduced = None
        self.scatter = None
        self.reset(iset)

    def reset(self, iset):
        """Point the context at a new inactive set, keeping the full work vectors."""
        self.iset = iset
        self.x_reduced = PETSc.Vec().create(comm=self.parent.comm)
        self.x_reduced.setSizes((iset.getLocalSize(), iset.getSize()))
        self.x_reduced.setUp()
        # Forward scatter takes the inactive entries out of a full vector; the reverse
        # of the same scatter puts a reduced vector back into a full one.
        self.scatter = PETSc.Scatter().create(self.x_full, self.iset, self.x_reduced, None)

    @property
    def observer(self):
        """The observer of the full-space context. See ``InnerProductMat``."""
        return self.parent.observer

    @property
    def sizes(self):
        """``((n, N), (n, N))`` of the reduced space."""
        n, N = self.iset.getLocalSize(), self.iset.getSize()
        return ((n, N), (n, N))

    def create_mat(self):
        """Build the PETSc ``Mat`` that carries this context."""
        mat = PETSc.Mat().createPython(self.sizes, self, comm=self.parent.comm)
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        mat.setUp()
        return mat

    def _expand(self, x_reduced):
        """Scatter a reduced vector into ``x_full``, zero outside the inactive set."""
        self.x_full.set(0.0)
        self.scatter.scatter(x_reduced, self.x_full,
                             addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.REVERSE)
        return self.x_full

    def _restrict(self, y_reduced):
        """Scatter ``y_full`` back into a reduced vector."""
        self.scatter.scatter(self.y_full, y_reduced,
                             addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)

    def shift(self, A, alpha):
        """Report the Newton shift to the full-space context, and do not apply it.

        BNK shifts this masked sub-matrix instead of the full matrix whenever a
        bound is active, so the value goes to the same observer as there.
        """
        self.parent.shift(None, alpha)

    def mult(self, A, x, y):
        """``y = Z^T A Z x`` on the inactive set, with no Newton shift."""
        self.parent.mult(None, self._expand(x), self.y_full)
        self._restrict(y)

    def multHermitian(self, A, x, y):
        """Symmetric, so the same action."""
        self.mult(A, x, y)

    def apply_riesz(self, x, y):
        """``y = Z^T A^{-1} Z x`` on the inactive set."""
        self.parent.apply_riesz(self._expand(x), self.y_full)
        self._restrict(y)


class RieszPC:
    """PETSc Python preconditioner that applies the control Riesz map.

    Selected with the options ``pc_type python`` and
    ``pc_python_type riesz_hessian.RieszPC``, under the ``tao_bnk_`` prefix for the
    TAO bounded Newton types. PETSc builds it with no arguments, so everything it
    needs comes from the preconditioning matrix, whose Python context is either an
    ``InnerProductMat`` or a ``MaskedInnerProductMat``.
    """

    def setUp(self, pc):
        """Read the preconditioning matrix. Called again whenever it changes."""
        _, P = pc.getOperators()
        self.context = P.getPythonContext()

    def apply(self, pc, x, y):
        """``y = A^{-1} x``, restricted to the inactive set when there is one."""
        # setUp is called by PETSc before apply, but the operators can be replaced
        # between solves, so the context is read again here.
        _, P = pc.getOperators()
        context = P.getPythonContext()
        context.apply_riesz(x, y)
        # Hand the residual and the preconditioned residual to the recorder, if one
        # is attached. Both vectors are in the reduced space of the Krylov solve.
        observer = getattr(context, "observer", None)
        if observer is not None:
            observer(x, y)

    def view(self, pc, viewer=None):
        """Print what this preconditioner is, under ``-tao_view``."""
        if viewer is None or viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Riesz map of the control inner product\n")


class ShiftedHessianMat:
    """Python ``Mat`` context for the Hessian with BNK's shift taken in the metric.

    BNK regularises an indefinite Hessian by ``MatShift(H, pert)``, which adds
    ``pert * I`` in nodal coordinates, and it sizes ``pert`` from
    ``gnorm = sqrt(g^T A^{-1} g)``, the norm of the gradient in the metric ``A`` of
    the control (``TaoGradientNorm`` with pyadjoint's Riesz map as the gradient-norm
    matrix). The nodal Hessian of a CG1 control scales like the mass matrix,
    ``h^2``, while ``gnorm`` does not depend on the mesh, so ``pert * I`` is far
    larger than the Hessian: 2100 times on smooth directions at 64 cells and
    12000 times at 150 cells (step S2b of the box campaign). ``pert * A`` has the
    scale that BNK's rule assumes: its size against ``H`` is ``pert`` against the
    eigenvalues of ``A^{-1} H``, and both are mesh independent.

    This context is installed as the Hessian of the TAO object, in place of
    pyadjoint's ``ReducedFunctionalHessianMat``. Its action is

        y = H x + pert * A x,

    with ``H x`` from pyadjoint's Hessian matrix and ``A x`` from the same
    ``InnerProductMat`` whose inverse is the Krylov preconditioner. The preconditioned
    operator is then ``A^{-1} H + pert I``. With active bounds the preconditioner is
    ``Z^T A^{-1} Z`` and the operator ``Z^T (H + pert A) Z``, so that identity holds
    only approximately; the preconditioner stays symmetric positive definite.

    ``shift`` SETS ``pert``. BNK calls ``MatShift`` once per Krylov solve with its
    current ``pert``, and the bounded trust-region types (BNTR, BNTL) call it again
    for a re-solve with the same Hessian, so a sum would count the shift twice.
    ``update``, the Hessian callback of TAO, resets ``pert`` to zero, because BNK
    makes no shift call at all for a solve with ``pert = 0``.

    ``createSubMatrix`` returns a ``MaskedShiftedHessianMat``. Without it, PETSc
    would build a virtual sub-matrix (``MatCreateSubMatrixVirtual``) that wraps this
    matrix in a shell and keeps the shift in that shell, so ``shift`` here would
    never be called and the shift would be the identity again.

    The context defines the same operations as pyadjoint's Hessian matrix and no
    others (no ``getDiagonal``), so BNK's active-set estimate takes the same branch
    with either matrix.

    Args:
        hessian_mat: pyadjoint's Hessian ``Mat``, from ``TAOSolver``.
        metric: the ``InnerProductMat`` of the preconditioning matrix.

    Attributes:
        pert: the shift that the next action applies.
    """

    def __init__(self, hessian_mat, metric):
        self.hessian_mat = hessian_mat
        self.hessian_context = hessian_mat.getPythonContext()
        self.metric = metric
        self.pert = 0.0
        # A work vector of the full control space for A x.
        self.work = metric.vec_interface.new_petsc()

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

    def update(self, tao, x, H, P):
        """The Hessian callback of TAO: re-tape at ``x``, and clear the shift.

        pyadjoint's own callback reads its context from the matrix it is given, so
        it gets pyadjoint's matrix here and not the matrix of this context.
        """
        type(self.hessian_context).update(tao, x, self.hessian_mat, P)
        self.pert = 0.0

    def shift(self, A, alpha):
        """Set the shift to ``alpha``, BNK's ``pert``. See the class docstring."""
        self.pert = float(alpha)

    def mult(self, A, x, y):
        """``y = H x + pert * A x``.

        pyadjoint's matrix adds its own shift, which stays zero because nothing
        calls its ``shift`` once this context is the Hessian of TAO.
        """
        self.hessian_context.mult(self.hessian_mat, x, y)
        if self.pert != 0.0:
            self.metric.mult(None, x, self.work)
            y.axpy(self.pert, self.work)

    def multHermitian(self, A, x, y):
        """Symmetric, so the same action."""
        self.mult(A, x, y)

    def createSubMatrix(self, A, isrow, iscol, submat):
        """The operator on the inactive set, ``Z^T (H + pert A) Z``, as a Python ``Mat``."""
        if submat is not None:
            submat.getPythonContext().reset(isrow)
            return submat
        return MaskedShiftedHessianMat(self, isrow).create_mat()


class MaskedShiftedHessianMat(MaskedInnerProductMat):
    """``Z^T (H + pert A) Z`` on the inactive set of BNK, for ``ShiftedHessianMat``.

    The scatter between the full and the reduced vectors is the one of
    ``MaskedInnerProductMat``: a reduced vector is expanded into a full vector that
    is zero on the active set, the full operator is applied, and the result is
    restricted again. ``shift`` sets the shift of the full-space context, because
    BNK shifts this sub-matrix and not the full matrix whenever a bound is active.
    """

    def __init__(self, parent, iset):
        # ``MaskedInnerProductMat`` reads ``vec_interface`` and ``comm`` from its
        # parent; both are the ones of the metric.
        self.shifted = parent
        super().__init__(parent.metric, iset)

    def shift(self, A, alpha):
        """Set the shift of the full operator to ``alpha``."""
        self.shifted.shift(None, alpha)

    def mult(self, A, x, y):
        """``y = Z^T (H + pert A) Z x``."""
        self.shifted.mult(None, self._expand(x), self.y_full)
        self._restrict(y)

    def apply_riesz(self, x, y):
        """Not a preconditioning matrix; the Riesz map is never applied here."""
        raise NotImplementedError("MaskedShiftedHessianMat is a Hessian, not a preconditioner")


def build_shifted_hessian(tao, metric):
    """Install ``ShiftedHessianMat`` as the Hessian of a TAO object built by ``TAOSolver``.

    Args:
        tao: the TAO object of ``TAOSolver``, after its ``setUp``.
        metric: the ``InnerProductMat`` of the preconditioning matrix.

    Returns:
        ``(mat, context)``. The caller must keep both alive for the whole solve.
    """
    hessian_mat, pmat = tao.getHessian()[:2]
    context = ShiftedHessianMat(hessian_mat, metric)
    mat = context.create_mat()
    tao.setHessian(context.update, H=mat, P=pmat)
    return mat, context


def build_preconditioner_matrix(problem, comm=None):
    """Build the preconditioning matrix for ``TAOSolver(..., Pmat=...)``.

    Args:
        problem: the ``InverseProblem`` of ``problem.py``.
        comm: MPI communicator; defaults to the communicator of the control mesh.

    Returns:
        ``(mat, context)``. The caller must keep both alive for the whole solve,
        because a petsc4py ``TAO`` cannot carry Python attributes.
    """
    control_space = problem.Tic.function_space()
    if comm is None:
        comm = control_space.mesh().comm
    context = InnerProductMat(problem.inner_product, control_space, comm)
    return context.create_mat(), context
