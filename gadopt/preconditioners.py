r"""This module contains classes that augment default Firedrake preconditioners.

Includes preconditioners for:
- Stokes: FreeSurfaceMassInvPC, SPDAssembledPC
- Richards: VerticallyLumpedPC, VerticallyLumpedHMGPC

"""

import firedrake as fd
from firedrake import (
    FiniteElement,
    Function,
    FunctionSpace,
    MixedFunctionSpace,
    PCBase,
    TensorProductElement,
    TrialFunction,
)
from firedrake.assemble import assemble
from firedrake.dmhooks import get_function_space
from firedrake.interpolation import interpolate
from firedrake.mg.utils import get_level
from firedrake.petsc import PETSc
from ufl.indexed import Indexed
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


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the velocity fieldsplit_0 block in combination with gamg.
    Setting PETSc MatOption MAT_SPD (for Symmetric Positive Definite matrices)
    at the moment only changes the Krylov method for the eigenvalue
    estimate in the Chebyshev smoothers to CG.

    Users can provide this class as a `pc_python_type`
    entry to a PETSc solver option dictionary.

    """
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc: PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


# =========================================================================
# Richards equation preconditioners
# =========================================================================


class VerticallyLumpedPC(PCBase):
    """Two-level MG that collapses the vertical dimension on the coarse level.

    A preconditioner specifically designed for high-aspect-ratio extruded
    meshes, inspired by Kramer et al. (2010, doi:10.1016/j.ocemod.2010.08.001)
    and the ocean modelling community. The idea:

        Fine level   : V = V_horiz x V_vert  (full 3D tensor-product space)
        Coarse level : V_c = V_horiz x R     (vertically constant)

    The prolongation P is the natural injection from V_c into V (a coarse
    DOF is replicated along its column). The coarse operator is formed by
    Galerkin projection, A_c = P^T A P, collapsing the entire vertical
    dimension onto a 2D-like problem solvable cheaply with MUMPS or GAMG.

    Solver options for the coarse level use the prefix ``lumped_mg_coarse_``
    and the fine-level smoother uses ``lumped_mg_levels_``.
    """

    def initialize(self, pc):
        options_prefix = pc.getOptionsPrefix()
        A, P = pc.getOperators()
        V = get_function_space(pc.getDM())

        # Handle both scalar and mixed spaces
        if len(V) == 1:
            V = FunctionSpace(V.mesh(), V.ufl_element())
        else:
            V = MixedFunctionSpace([V_ for V_ in V])

        # Build vertically constant function space (V_horiz x R)
        mesh = V.mesh()
        _, vcell = mesh.ufl_cell().sub_cells
        hele, _ = V.ufl_element().factor_elements
        vele = FiniteElement("R", vcell, 0)
        ele = TensorProductElement(hele, vele)
        V_coarse = FunctionSpace(mesh, ele)

        # Build prolongation: interpolation from V_coarse to V
        trial = TrialFunction(V_coarse)
        Prol = assemble(interpolate(trial, V)).petscmat

        # Set up 2-level multigrid
        self.pc = PETSc.PC().create(comm=pc.comm)
        self.pc.setOptionsPrefix(options_prefix + "lumped_")
        self.pc.incrementTabLevel(1, parent=pc)
        # Propagate the outer DM so level smoothers (e.g. ASMLinesmoothPC)
        # can resolve the fine function space.
        self.pc.setDM(pc.getDM())

        # Enable Galerkin coarse operator: A_c = P^T A P.
        # petsc4py has no setMGGalerkin() binding, so we go via the options DB.
        options = PETSc.Options()
        options[options_prefix + "lumped_pc_mg_galerkin"] = "both"

        self.pc.setOperators(A, P)
        self.pc.setType("mg")
        self.pc.setMGLevels(2)
        self.pc.setMGInterpolation(1, Prol)
        self.pc.setFromOptions()
        self.pc.setUp()
        self.update(pc)

    def update(self, pc):
        # Firedrake reassembles the Jacobian in-place every Newton iteration;
        # the inner PCMG holds a reference to the same Mat but needs setUp()
        # to recompute the Galerkin coarse operator A_c = P^T A P from the
        # updated state.
        self.pc.setUp()

    def apply(self, pc, X, Y):
        self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self.pc.applyTranspose(X, Y)

    def destroy(self, pc):
        if hasattr(self, "pc"):
            self.pc.destroy()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        viewer.printfASCII(
            "Vertically lumped MG: collapses vertical dimension on "
            "coarse level (cf. Kramer et al. 2010)\n"
        )
        self.pc.view(viewer)


class VerticallyLumpedHMGPC(PCBase):
    """Two-level MG with the coarse level on the fine mesh's 2D base.

    Same idea as ``VerticallyLumpedPC``, but the coarse level is built on
    the fine mesh's 2D base (via ``mesh._base_mesh``) rather than on the
    extruded mesh with an R-element vertically. The coarse KSP descends
    a full geometric multigrid on the 2D base ``MeshHierarchy``, giving
    optimal complexity for very large horizontal problems.

    Fine space : V         = hele x vele     (full 3D tensor-product)
    Coarse     : V_coarse  = hele on base_2d (same horizontal element)

    The prolongation V_coarse -> V is the same-mesh interpolation from the
    vertically-constant 3D space ``V_R = hele x R`` into ``V``. No explicit
    V_coarse -> V_R relabelling is needed: a ``hele x Real`` space and the
    horizontal ``hele`` space on the base share an identical PETSc DOF layout,
    so that interpolation matrix already has the right column space (see the
    detailed note in ``initialize``).

    Requires the fine mesh's base mesh to live in a ``MeshHierarchy``; if
    not, ``initialize`` raises ``RuntimeError``.
    """

    def initialize(self, pc):
        options_prefix = pc.getOptionsPrefix() or ""
        A, P = pc.getOperators()
        V = get_function_space(pc.getDM())

        # Handle both scalar and mixed spaces.
        if len(V) == 1:
            V = FunctionSpace(V.mesh(), V.ufl_element())
        else:
            V = MixedFunctionSpace([V_ for V_ in V])

        mesh = V.mesh()

        base_mesh = getattr(mesh, "_base_mesh", None)
        if base_mesh is None:
            raise RuntimeError(
                "VerticallyLumpedHMGPC requires an extruded fine mesh "
                "(mesh._base_mesh not found). Use VerticallyLumpedPC for "
                "non-extruded meshes."
            )

        hierarchy, level = get_level(base_mesh)
        if hierarchy is None or level is None:
            raise RuntimeError(
                "VerticallyLumpedHMGPC requires the fine mesh's base mesh "
                "to be part of a MeshHierarchy so the coarse PCMG can "
                "descend a geometric hierarchy. Build the mesh with "
                "ExtrudedMeshHierarchy(MeshHierarchy(base, L), ...) with "
                "L >= 1, or fall back to VerticallyLumpedPC."
            )

        hele, _ = V.ufl_element().factor_elements
        V_coarse = FunctionSpace(base_mesh, hele)
        self.V_coarse = V_coarse

        # Prolongation V_coarse (hele on the 2D base) -> V (full 3D).
        #
        # We build it as the same-mesh interpolation V_R -> V, where
        # V_R = hele x Real is the vertically-constant 3D space. That matrix
        # *is already* the prolongation we want: V_R and V_coarse share an
        # identical PETSc layout (owned size, ownership range, and lgmap
        # including ghost order), so no V_coarse -> V_R relabelling is needed.
        #
        # This is structural, not coincidental. An ExtrudedMeshTopology reuses
        # the base mesh's topology_dm, _dm_renumbering and _entity_classes, and
        # for a hele x Real element the section and node-class computations
        # collapse the vertical extent away (create_section on_base=True;
        # node_classes real_tensorproduct=True; zero dof offset), so V_R's
        # global numbering is the same computation over the same DM as
        # V_coarse's. Note a base<->extruded relabelling cannot be expressed as
        # cross-mesh interpolation anyway: that requires matching geometric
        # dimension (2D base vs 3D extruded).
        vcell = mesh.ufl_cell().sub_cells[1]
        vele_R = FiniteElement("R", vcell, 0)
        V_R = FunctionSpace(mesh, TensorProductElement(hele, vele_R))
        Prol = assemble(interpolate(TrialFunction(V_R), V)).petscmat
        self.Prol = Prol

        # Guard the layout assumption. If V_R and V_coarse ever stop sharing a
        # layout, a pure size mismatch would surface later as a cryptic PETSc
        # MatMatMult error in the galerkin setup; a same-size reordering would
        # be worse still (a silently wrong coarse operator). Fail fast here.
        with Function(V_coarse).dat.vec_ro as cvec:
            coarse_owned = cvec.getLocalSize()
        prol_cols_owned = Prol.getLocalSize()[1]
        if coarse_owned != prol_cols_owned:
            raise RuntimeError(
                "VerticallyLumpedHMGPC: prolongation column layout "
                f"({prol_cols_owned} owned) does not match the coarse space "
                f"V_coarse ({coarse_owned} owned). The hele x Real "
                "layout-collapse assumption has been violated."
            )

        # Set up the 2-level MG.
        self.pc = PETSc.PC().create(comm=pc.comm)
        self.pc.setOptionsPrefix(options_prefix + "lumped_")
        self.pc.incrementTabLevel(1, parent=pc)
        self.pc.setDM(pc.getDM())

        # Galerkin coarse operator: A_c = P^T A P.
        options = PETSc.Options()
        options[options_prefix + "lumped_pc_mg_galerkin"] = "both"

        self.pc.setOperators(A, P)
        self.pc.setType("mg")
        self.pc.setMGLevels(2)
        self.pc.setMGInterpolation(1, Prol)

        # Build the chain of horizontal interpolations along the base
        # MeshHierarchy (coarsest level is 0, V_coarse is at `level`).
        base_interp_mats = self._build_base_hierarchy_interpolations(
            hierarchy, level, hele
        )
        self._base_interp_mats = base_interp_mats

        # Explicit base-mesh interpolations + galerkin coarsening side-step
        # the DM-based coarsen hook, which would require a coarsened UFL
        # appctx we don't have.
        if base_interp_mats:
            coarse_ksp = self.pc.getMGCoarseSolve()
            coarse_pc = coarse_ksp.getPC()
            coarse_pc.setType("mg")
            nlevels_coarse = len(base_interp_mats) + 1
            coarse_pc.setMGLevels(nlevels_coarse)
            for i, Pi in enumerate(base_interp_mats):
                coarse_pc.setMGInterpolation(i + 1, Pi)
            coarse_pc.setOptionsPrefix(
                options_prefix + "lumped_mg_coarse_"
            )
            options = PETSc.Options()
            options[options_prefix + "lumped_mg_coarse_pc_mg_galerkin"] = "both"

        self.pc.setFromOptions()
        self.pc.setUp()
        self.update(pc)

    @staticmethod
    def _build_base_hierarchy_interpolations(hierarchy, fine_level, hele):
        """Assemble horizontal interpolation matrices for the base MG."""
        mats = []
        for k in range(1, fine_level + 1):
            mesh_c = hierarchy[k - 1]
            mesh_f = hierarchy[k]
            V_c = FunctionSpace(mesh_c, hele)
            V_f = FunctionSpace(mesh_f, hele)
            trial = TrialFunction(V_c)
            P_k = assemble(interpolate(trial, V_f)).petscmat
            mats.append(P_k)
        return mats

    def update(self, pc):
        self.pc.setUp()

    def apply(self, pc, X, Y):
        self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self.pc.applyTranspose(X, Y)

    def destroy(self, pc):
        if hasattr(self, "pc"):
            self.pc.destroy()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.pushASCIITab()
        viewer.printfASCII(
            "Vertically lumped MG (HMG variant): coarse level built on "
            "the 2D base of an ExtrudedMeshHierarchy, coarse solve is "
            "geometric MG on the base hierarchy.\n"
        )
        self.pc.view(viewer)
