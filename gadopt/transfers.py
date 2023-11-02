import firedrake as fd
from pyop2.profiling import timed_function
from pyop2.data_types import IntType
from firedrake.mg.utils import get_level
import numpy as np
import ufl


def get_basemesh_nodes(W):
    pstart, pend = W.mesh()._topology_dm.getChart()
    section = W.dm.getDefaultSection()
    basemeshoff = {}
    basemeshdof = {}
    degree = W.ufl_element().degree()[1]
    nlayers = W.mesh().layers
    div = nlayers + (degree-1)*(nlayers-1)
    for p in range(pstart, pend):
        dof = section.getDof(p)
        off = section.getOffset(p)
        assert dof % div == 0
        basemeshoff[p] = off
        basemeshdof[p] = dof//div
    return basemeshoff, basemeshdof


@timed_function("fix_coarse_boundaries")
def fix_coarse_boundaries(V):
    if isinstance(V.mesh()._topology, fd.mesh.ExtrudedMeshTopology):
        hierarchy, level = get_level(V.mesh())
        dm = V.mesh()._topology_dm
        nlayers = V.mesh().layers
        degree = V.ufl_element().degree()[1]

        baseoff, basedof = get_basemesh_nodes(V)
        section = V.dm.getDefaultSection()
        indices = []
        fStart, fEnd = dm.getHeightStratum(1)
        # Spin over faces, if the face is marked with a magic label
        # value, it means it was in the coarse mesh.
        for p in range(fStart, fEnd):
            value = dm.getLabelValue("prolongation", p)
            if value > -1 and value <= level:
                # OK, so this is a coarse mesh face.
                # Grab all the points in the closure.
                closure, _ = dm.getTransitiveClosure(p)
                for c in closure:
                    # Now add all the dofs on that point to the list
                    # of boundary nodes.
                    dof = basedof[c]
                    off = baseoff[c]
                    for d in range(dof*degree*(nlayers-1)+dof):
                        indices.append(off + d)
        # indices now contains all the nodes on boundaries of course cells on the basemesh extruded up vertically
        # now we need to add every other horizontal layer as well
        for i in range(0, nlayers, 2):
            for p in basedof.keys():
                start = baseoff[p] + degree*i*basedof[p]
                indices += list(range(start, start+basedof[p]))
        nodelist = np.unique(indices).astype(IntType)
    else:
        hierarchy, level = get_level(V.mesh())
        dm = V.mesh()._topology_dm

        section = V.dm.getDefaultSection()
        indices = []
        fStart, fEnd = dm.getHeightStratum(1)
        # Spin over faces, if the face is marked with a magic label
        # value, it means it was in the coarse mesh.
        for p in range(fStart, fEnd):
            value = dm.getLabelValue("prolongation", p)
            if value > -1 and value <= level:
                # OK, so this is a coarse mesh face.
                # Grab all the points in the closure.
                closure, _ = dm.getTransitiveClosure(p)
                for c in closure:
                    # Now add all the dofs on that point to the list
                    # of boundary nodes.
                    dof = section.getDof(c)
                    off = section.getOffset(c)
                    for d in range(dof):
                        indices.append(off + d)
        nodelist = np.unique(indices).astype(IntType)

    class FixedDirichletBC(fd.DirichletBC):
        def __init__(self, V, g, nodelist):
            self.nodelist = nodelist
            super().__init__(self, V, g, "on_boundary")

        @fd.utils.cached_property
        def nodes(self):
            return self.nodelist

    bc = FixedDirichletBC(V, ufl.zero(V.ufl_element().value_shape()), nodelist)
    return bc


class AlgebraicSchoeberlTransfer(object):
    def __init__(self, parameters, A_callback, BTWB_callback, tdim, hierarchy, backend='tinyasm', hexmesh=False):
        assert backend in ['tinyasm', 'petscasm', 'lu']
        assert hierarchy == 'uniform'
        self.tdim = tdim
        self.solver = {}
        self.bcs = {}
        self.rhs = {}
        self.tensors = {}
        self.parameters = parameters
        self.prev_parameters = {}
        self.force_rebuild_d = {}
        patchparams = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "ksp_convergence_test": "skip",
            "mat_type": "aij",
        }
        if backend in ['tinyasm', 'petscasm']:
            backendparams = {
                "pc_type": "python",
                "pc_python_type": "alfi.transfer.ASMPCHexCoarseCellPatches" if hexmesh else "alfi.transfer.ASMPCCoarseCellPatches",
                "pc_coarsecell_backend": backend,
                # "pc_coarsecell_sub_pc_asm_sub_mat_type": "seqaij",
                # "pc_coarsecell_sub_sub_pc_factor_mat_solver_type": "umfpack",
            }
        else:
            backendparams = {
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        self.patchparams = {**patchparams, **backendparams}
        self.backend = backend

        self.A_callback = A_callback
        self.BTWB_callback = BTWB_callback
        self.called_first_restriction = 100 * [False]

    def break_ref_cycles(self):
        for attr in ["solver", "bcs", "rhs", "tensors", "parameters", "prev_parameters"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def force_rebuild(self):
        self.force_rebuild_d = {}
        for k in self.prev_parameters:
            self.force_rebuild_d[k] = True

    def rebuild(self, key):
        if key in self.force_rebuild_d and self.force_rebuild_d[key]:
            self.force_rebuild_d[key] = False
            fd.warn(fd.RED % ("Rebuild prolongation for key %i" % key))
            return True
        prev_parameters = self.prev_parameters.get(key, [])
        update = False
        for (prev_param, param) in zip(prev_parameters, self.parameters):
            if param != prev_param:
                update = True
                break
        return update

    @timed_function("SchoeberlProlong")
    def prolong(self, coarse, fine):
        self.restrict_or_prolong(coarse, fine, "prolong")

    @timed_function("SchoeberlRestrict")
    def restrict(self, fine, coarse):
        self.restrict_or_prolong(fine, coarse, "restrict")

    def restrict_or_prolong(self, source, target, mode):
        if mode == "prolong":
            coarse = source
            fine = target
        else:
            fine = source
            coarse = target
        # Rebuild without any indices
        V = fd.FunctionSpace(fine.ufl_domain(), fine.function_space().ufl_element())
        key = V.dim()

        firsttime = self.bcs.get(key, None) is None

        _, level = get_level(V.ufl_domain())

        # When building the PCMG, petsc restricts a constant vector of ones
        # from the finest level down the hierarchy. At this point the operators
        # on the levels haven't been built yet, so we just perform a standard
        # transfer here.
        if not self.called_first_restriction[level]:
            self.standard_transfer(source, target, mode)
            self.called_first_restriction[level] = True
            return
        if firsttime:
            from firedrake.solving_utils import _SNESContext

            A = self.A_callback(level)
            bcs = fix_coarse_boundaries(V)
            if self.backend == "lu":
                A.petscmat = A.petscmat.copy()
                for i in range(self.tdim):
                    A.petscmat.zeroRowsColumnsLocal(i+bcs.nodes*self.tdim, 1.0)

            BTWB = self.BTWB_callback(level)

            tildeu, rhs = fd.Function(V), fd.Function(V)

            b = fd.Function(V)
            a = A.form
            problem = fd.LinearVariationalProblem(a=a, L=0, u=tildeu, bcs=bcs)
            ctx = _SNESContext(problem, mat_type=self.patchparams["mat_type"],
                               pmat_type=self.patchparams["mat_type"],
                               appctx={}, options_prefix="prolongation")

            solver = fd.LinearSolver(A, solver_parameters=self.patchparams,
                                     options_prefix="prolongation")
            solver._ctx = ctx
            self.bcs[key] = bcs
            self.solver[key] = solver
            self.rhs[key] = tildeu, rhs
            self.tensors[key] = A, b, BTWB
            self.prev_parameters[key] = self.parameters
        else:
            bcs = self.bcs[key]
            solver = self.solver[key]
            A, b, BTWB = self.tensors[key]
            tildeu, rhs = self.rhs[key]

            # Update operator if parameters have changed.

            if self.rebuild(key):
                raise NotImplementedError("Gotta figure out if the PC needs to be forced to be rebuild")

                A = solver.A
                self.A_callback(level, mat=A)
                self.BTWB_callback(level, mat=BTWB)
                A.petscmat += BTWB
                for i in range(self.tdim):
                    A.petscmat.zeroRowsColumnsLocal(i+bcs.nodes*self.tdim, 1.0)
                self.tensors[key] = A, b, BTWB
                self.prev_parameters[key] = self.parameters

        if mode == "prolong":
            self.standard_transfer(coarse, rhs, "prolong")

            with rhs.dat.vec_ro as rhsvec:
                with b.dat.vec_wo as bvec:
                    BTWB.mult(rhsvec, bvec)
            bcs.apply(b)
            with solver.inserted_options(), fd.dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with b.dat.vec_ro as rhsv:
                    with tildeu.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            fine.dat.data[:] = rhs.dat.data_ro - tildeu.dat.data_ro

        else:
            tildeu.dat.data[:] = fine.dat.data_ro
            bcs.apply(tildeu)
            with solver.inserted_options(), fd.dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with tildeu.dat.vec_ro as rhsv:
                    with rhs.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            with rhs.dat.vec_ro as rhsvec:
                with b.dat.vec_wo as bvec:
                    BTWB.mult(rhsvec, bvec)
            rhs.dat.data[:] = fine.dat.data_ro - b.dat.data_ro
            self.standard_transfer(rhs, coarse, "restrict")

        def h1_norm(u):
            return fd.assemble(fd.inner(fd.grad(u), fd.grad(u)) * fd.dx)

        def energy_norm(u):
            v = u.copy(deepcopy=True)
            with u.dat.vec_ro as x:
                with v.dat.vec_wo as y:
                    A.petscmat.mult(x, y)
                    res = x.dot(y)
            return res

        def divergence_norm(u):
            v = u.copy(deepcopy=True)
            with u.dat.vec_ro as x:
                with v.dat.vec_wo as y:
                    BTWB.mult(x, y)
                    res = x.dot(y)
            return res

    def standard_transfer(self, source, target, mode):
        if mode == "prolong":
            fd.prolong(source, target)
        elif mode == "restrict":
            fd.restrict(source, target)
        else:
            raise NotImplementedError
