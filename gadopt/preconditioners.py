from firedrake.petsc import PETSc
import firedrake as fd


class P0MassInvPC(fd.PCBase):
    """Scaled inverse pressure mass preconditioner to be used with P0 pressure"""

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = fd.dmhooks.get_function_space(pc.getDM())
        # get function spaces
        assert V.ufl_element().degree() == 0
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        massinv = fd.assemble(fd.Tensor(fd.inner(u, v)*fd.dx).inv)
        self.massinv = massinv.petscmat
        self.mu = appctx["mu"]
        self.gamma = appctx["gamma"]
        if not isinstance(self.mu, fd.Constant):
            self.scale_func = fd.Function(V)

        assert isinstance(self.gamma, fd.Constant)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        if isinstance(self.mu, fd.Constant):
            scaling = float(self.mu) + float(self.gamma)
            y.scale(-scaling)
        else:
            self.scale_func.project(-(self.mu + self.gamma))
            with self.scale_func.dat.vec as scaling:
                y.pointwiseMult(y, scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")


class VariableMassInvPC(fd.PCBase):

    needs_python_pmat = False

    """A preconditioner that approximates the system by a scaled mass matrix.

    The mass matrix scaled by the inverse of viscosity is used to approximate
    the Stokes Schur complement. The viscosity needs to be supplied via the
    "mu" entry in the appctx dictionary.

    This is a variant of firedrake's MassInvPC that reassembles the scaled mass matrix
    everytime the Stokes system changes such that viscosity changes are taken into account
    in nonlinear solves and when reusing the preconditioner in subsequent solves. In the
    future this variant will also be extended to include augmented lagrangian terms.

    To apply the inverse of this scaled mass matrix
    an internal PETSc KSP object is created with prefix Mp_ that solves
    the mass matrix system.
    """
    def initialize(self, pc):
        prefix = pc.getOptionsPrefix()
        self.options_prefix = prefix + "Mp_"

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        Z = appctx['state'].function_space()
        u = fd.TestFunction(Z.sub(1))
        v = fd.TrialFunction(Z.sub(1))

        mu = appctx.get("mu", 1.0)
        self.fc_params = appctx['form_compiler_parameters']

        # Handle vector and tensor-valued spaces.

        # 1/mu goes into the inner product in case it varies spatially.
        self.a = fd.inner(1/mu * u, v)*fd.dx

        opts = PETSc.Options()
        mat_type = opts.getString(self.options_prefix + "mat_type",
                                  fd.parameters["default_matrix_type"])

        self.A = fd.assemble(self.a, form_compiler_parameters=self.fc_params,
                             mat_type=mat_type, options_prefix=self.options_prefix)

        Pmat = self.A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        self.nullsp = P.getNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(self.options_prefix)
        ksp.setFromOptions()
        ksp.setUp()
        self.ksp = ksp

        self.alc = appctx.get('alc')
        if self.alc:
            self.al_case = self.alc['alc_case']
        else:
            self.al_case = 0
        if self.al_case == 1:
            a2 = fd.inner(u, v)*fd.dx
            A2 = fd.assemble(a2, form_compiler_parameters=self.fc_params,
                             mat_type=mat_type, options_prefix=self.options_prefix2)
            Pmat2 = A2.petscmat
            Pmat2.setNullSpace(P.getNullSpace())
            tnullsp2 = P.getTransposeNullSpace()
            if tnullsp2.handle != 0:
                Pmat2.setTransposeNullSpace(tnullsp2)

            ksp2 = PETSc.KSP().create(comm=pc.comm)
            ksp2.incrementTabLevel(1, parent=pc)
            ksp2.setOperators(Pmat2)
            self.options_prefix2 = prefix + "Mp2_"
            ksp2.setOptionsPrefix(self.options_prefix2)
            ksp2.setFromOptions()
            ksp2.setUp()
            self.ksp2 = ksp2
            self.y2 = Pmat2.createVecLeft()

    def update(self, pc):
        fd.assemble(self.a, form_compiler_parameters=self.fc_params,
                    tensor=self.A, options_prefix=self.options_prefix)

    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)
        if self.al_case == 1:
            self.ksp2.solve(X, self.y2)
            Y.axpy(float(self.alc.gamma), self.y2)
        elif self.al_case == 2:
            Y.scale(1.0 + float(self.alc.gamma))

    # Mass matrix is symmetric
    applyTranspose = apply

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        viewer.printfASCII("KSP solver for M^-1\n")
        self.ksp.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "ksp"):
            self.ksp.destroy()


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the fieldsplit_0 block in combination with gamg."""
    def initialize(self, pc):
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)


class AugmentedAssembledPC(SPDAssembledPC):
    def initialize(self, pc):
        super().initialize(pc)
        appctx = self.get_appctx(pc)
        self.mu = appctx['mu']
        self.alc = appctx['alc']
        self.alc.augment_jacobian(self.P.petscmat)

    def update(self, pc):
        super().update(pc)
        self.alc.augment_jacobian(self.P.petscmat)


class AugmentedLagrangianContext:
    def __init__(self, F, z, gamma, bcs=None, al_case=1):
        self.F = F
        self.Z = z.function_space()
        self.gamma = gamma
        self.bcs = bcs
        self.al_case = al_case
        self.J = fd.derivative(F, z)
        fs = dict(fd.formmanipulation.split_form(self.J))
        self.a01 = fs[(0, 1)]
        self.a10 = fs[(1, 0)]
        self.a11 = fs[(1, 1)]
        # TODO: reassemble when needed?
        M = fd.assemble(self.J, bcs=bcs,mat_type='nest').petscmat
        M.getNestSubMatrix(0, 0).destroy()
        M.getNestSubMatrix(1, 1).destroy()
        self.A10 = M.getNestSubMatrix(1, 0)
        self.A01 = M.getNestSubMatrix(0, 1)
        self.A01_Winv = None
        self.augl = None

    def augment_jacobian(self, Kmat):
        print("Augmenting Jacobian!")
        Winv = self.get_Winv_mat().petscmat
        A01 = self.A01
        A10 = self.A10
        self.A01_Winv = A01.matMult(Winv, self.A01_Winv, 2.0)
        self.augl = self.A01_Winv.matMult(A10, self.augl, 2.0)
        Kmat.axpy(float(self.gamma), self.augl, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

    def augment_residual(self, res):
        is_u, is_p = self.Z._ises
        res_u = res.getSubVector(is_u)
        res_p = res.getSubVector(is_p)
        Winv = self.get_Winv_mat()
        res_u += self.A01 * (Winv.petscmat * res_p)
        res.restoreSubVector(is_u, res_u)
        res.restoreSubVector(is_p, res_p)

    def post_function_callback(self, X, F):
        self.augment_residual(F)

    def post_jacobian_callback(self, X, J):
        pass

    def get_Winv_mat(self):
        W = fd.FunctionSpace(self.Z.mesh(), self.Z.sub(1).ufl_element())
        p = fd.TrialFunction(W)
        q = fd.TestFunction(W)
        if self.al_case == 1:
            Winv = fd.assemble(fd.Tensor(fd.inner(p, q)*fd.dx).inv, mat_type='aij')
        elif self.al_case == 2:
            Winv = fd.assemble(fd.Tensor(1/self.mu*fd.inner(p, q)*fd.dx).inv, mat_type='aij')
        else:
            raise NotImplementedError(f"Unknown al_case=={self.al_case}")
        return Winv
