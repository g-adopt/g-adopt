import firedrake as fd
from firedrake.petsc import PETSc


class VariableMassInvPC(fd.PCBase):
    """A preconditioner that approximates the system by a scaled mass matrix.

    The mass matrix scaled by the inverse of viscosity is used to approximate the Stokes
    Schur complement. The viscosity needs to be supplied via the `mu` entry in the
    `appctx` dictionary.

    This is a variant of firedrake's `MassInvPC` that reassembles the scaled mass matrix
    everytime the Stokes system changes such that viscosity changes are taken into
    account in nonlinear solves and when reusing the preconditioner in subsequent
    solves. In the future, this variant will also be extended to include augmented
    lagrangian terms.

    To apply the inverse of this scaled mass matrix, an internal PETSc KSP object is
    created with prefix Mp_ that solves the mass matrix system.

    Attributes:
      needs_python_pmat:
        ???
    """
    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc:
            PETSc preconditioner.

        Raises:
          ValueError:
            Incompatible test and trial spaces.
        """
        prefix = pc.getOptionsPrefix()
        self.options_prefix = prefix + "Mp_"
        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError(
                "MassInvPC only makes sense if test and trial space are the same"
            )

        V = test.function_space()

        mu = context.appctx.get("mu", 1.0)
        self.fc_params = context.fc_params

        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        # Handle vector and tensor-valued spaces.

        # 1/mu goes into the inner product in case it varies spatially.
        self.a = fd.inner(1/mu * u, v)*fd.dx

        opts = PETSc.Options()
        mat_type = opts.getString(self.options_prefix + "mat_type",
                                  fd.parameters["default_matrix_type"])

        self.A = fd.assemble(self.a, form_compiler_parameters=self.fc_params,
                             mat_type=mat_type, options_prefix=self.options_prefix)

        Pmat = self.A.petscmat

        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(self.options_prefix)
        ksp.setFromOptions()
        ksp.setUp()
        self.ksp = ksp

    def update(self, pc: PETSc.PC):
        """Updates the matrix system.

        Args:
          pc:
            PETSc preconditioner.
        """
        fd.assemble(self.a, form_compiler_parameters=self.fc_params,
                    tensor=self.A, options_prefix=self.options_prefix)

    def apply(self, pc: PETSc.PC, X, Y):
        """Solves the preconditioned system.

        Args:
          pc:
            PETSc preconditioner.
          X:
            ???
          Y:
            ???
        """
        self.ksp.solve(X, Y)

    # Mass matrix is symmetric
    applyTranspose = apply

    def view(self, pc: PETSc.PC, viewer=None):
        """???

        Args:
          pc:
            PETSc preconditioner.
          viewer:
            ???
        """
        super().view(pc, viewer)
        viewer.printfASCII("KSP solver for M^-1\n")
        self.ksp.view(viewer)

    def destroy(self, pc: PETSc.PC):
        """Destroys the KSP object.

        Args:
          pc:
            PETSc preconditioner.
        """
        if hasattr(self, "ksp"):
            self.ksp.destroy()


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the fieldsplit_0 block in combination with gamg.
    """
    def initialize(self, pc: PETSc.PC):
        """Initialises the preconditioner.

        Args:
          pc:
            PETSc preconditioner.
        """
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)
