import firedrake as fd


class P0MassInv(fd.PCBase):
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
        assert isinstance(self.mu, fd.Constant)
        assert isinstance(self.gamma, fd.Constant)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        scaling = float(self.mu) + float(self.gamma)
        y.scale(-scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")


class SPDAssembledPC(fd.AssembledPC):
    """Version of AssembledPC that sets the SPD flag for the matrix.

    For use in the fieldsplit_0 block in combination with gamg."""
    def initialize(self, pc):
        super().initialize(pc)
        mat = self.P.petscmat
        mat.setOption(mat.Option.SPD, True)
