from alfi.transfer import AutoSchoeberlTransfer, NullTransfer, BubbleTransfer
from firedrake.mg.utils import get_level

__all__ = ['VariableSchoeberlTransfer', 'VariablePkP0SchoeberlTransfer', 'NullTransfer']


class VariableSchoeberlTransfer(AutoSchoeberlTransfer):
    def __init__(self, tdim, hierarchy):
        super().__init__([], tdim, hierarchy)
        self.forms = {}
        self.statuses = {}

    def form(self, V):
        a = super().form(V)
        key = V.dim()
        self.forms[key] = a
        return a

    def rebuild(self, key):
        a = self.forms[key]
        status = [x.dat.dat_version for x in a.coefficients()]
        try:
            if self.statuses[key] == status:
                return False
        except KeyError:
            pass
        self.statuses[key] = status
        return True


class VariablePkP0SchoeberlTransfer(VariableSchoeberlTransfer):
    def __init__(self, tdim, hierarchy):
        super().__init__(tdim, hierarchy)
        self.transfers = {}

    def standard_transfer(self, source, target, mode):
        if not (source.ufl_shape[0] == 3 and
                "CG1" in source.ufl_element().shortstr()):
            return super().standard_transfer(source, target, mode)

        if mode == "prolong":
            coarse = source
            fine = target
        elif mode == "restrict":
            fine = source
            coarse = target
        else:
            raise NotImplementedError
        (mh, level) = get_level(coarse.ufl_domain())
        if level not in self.transfers:
            self.transfers[level] = BubbleTransfer(
                coarse.function_space(), fine.function_space())
        if mode == "prolong":
            self.transfers[level].prolong(coarse, fine)
        elif mode == "restrict":
            self.transfers[level].restrict(fine, coarse)
        else:
            raise NotImplementedError
