"""S6 — can the four sheet sites be written on an interior facet?

Phase 1b needs the sheet spelling `4 pi G sigma v ds(id)` to work on a tagged
*interior* facet.  S1 settled that `dS(id)` finds the facet and that `avg`
is the restriction to use.  What it did not settle is whether the three other
integrands the sheet appears in survive the same treatment:

  * the residual/right-hand-side term, with `v` a CG test function,
  * the enclosed-mass form, whose test function lives on a `Real` space,
  * `check_net_mass`'s scale, a 0-form in `abs(sigma)`.

The `Real` one is the doubtful case: a globally constant function has no
natural notion of a restriction, and nothing in the spikes so far has put one
inside a `dS`.
"""
import gadopt  # noqa: F401  - before firedrake; see SPIKE-RESULTS.md S2(d)
import numpy as np
from firedrake import (
    Constant, Function, FunctionSpace, SpatialCoordinate, TestFunction,
    assemble, atan2, avg, cos, sqrt)
from firedrake import Measure, Mesh

from spike_mesh import RE, generate

mesh = Mesh(generate("spike_annulus.msh"))
X = SpatialCoordinate(mesh)
phi = atan2(X[1], X[0])
dS = Measure("dS", domain=mesh)
ds = Measure("ds", domain=mesh)
V = FunctionSpace(mesh, "CG", 2)
R = FunctionSpace(mesh, "R", 0)

exact = 2 * np.pi * RE
print(f"2 pi Re = {exact:.8f}")


def report(label, form):
    try:
        value = assemble(form)
    except Exception as exc:  # noqa: BLE001 - the point is to see the failure
        print(f"  {label:<44s} FAILED  {type(exc).__name__}: {exc}")
        return None
    if hasattr(value, "dat"):
        value = float(np.sum(value.dat.data_ro))
    print(f"  {label:<44s} {float(value):16.8f}")
    return float(value)


print("\nscalar measures")
report("1 * dS(2)", Constant(1.0) * dS(2))
report("avg(Constant(1)) * dS(2)", avg(Constant(1.0)) * dS(2))
report("1 * ds(4)  [exterior control, 4 pi Re]", Constant(1.0) * ds(4))

print("\nthe residual term, v in CG2 (assembled against v = 1 by 0-form trick)")
v = TestFunction(V)
vec = assemble(avg(Constant(1.0) * v) * dS(2))
print(f"  sum of avg(sigma*v)*dS(2) entries          "
      f"{float(np.sum(vec.dat.data_ro)):16.8f}")
sigma = cos(2 * phi)
vec = assemble(avg(sigma * v) * dS(2))
print(f"  sum with sigma = cos(2 phi) (want ~0)      "
      f"{float(np.sum(vec.dat.data_ro)):16.8e}")

print("\nthe enclosed-mass form, test function on a Real space")
mu = TestFunction(R)
report("avg(Constant(1) * mu) * dS(2)", avg(Constant(1.0) * mu) * dS(2))
report("avg(mu) * dS(2)", avg(mu) * dS(2))
report("Constant(1) * mu('+') * dS(2)", Constant(1.0) * mu("+") * dS(2))
report("Constant(0.3) * avg(mu) * dS(2)", Constant(0.3) * avg(mu) * dS(2))
report("avg(cos(2*phi) * mu) * dS(2)  (want ~0)", avg(cos(2 * phi) * mu) * dS(2))

print("\ncheck_net_mass's scale, a 0-form")
report("avg(abs(Constant(0.3))) * dS(2)", avg(abs(Constant(0.3))) * dS(2))
report("avg(abs(cos(2*phi))) * dS(2)", avg(abs(cos(2 * phi))) * dS(2))

print("\na DG0 (two-valued) sigma, to see what avg does with one")
chi = Function(FunctionSpace(mesh, "DG", 0)).interpolate(
    Constant(1.0) * (sqrt(X[0] ** 2 + X[1] ** 2) < RE))
report("avg(chi) * dS(2)  (mantle indicator, want pi Re)", avg(chi) * dS(2))

print("\nthe empty-measure failure this whole step is about")
report("1 * ds(2)   [Re is interior; want 0.0 + warning]", Constant(1.0) * ds(2))
report("avg(1) * dS(4)  [2Re is exterior; want 0.0]", avg(Constant(1.0)) * dS(4))
