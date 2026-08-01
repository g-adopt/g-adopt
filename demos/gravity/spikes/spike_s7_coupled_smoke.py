"""S7 — does `SelfGravitatingGIASolver` construct and solve at all?

Scratch, written while implementing Phase 4.  Not a gate; the gates live in
`demos/gravity/selfgrav_gia_annulus.py` and `tests/unit/`.  Kept because the
sequence of errors it shook out is the record of what the base classes do on a
two-mesh mixed space.

    PYTHONPATH=<worktree> python3 demos/gravity/spikes/spike_s7_coupled_smoke.py
"""
import os
import sys

import gadopt  # noqa: F401  BEFORE firedrake: PETSc imports gadopt lazily when
# it reads `pc_python_type`, long after a UFL multifunction has run, and
# Irksome's import-order guard then fires as a bare `PETSc.Error: 101`.
from gadopt import *  # noqa: E402
from gadopt.gia_gravity import OMEGA_SQ_EARTH  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_selfgrav_annulus as gen  # noqa: E402
from validate_selfgrav_annulus import curve_mesh  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "spike_s7_annulus.msh")

if COMM_WORLD.rank == 0:
    gen.generate(MESH, dr_mantle=0.15, n_azimuthal=32)
COMM_WORLD.barrier()

parent = curve_mesh(Mesh(MESH))
parent.cartesian = False
sub = curve_mesh(Submesh(parent, 2, gen.CELL_MANTLE))
sub.cartesian = False
log(f"parent cells {parent.num_cells()}, mantle cells {sub.num_cells()}")

B_mu, Lambda = 1.2769, 1.1116
X = SpatialCoordinate(parent)
phi = atan2(X[1], X[0])
sigma = 1.0e-3 * cos(2 * phi)

gravity_bcs = {
    gen.CURVE_OUTER: {"dtn": CylindricalDtN(5)},
    gen.CURVE_INNER: {"dtn": CylindricalDtN(5)},
    gen.CURVE_RE: {"interior_sigma": sigma},
}

Z, layout = self_gravitating_gia_space(
    sub, parent, gravity_bcs=gravity_bcs, rotation=True,
    self_gravity_number=Lambda)
log(f"fields {len(Z)}  layout {layout.multipliers=} {layout.rotation=}")
log(f"dim {Z.dim()}")

z = Function(Z)
approximation = CompressibleInternalVariableApproximation(
    bulk_modulus=1.0, density=1.0, shear_modulus=1.0, viscosity=1.0,
    g=1.0, B_mu=B_mu, self_gravity_number=Lambda)

Xm = SpatialCoordinate(sub)
dx_m = Measure("dx", domain=sub, intersect_measures=(Measure("dx", domain=parent),))
C = assemble(1.0 * dot(Xm, Xm) * dx_m)
log(f"polar second moment C = {C}")

phi_m = atan2(Xm[1], Xm[0])
bcs = {
    gen.CURVE_RC: {"un": 0.0},
    gen.CURVE_RE: {"normal_stress": B_mu * 1.0e-3 * cos(2 * phi_m)},
}

solver = SelfGravitatingGIASolver(
    z, approximation, layout=layout, dt=1.0, bcs=bcs,
    rotation_moments={"C": C}, Omega_sq=OMEGA_SQ_EARTH)
log(f"theta_psi = {float(solver.theta_psi)}")
log(f"theta_rot(m3) = {float(solver._theta_rot(2))}")

solver.solve()
log(f"||u|| = {norm(solver.displacement)}")
log(f"||psi|| = {norm(solver.potential)}")
log(f"rotation {solver.rotation_values()}")
log(f"inertia {solver.inertia_perturbation()}")
log(f"coefficients {solver.coefficients()}")
log(f"source mass form -> {assemble(solver.source_mass_form()):.3e}")
