"""Fluid-limit probe of the mechanics alone: no self-gravity, rigid CMB (un = 0),
segregated InternalVariableSolver, direct LU. Same mesh, same P2 penalty
formulation and prestress form as B5.  Compare with the propagator run without
self-gravity? -- no such propagator; compare instead with plain Airy with the
plate: the exact fluid limit without self-gravity is U_f = -sigma/(rho_top g)
modified only by the lithosphere (a few percent).  We report U_n / U_Airy(3438).

    python probe_fluid_mechonly.py --s 1e-7 [--no-curve] [--ratio 100]
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gadopt  # noqa: F401,E402
from gadopt import InternalVariableSolver, rigid_body_modes, CompressibleInternalVariableApproximation  # noqa: E402
import numpy as np  # noqa: E402
from firedrake import *  # noqa: E402,F403
import b1_elastic as b1  # noqa: E402
import generate_selfgrav_sphere as gen  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402
import reference_state as refstate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", default=os.path.join(HERE, "b2_coarse_ar7.msh"))
    p.add_argument("--s", type=float, default=1e-7)
    p.add_argument("--ratio", type=float, default=100.0)
    p.add_argument("--nmax", type=int, default=10)
    p.add_argument("--nproj", type=int, default=20)
    p.add_argument("--quad-degree", type=int, default=60)
    p.add_argument("--no-curve", action="store_true")
    p.add_argument("--rot-penalty", type=float, default=0.0)
    args = p.parse_args()

    L = [list(r) for r in b1.LAYERS]
    KYR = 3.15576e10; MU_BAR, ETA_BAR = 1.0e11, 1.0e21
    facs = []
    for r in L[1:]:
        tau_kyr = (r[4] * ETA_BAR) / (r[3] * MU_BAR) / KYR
        f = args.s / (args.s + 1.0 / tau_kyr)
        facs.append(f); r[3] *= f
    b1.LAYERS = [tuple(r) for r in L]
    print(f"mechonly probe: s={args.s} ratio={args.ratio} facs={facs} no_curve={args.no_curve}", flush=True)

    parent, sub, _, _ = b1.build_meshes("coarse", path=args.mesh, curve_enabled=not args.no_curve)
    nmax = args.nmax
    sigma_n = b1.cap_sigma_hat(nmax)
    sigma_sub = b1.load_field(sub, nmax, sigma_n)
    rho = b1.layered(sub, 2, "density")
    mu = b1.layered(sub, 3, "shear_modulus")
    eta = b1.layered(sub, 4, "viscosity")
    Xm = SpatialCoordinate(sub)
    rm = sqrt(dot(Xm, Xm))
    expr = Constant(args.ratio / facs[2])
    expr = conditional(rm > Constant(b1.LAYERS[2][1]), Constant(args.ratio / facs[1]), expr)
    expr = conditional(rm > Constant(b1.LAYERS[1][1]), Constant(args.ratio / facs[0]), expr)
    expr = conditional(rm > Constant(b1.LAYERS[0][1]), Constant(args.ratio), expr)
    ratio_field = Function(FunctionSpace(sub, "DG", 0)).interpolate(expr)
    approx = CompressibleInternalVariableApproximation(
        bulk_modulus=mu, density=rho, shear_modulus=[mu], viscosity=[eta],
        bulk_shear_ratio=ratio_field, g=refstate.gravity_exact_ufl(rm), B_mu=b1.B_MU)
    V = VectorFunctionSpace(sub, "CG", 2)
    S = TensorFunctionSpace(sub, "DG", 1)
    u = Function(V, name="displacement"); m = Function(S, name="m")
    params = {"mat_type": "aij", "snes_type": "ksponly", "ksp_type": "preonly",
              "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 100, "mat_mumps_icntl_24": 1}
    rbm = rigid_body_modes(V, rotational=True)
    solver = InternalVariableSolver(
        u, approx, dt=b1.DT_ELASTIC, internal_variables=[m],
        bcs={gen.SURF_RE: {"normal_stress": b1.B_MU * sigma_sub}, gen.SURF_RC: {"un": 0.0}},
        solver_parameters=params, nullspace=rbm, transpose_nullspace=rigid_body_modes(V, rotational=True))
    print(f"  dofs {V.dim()}", flush=True)
    t0 = time.perf_counter(); solver.solve(); print(f"  solved in {time.perf_counter()-t0:.1f}s", flush=True)
    rbm.orthogonalize(u)
    ds_sub = ds(gen.SURF_RE, domain=sub)
    U_n = b1.project_surface(dot(u, Xm / rm), sub, args.nproj, ds_sub, interior=False,
                             quad_degree=args.quad_degree) * b1.D_M
    sig_dim = taboo.cap_load(nmax)
    nn = np.arange(2, nmax + 1)
    print("  n   U_n(m)   U_Airy(3438)   ratio")
    for n in nn:
        # Airy with the compensating density 3438 (lithosphere + its base move together)
        U_airy = -sig_dim[n] / 3438.0
        print(f"  {n:2d} {U_n[n]:10.5f} {U_airy:10.5f} {U_n[n]/U_airy:8.4f}")
    print("RESULT probe_fluid_mechonly done")


if __name__ == "__main__":
    main()
