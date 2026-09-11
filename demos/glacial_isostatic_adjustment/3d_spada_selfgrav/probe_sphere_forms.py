"""Fluid-limit probe on the actual mantle submesh with a hand-written form, so the
element (P2 penalty vs P2-P1 mixed), the quadrature degree and the density
layering can be varied. No self-gravity, un = 0 at the CMB, all layers fluid
(mu -> eps*mu, K = ratio*mu held), so the exact answer is Airy with the top
density: U_n = -sigma_n / rho_top, interfaces flat.

    python probe_sphere_forms.py [--mixed] [--quad 6] [--uniform-rho] [--kmu 100] [--no-curve]
"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gadopt  # noqa: F401,E402
import numpy as np  # noqa: E402
from firedrake import *  # noqa: E402,F403
import b1_elastic as b1  # noqa: E402
import generate_selfgrav_sphere as gen  # noqa: E402
import taboo_synthesis as taboo  # noqa: E402
import reference_state as refstate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mesh", default=os.path.join(HERE, "b2_coarse_ar7.msh"))
p.add_argument("--eps", type=float, default=1e-6)
p.add_argument("--kmu", type=float, default=100.0)
p.add_argument("--mixed", action="store_true")
p.add_argument("--quad", type=int, default=6)
p.add_argument("--uniform-rho", action="store_true")
p.add_argument("--rhos", type=str, default=None, help="comma list of the 4 layer densities (lith,UM,TZ,LM), kg/m3")
p.add_argument("--no-curve", action="store_true")
p.add_argument("--sheet", action="store_true", help="explicit surface buoyancy sheet instead of the volume form (uniform rho only)")
p.add_argument("--sheet-layered", action="store_true", help="explicit sheets at the surface and at every density interface (interior facets picked by radius)")
p.add_argument("--nmax", type=int, default=10)
p.add_argument("--nproj", type=int, default=12)
p.add_argument("--quad-degree", type=int, default=60)
p.add_argument("--degree", type=int, default=2)
p.add_argument("--pspace", type=str, default="CG1", help="pressure space for --mixed: CG1 (Taylor-Hood) or DG0")
args = p.parse_args()

parent, sub, _, _ = b1.build_meshes("coarse", path=args.mesh, curve_enabled=not args.no_curve)
mesh = sub
X = SpatialCoordinate(mesh)
r = sqrt(dot(X, X))
er = X / r
nmax = args.nmax
sigma_n = b1.cap_sigma_hat(nmax)
sigma_sub = b1.load_field(sub, nmax, sigma_n)
if args.uniform_rho:
    L = [list(x) for x in b1.LAYERS]
    for x in L:
        x[2] = L[1][2]      # 3438 everywhere
    b1.LAYERS = [tuple(x) for x in L]
if args.rhos:
    vals = [float(x) for x in args.rhos.split(",")]
    L = [list(x) for x in b1.LAYERS]
    for x, v in zip(L, vals):
        x[2] = v / b1.RHO_BAR
    b1.LAYERS = [tuple(x) for x in L]
rho = b1.layered(sub, 2, "density")
mu_e = b1.layered(sub, 3, "shear_modulus")
mu = args.eps * mu_e
K = args.kmu * mu_e
g = refstate.gravity_exact_ufl(r)
B = b1.B_MU
dx_q = dx(degree=args.quad)


def eps_dev(u):
    e = sym(grad(u))
    return e - tr(e) / 3 * Identity(3)


def grav_form(u, v):
    if args.sheet_layered:
        F = B * Constant(b1.LAYERS[0][2]) * g * dot(u, er) * dot(v, er) * ds(gen.SURF_RE, degree=args.quad)
        tol = 1e-4
        for k in range(len(b1.LAYERS) - 1):
            r_i = b1.LAYERS[k][1]
            drho = Constant(b1.LAYERS[k + 1][2] - b1.LAYERS[k][2])
            ind = conditional(abs(r - Constant(r_i)) < tol, 1.0, 0.0)
            F += B * drho * avg(g * ind) * avg(dot(u, er)) * avg(dot(v, er)) * dS(degree=args.quad)
        return F
    if args.sheet:
        return B * rho * g * dot(u, er) * dot(v, er) * ds(gen.SURF_RE, degree=args.quad)
    gphi = g * er
    return 0.5 * B * rho * (dot(grad(dot(u, gphi)), v) + dot(u, grad(dot(v, gphi)))
                            - div(u) * dot(gphi, v) - dot(gphi, u) * div(v)) * dx_q


V = VectorFunctionSpace(mesh, "CG", args.degree)
n = FacetNormal(mesh)
load = B * sigma_sub
# un = 0 at the CMB via Nitsche is what G-ADOPT does; here use a Lagrange-free penalty-free approach:
# strong Dirichlet is not possible for u.n on a curved surface, so use a Nitsche penalty on u.n
h = FacetArea(mesh) / avg(CellVolume(mesh)) if False else Constant(500e3 / b1.D_M / 1e3)  # ~ mesh size (nondim)
if not args.mixed:
    u = TrialFunction(V); v = TestFunction(V)
    a = inner(2 * mu * eps_dev(u), eps_dev(v)) * dx_q + K * div(u) * div(v) * dx_q + grav_form(u, v)
    # Nitsche for u.n = 0 at Rc (symmetric, penalty on the normal component)
    sig_n = 20.0 * (mu + K) / h
    stress = lambda w: 2 * mu * eps_dev(w) + K * div(w) * Identity(3)  # noqa: E731
    a += (sig_n * dot(u, n) * dot(v, n) - dot(v, n) * dot(n, dot(stress(u), n))
          - dot(u, n) * dot(n, dot(stress(v), n))) * ds(gen.SURF_RC, degree=args.quad)
    Lf = -load * dot(v, n) * ds(gen.SURF_RE, degree=args.quad)
    w = Function(V)
    solve(a == Lf, w, solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                                         "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 100,
                                         "mat_mumps_icntl_24": 1})
    uh = w
else:
    Q = FunctionSpace(mesh, "CG", args.degree - 1) if args.pspace == "CG1" else FunctionSpace(mesh, "DG", 0)
    W = V * Q
    (u, pr) = TrialFunctions(W); (v, q) = TestFunctions(W)
    a = inner(2 * mu * eps_dev(u), eps_dev(v)) * dx_q - pr * div(v) * dx_q - q * div(u) * dx_q \
        - (1.0 / K) * pr * q * dx_q + grav_form(u, v)
    sig_n = 20.0 * mu_e / h
    a += (sig_n * dot(u, n) * dot(v, n) - dot(v, n) * dot(n, dot(2 * mu * eps_dev(u) - pr * Identity(3), n))
          - dot(u, n) * dot(n, dot(2 * mu * eps_dev(v) - q * Identity(3), n))) * ds(gen.SURF_RC, degree=args.quad)
    Lf = -load * dot(v, n) * ds(gen.SURF_RE, degree=args.quad)
    w = Function(W)
    solve(a == Lf, w, solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                                         "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 200,
                                         "mat_mumps_icntl_24": 1})
    uh = w.sub(0)

ds_sub = ds(gen.SURF_RE, domain=sub)
U_n = b1.project_surface(dot(uh, er), sub, args.nproj, ds_sub, interior=False,
                         quad_degree=args.quad_degree) * b1.D_M
sig_dim = taboo.cap_load(nmax)
rho_top = 3438.0 if args.uniform_rho else (float(args.rhos.split(",")[0]) if args.rhos else 3037.0)
print(f"probe_sphere_forms mixed={args.mixed} pspace={args.pspace} quad={args.quad} uniform_rho={args.uniform_rho} kmu={args.kmu} "
      f"eps={args.eps} sheet={args.sheet} sheet_layered={args.sheet_layered} no_curve={args.no_curve} deg={args.degree}")
print("  n   U_n(m)   U_Airy   ratio")
for nn in range(2, nmax + 1):
    U_airy = -sig_dim[nn] / rho_top
    print(f"  {nn:2d} {U_n[nn]:10.5f} {U_airy:10.5f} {U_n[nn]/U_airy:8.4f}")
print(f"  U0 {U_n[0]:.3e} U1 {U_n[1]:.3e}")
print("RESULT done")
