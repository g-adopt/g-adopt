"""Modal Dirichlet-to-Neumann solver for the gravitational Poisson equation.

Solves -nabla^2 psi = 4 pi gamma rho with the infinite-domain boundary
condition imposed exactly (up to mode truncation M) on circular boundaries
of the finite mesh, via R-space Lagrange multipliers.

Exterior boundary (sources inside, radius R_out): each azimuthal mode m of
the boundary trace extends into the exterior as r^(-m), giving
dpsi/dr + (m/R_out) psi = 0. Interior boundary (source-free core inside,
radius R_in): the interior solution is spanned by r^(+m), giving
dpsi/dr - (m/R_in) psi = 0; both enter the weak form with a positive sign.
The m = 0 (monopole) exterior mode is handled under the zero-total-mass
assumption (integral rho dx = 0), for which the boundary mean of psi
vanishes; the Robin-shifted formulation (see the comment in __init__)
imposes exactly that, and removes the constant nullspace for free.

The map needs only the trace of psi, never the density: the Fourier
coefficients of the trace are unknowns in an R-space (one global DOF per
coefficient), defined by scalar constraint rows and fed back as boundary
flux. Everything is a plain UFL form, so the solver works inside larger
coupled systems and under pyadjoint.

The density may live on the same mesh as psi or on a Submesh of it (the
cross-mesh coupling then goes through intersect_measures and a dummy
field, as in gravity_poisson_test.py).
"""
import numpy as np

from firedrake import *  # noqa: F401,F403


class GravityPoissonSolver:
    """Gravitational Poisson solver with modal DtN boundary treatment.

    Parameters
    ----------
    mesh:
        Mesh carrying the potential.
    rho:
        Density Function, on `mesh` or on a Submesh of it.
    M:
        Mode truncation: modes 1..M are treated exactly on each DtN
        boundary; higher modes see homogeneous Neumann. The neglected
        content is O((r_source/R)^(M+1)).
    outer:
        (boundary_id, radius) of the exterior DtN boundary (required).
    inner:
        Optional (boundary_id, radius) of an interior DtN boundary
        (source-free core inside it).
    gamma:
        Gravitational constant.
    degree:
        Polynomial degree of the CG space for psi.
    quad_degree:
        Quadrature degree for the boundary integrals involving
        cos(m*phi); UFL's automatic estimate is unreliable for these.
        Defaults to 2*(M + degree).
    """

    def __init__(self, mesh, rho=None, M=None, *, outer, inner=None,
                 gamma=1.0, degree=2, quad_degree=None,
                 source_expr=None, source_id=None, source_degree=None,
                 solver_parameters=None):
        self.mesh = mesh
        self.rho = rho
        self.M = M
        self.gamma = gamma

        # Two source paths, sharing every DtN term unchanged:
        #  - discrete: `rho` is a Function (on `mesh` or a Submesh of it),
        #    entering as -4 pi gamma rho v dx (cross-mesh if on a Submesh);
        #  - exact: `source_expr` is a UFL density expression integrated on
        #    the full mesh (optionally only over subdomain `source_id`),
        #    with quadrature `source_degree`. No density field, no submesh
        #    -- used to isolate the discrete-density error.
        exact_source = source_expr is not None
        if exact_source == (rho is not None):
            raise ValueError("provide exactly one of rho or source_expr")

        cross_mesh = False
        rho_mesh = None
        if not exact_source:
            rho_mesh = rho.function_space().mesh()
            cross_mesh = rho_mesh is not mesh

        self.V = FunctionSpace(mesh, "CG", degree)
        spaces = [self.V]
        if cross_mesh:
            spaces.append(FunctionSpace(rho_mesh, "DG", 0))

        # R-block layout: [outer 2M | inner 2M + mean (optional)].
        # One scalar R space per coefficient: Arguments on vector-valued R
        # spaces are not supported by Firedrake (ufl_expr.py raises
        # NotImplementedError), so a packed VectorFunctionSpace R-block
        # cannot be an unknown -- only separate scalar R fields can.
        self.boundaries = [("outer", *outer)]
        n_R = 2 * M
        if inner is not None:
            self.boundaries.append(("inner", *inner))
            n_R += 2 * M + 1
        R_scalar = FunctionSpace(mesh, "R", 0)
        spaces.extend([R_scalar] * n_R)
        self._iR0 = len(spaces) - n_R

        W = MixedFunctionSpace(spaces)
        self.w = Function(W)
        trials = split(self.w)
        tests = TestFunctions(W)
        psi, v = trials[0], tests[0]
        c = trials[self._iR0:]
        mu = tests[self._iR0:]

        qd = quad_degree if quad_degree is not None else 2 * (M + degree)

        # NB: dot/* rather than inner() -- the `inner` kwarg shadows ufl.inner here.
        if exact_source:
            sd = source_degree if source_degree is not None else qd
            src_meas = (dx(source_id, domain=mesh, degree=sd)
                        if source_id is not None
                        else dx(domain=mesh, degree=sd))
            F = dot(grad(psi), grad(v)) * dx(domain=mesh)
            F -= 4 * pi * gamma * source_expr * v * src_meas
        else:
            if cross_mesh:
                dx_full = Measure("dx", domain=mesh,
                                  intersect_measures=(Measure("dx", domain=rho_mesh),))
                dx_rho = Measure("dx", domain=rho_mesh,
                                 intersect_measures=(Measure("dx", domain=mesh),))
            else:
                dx_full = dx_rho = dx(domain=mesh)
            F = dot(grad(psi), grad(v)) * dx_full
            F -= 4 * pi * gamma * rho * v * dx_rho
            if cross_mesh:
                lam, mu_lam = trials[1], tests[1]
                F += lam * mu_lam * Measure("dx", domain=rho_mesh)

        X = SpatialCoordinate(mesh)
        phi = atan2(X[1], X[0])
        self.quad_degree = qd

        # Robin-shifted modal DtN. The naive form puts all boundary
        # stiffness in the psi<->R coupling, leaving the psi-psi block a
        # singular pure-Neumann Laplacian -- fatal for the Schur fieldsplit
        # below. Split the modal flux instead as
        #     dpsi/dn ~ -(alpha/R) psi + sum_m ((alpha - m)/R) c_m e_m,
        # identical for modes 1..M, but the pointwise Robin term lands in
        # the psi-psi block (SPD, no nullspace). Side effects, both benign:
        # untreated modes > M see Robin(alpha/R) instead of Neumann, and
        # the exterior m = 0 sees Robin -- which for zero-total-mass
        # sources is exact (the boundary mean must vanish), so no monopole
        # multiplier is needed. On an interior boundary the exact m = 0
        # condition is Neumann, so one mean multiplier undoes the shift.
        alpha = 1.0
        off = 0
        for name, bid, R in self.boundaries:
            dss = ds(bid, domain=mesh, degree=qd)
            F += (alpha / R) * psi * v * dss
            for m in range(1, M + 1):
                cm, sm = c[off + 2*m - 2], c[off + 2*m - 1]
                muc, mus = mu[off + 2*m - 2], mu[off + 2*m - 1]
                # Constraint rows: mu* are globally constant, so each is one
                # scalar equation; since integral(ds) = 2 pi R, the -cm/2
                # enforces cm = (1/(pi R)) integral(psi cos(m phi) ds) -- the
                # Fourier coefficient of the trace.
                F += (psi * cos(m * phi) - cm / 2) * muc * dss
                F += (psi * sin(m * phi) - sm / 2) * mus * dss
                # Modal correction to the shifted flux (same positive sign
                # for exterior and interior maps).
                F += ((m - alpha) / R) * (cm * cos(m * phi)
                                          + sm * sin(m * phi)) * v * dss
            off += 2 * M
            if name == "inner":
                # Undo the Robin shift on the interior monopole: exact
                # interior m = 0 is homogeneous Neumann, so subtract
                # (alpha/R) times the trace mean.
                c0, mu0 = c[off], mu[off]
                F += (psi - c0) * mu0 * dss
                F -= (alpha / R) * c0 * v * dss
                off += 1

        self.F = F

        # Matrices with R-space blocks cannot be assembled monolithically
        # (aij), so direct LU on the full system is unavailable. Instead:
        # full Schur complement eliminating onto the R fields, with the
        # assembled psi block factorised by MUMPS and the (dense, tiny)
        # R-R Schur complement handled by GMRES. This mirrors the defaults
        # Firedrake generates for Real blocks (solving_utils.set_defaults),
        # spelled out so it is visible and tunable.
        fields = ",".join(map(str, range(self._iR0)))
        reals = ",".join(map(str, range(self._iR0, self._iR0 + n_R)))
        self.solver_parameters = solver_parameters or {
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "fgmres",
            "ksp_rtol": 1e-11,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": fields,
            "pc_fieldsplit_1_fields": reals,
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            },
            "fieldsplit_1": {
                "ksp_type": "gmres",
                "pc_type": "none",
            },
        }

    def solve(self):
        solve(self.F == 0, self.w, solver_parameters=self.solver_parameters)
        return self.w.subfunctions[0]

    def coefficients(self):
        """Solved trace Fourier coefficients per DtN boundary.

        Returns {name: {"cos": array(M), "sin": array(M)}} plus the mean
        multiplier under "mean".
        """
        data = np.array([float(f) for f in self.w.subfunctions[self._iR0:]])
        out = {}
        off = 0
        for name, _, _ in self.boundaries:
            block = data[off:off + 2 * self.M]
            out[name] = {"cos": block[0::2].copy(), "sin": block[1::2].copy()}
            off += 2 * self.M
            if name == "inner":
                out[name]["mean"] = float(data[off])
                off += 1
        return out

    def check_boundary_quadrature(self, rtol=1e-8):
        """Verify the boundary quadrature resolves the modal basis.

        Checks integral(cos(m phi)^2 ds) = pi R for m = 1..M on each DtN
        boundary; returns the worst relative deviation.
        """
        X = SpatialCoordinate(self.mesh)
        phi = atan2(X[1], X[0])
        worst = 0.0
        for _, bid, R in self.boundaries:
            dss = ds(bid, domain=self.mesh, degree=self.quad_degree)
            for m in range(1, self.M + 1):
                val = assemble(cos(m * phi)**2 * dss)
                worst = max(worst, abs(val - np.pi * R) / (np.pi * R))
        if worst > rtol:
            raise ValueError(
                f"boundary quadrature/mesh does not resolve mode {self.M}: "
                f"worst relative deviation {worst:.3e} > {rtol:.1e}")
        return worst
