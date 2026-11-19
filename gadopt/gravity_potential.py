r"""The gravitational potential of a solver, as one part.

`GravityPotential` holds everything about the potential `psi` that does not
depend on which solver drives it: the DtN boundary form, the volume measures
of a potential on a parent mesh coupled to a density on a submesh, the
residual row of the Poisson equation, the 2-D monopole datum and its
enclosed-mass bookkeeping, the strong `psi` conditions, and the eliminated
mode rows of the low-rank representation of the DtN map.

Two solvers compose it. `gadopt.gravity_solver.GravitySolver` solves the
Poisson equation alone, driven by a volume density and by the sheets of its
boundary conditions. The coupled self-gravitating GIA solver
(`gadopt.gia_solver`) drives it by the displacement of a reference density
and writes the sheets of its couplings into the residual itself. The part
knows neither solver: the datum takes the caller's sheets as
`(density, measure)` pairs, and the residual takes the displacement source
and the row scaling as arguments.

## Sign convention

`psi` is minus the Newtonian potential: `grad^2 psi = -4 pi G rho`, the
perturbation gravity is `+grad psi`, and a positive point mass has
`psi = +G M / r`. The boundary form, the monopole flux `-2 G M / R` and the
sheet terms are written in this convention.

## The residual row

    theta_psi [ int grad(psi) . grad(v) dx_g
                - Lambda int rho_0 u . grad(v) dx_m
                - 4 pi G int rho v dx_m
                + boundary_residual(psi, v, multipliers, monopole flux) ]

with `Lambda = 4 pi G` in the units of the potential. The displacement
source is the divergence form, so with `v = 1` it is identically zero and the
displaced reference density carries exactly zero net mass. The volume
density `rho` is a prescribed source with a net mass of its own, which the
2-D datum counts. Each source term is present only when its caller gives it.
"""

import numpy as np
from firedrake import (Constant, DirichletBC, Form, Function, FunctionSpace,
                       Measure, SpatialCoordinate, TestFunction, TrialFunction,
                       assemble, dot, grad, solve, sqrt)
from warnings import warn

from .dtn_coupled import CoupledLowRankDtN
from .dtn_form import DtNGravityForm
from .dtn_lowrank import apply_dirichlet_to_rows
from .utility import ensure_constant, interior_facet_integrand, scalar_value

__all__ = ["GravityPotential", "real_scalar_solver_parameters"]


#: For the one-by-one systems on a `Real` space that define the enclosed-mass
#: scalars. The operator is a nonzero number, so `preonly` + Jacobi is an exact
#: solve and not a preconditioning choice, and it must not inherit the
#: options of the solver that owns the potential. `snes_type: ksponly` is
#: named although `LinearVariationalSolver` defaults to it, because a tape
#: replay can rebuild the solve as a `NonlinearVariationalSolver` (Firedrake
#: PR 4638), whose default is Newton line search. For a control with zero net
#: mass the right-hand side is at round-off, and Newton then never meets its
#: relative tolerance.
real_scalar_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}


class GravityPotential:
    """The potential part of a solver: boundary form, residual, datum, low rank.

    Give exactly one of `Lambda` and `gravitational_constant`.

    Args:
      potential_space: the scalar space of `psi`, on the parent mesh.
      density_mesh: the mesh of the density that drives the potential, a
        `Submesh` of the parent or the parent itself.
      gravity_bcs: the boundary conditions of `psi` in `DtNGravityForm`'s
        dictionary form (`"dtn"`, `"psi"`, `"flux"`, `"sigma"`,
        `"interior_sigma"`). A sheet given here enters the residual through
        the boundary form and the 2-D datum through `enclosed_mass_forms`. A
        caller that writes its sheets into the residual itself passes them to
        the datum as `(density, measure)` pairs instead (`update_total_mass`).
      Lambda: the self-gravity number `4 pi G rho_bar L / g_bar` of a
        non-dimensional coupled system. The boundary form's gravitational
        constant is `Lambda / (4 pi)`, so that a sheet's `4 pi G sigma` and
        the displacement source agree.
      gravitational_constant: `G` of the potential's units. `Lambda` is then
        `4 pi G`.
      volume_density: a prescribed density on `density_mesh` (a Function, a
        UFL expression or a number), or `None` for no volume source. It
        enters the residual as `-4 pi G int rho v` and the 2-D datum with its
        net mass.
      source_quad_degree: the quadrature degree of the volume density's
        integrals, `None` for UFL's estimate.
      quad_degree: the boundary quadrature degree of the DtN form, `None`
        for its calibrated default.
      alpha: the Robin shift of the DtN form, `None` for its default.
      representation: `"multiplier"` (one `Real` unknown per DtN mode) or
        `"lowrank"` (the modes eliminated into a rank-n update).
    """

    def __init__(self, potential_space, density_mesh, gravity_bcs, *,
                 Lambda=None, gravitational_constant=None, volume_density=None,
                 source_quad_degree=None, quad_degree=None, alpha=None,
                 representation="lowrank"):
        if representation not in ("multiplier", "lowrank"):
            raise ValueError(
                f"dtn_representation must be 'multiplier' or 'lowrank', got "
                f"{representation!r}.")
        if (Lambda is None) == (gravitational_constant is None):
            raise ValueError(
                "GravityPotential needs exactly one of Lambda and "
                "gravitational_constant.")
        self.representation = representation
        self.potential_mesh = potential_space.mesh()
        self.density_mesh = density_mesh
        self.cross_mesh = density_mesh is not self.potential_mesh
        G = Lambda / (4.0 * np.pi) if Lambda is not None else gravitational_constant
        self.form = DtNGravityForm(
            potential_space, gravity_bcs, gravitational_constant=G,
            quad_degree=quad_degree, alpha=alpha)
        #: `4 pi G`, written as the boundary form writes it.
        self.four_pi_G = 4 * np.pi * self.form.G
        #: The coefficient of the displacement source. The caller's own
        #: `Lambda` object when it gives one, so that a control on it is the
        #: coefficient in the form.
        self.Lambda = Lambda if Lambda is not None else self.four_pi_G
        self.volume_density = (None if volume_density is None
                               else ensure_constant(volume_density))
        self.set_measures(source_quad_degree)
        self.set_monopole_datum()
        #: The low-rank operator of a coupled solver, built by
        #: `build_lowrank_operator`.
        self.dtn_operator = None

    @property
    def n_multipliers(self) -> int:
        """The number of `Real` unknowns this part adds to a mixed space."""
        return 0 if self.representation == "lowrank" else self.form.n_multipliers

    # -- Measures and geometry ------------------------------------------------

    def set_measures(self, source_quad_degree=None) -> None:
        """The volume measures of the potential and of the density.

        `dx_g` is the potential's, over the whole parent; `dx_m` the
        density's. Both are intersected with the other mesh when the two
        differ: Firedrake builds the cross-mesh entity maps from the
        intersection, and a sub-block that a preconditioner extracts from a
        mixed space spanning both meshes needs every integral to admit both
        domains. The intersection does not restrict the Laplacian to the
        submesh; `check_geometry` asserts that. `dx_parent` is the plain
        parent measure of the enclosed-mass system, whose arguments are
        `Real` fields of the parent alone. `dx_density` is `dx_m` at the
        quadrature degree of the volume density.
        """
        parent, sub = self.potential_mesh, self.density_mesh
        self.dx_parent = Measure("dx", domain=parent)
        if self.cross_mesh:
            self.dx_g = Measure("dx", domain=parent,
                                intersect_measures=(Measure("dx", domain=sub),))
            self.dx_m = Measure("dx", domain=sub,
                                intersect_measures=(Measure("dx", domain=parent),))
        else:
            self.dx_g = self.dx_parent
            self.dx_m = Measure("dx", domain=sub)
        self.dx_density = (self.dx_m if source_quad_degree is None
                           else self.dx_m(degree=source_quad_degree))

    def check_geometry(self) -> None:
        """Measure every DtN boundary and the cross-mesh measures.

        Each check is for a failure that gives a converged solve and no
        error: a DtN tag that matches nothing or part of a circle, an
        intersected measure that is empty, and a cross-mesh entity map that
        evaluates a parent field at the wrong points.

        The constructor does not call it: whether a DtN boundary must be a
        complete circle or sphere is the policy of the solver. The coupled
        GIA solver calls it. `GravitySolver` accepts a boundary that deviates
        from constant radius with the warning of `DtNGravityForm`. It
        assembles numbers that nothing on the tape reads, so a caller runs it
        under `stop_annotating`.
        """
        dim = self.potential_mesh.geometric_dimension
        X = SpatialCoordinate(self.potential_mesh)
        r = sqrt(dot(X, X))
        for bc_id, _ in self.form.dtn_boundaries:
            dss = self.form.ds(bc_id)
            extent = assemble(Constant(1.0) * dss)
            if extent <= 0.0:
                raise ValueError(
                    f"DtN boundary {bc_id}: the measure is empty. A tag of the "
                    "wrong facet kind gives zero and a warning.")
            radius = assemble(r * dss) / extent
            expected = 2 * np.pi * radius if dim == 2 else 4 * np.pi * radius ** 2
            if abs(extent - expected) > 1e-2 * expected:
                raise ValueError(
                    f"DtN boundary {bc_id} measures {extent:.6e} at mean radius "
                    f"{radius:.6g}, against {expected:.6e} for a complete "
                    "circle or sphere of that radius.")
        volume = assemble(Constant(1.0) * Measure("dx", domain=self.density_mesh))
        if not np.isclose(assemble(Constant(1.0) * self.dx_m), volume, rtol=1e-12):
            raise ValueError(
                "The intersected density measure does not assemble to the "
                "volume of the density's mesh; an empty intersection returns "
                "zero silently.")
        parent_volume = assemble(Constant(1.0) * self.dx_parent)
        if not np.isclose(assemble(Constant(1.0) * self.dx_g), parent_volume,
                          rtol=1e-12):
            raise ValueError(
                "The intersected parent measure does not cover the parent, so "
                "the Laplacian would miss the stand-off buffer.")
        if self.cross_mesh:
            probe = Function(FunctionSpace(self.potential_mesh, "CG", 1)).assign(1.0)
            crossed = assemble(probe * self.dx_m)
            if not np.isclose(crossed, volume, rtol=1e-10):
                raise ValueError(
                    f"A parent coefficient integrated over the density's mesh "
                    f"gives {crossed:.6e} against its volume {volume:.6e}.")

    # -- The residual row -------------------------------------------------------

    def volume_source(self, v, u=None, *, density=None):
        """The volume terms linear in the test function, right-hand-side convention.

        `Lambda int rho_0 u . grad(v) dx_m` when the displacement `u` is given
        (with `density` the reference density `rho_0`), plus
        `4 pi G int rho v dx_density` when the part has a volume density.

        Returns:
          A `Form`, empty when there is neither.
        """
        L = Form([])
        if u is not None:
            L += self.Lambda * density * dot(u, grad(v)) * self.dx_m
        if self.volume_density is not None:
            L += self.four_pi_G * self.volume_density * v * self.dx_density
        return L

    def operator(self, psi, v, multipliers=None):
        """The terms bilinear in the unknowns: Laplacian, Robin shift and DtN.

        Args:
          psi, v: the potential (a trial function or a split component) and
            its test function.
          multipliers: `None` for the Robin shift alone (the low-rank
            representation), or the `(trial, test)` pairs of the multiplier
            fields, in space order.
        """
        a = dot(grad(psi), grad(v)) * self.dx_g
        a += self.form.boundary_bilinear(psi, v, multipliers)
        return a

    def source(self, v, u=None, *, density=None):
        """Every term linear in the test function, right-hand-side convention.

        `volume_source`, the sheets and fluxes of the boundary form, and the
        2-D monopole datum. With `operator`, `operator - source` is the
        residual of a solver that assembles a matrix and a right-hand side.
        """
        L = self.volume_source(v, u, density=density)
        L += self.form.boundary_source(v, extra_flux=self.monopole_fluxes())
        return L

    def residual(self, psi, v, u=None, *, density=None, theta_psi=None,
                 multipliers=None):
        """The potential row and, on the multiplier path, the DtN constraint rows.

        The terms are the same as `operator - source`, in the order
        Laplacian, volume sources, boundary terms.

        Args:
          psi, v: the potential and its test function (split components).
          u: the displacement that moves the reference density, or `None`.
          density: the reference density `rho_0`, with `u`.
          theta_psi: the row scaling, which multiplies the constraint rows
            too, or `None` for no scaling.
          multipliers: `None` on the low-rank path; otherwise the
            `(trial, test)` pairs of the multiplier fields, in space order.

        Returns:
          The form.
        """
        if self.representation == "lowrank":
            # `None` writes the Robin shift alone; the modal feedback is added
            # by the solver (`build_lowrank_operator`, or the rank-n update of
            # `GravitySolver`).
            multipliers = None
        F = dot(grad(psi), grad(v)) * self.dx_g
        F -= self.volume_source(v, u, density=density)
        F += self.form.boundary_residual(psi, v, multipliers,
                                         extra_flux=self.monopole_fluxes())
        return F if theta_psi is None else theta_psi * F

    def dirichlet_bcs(self, potential_space):
        """The strong `psi` conditions of the boundary form, on `potential_space`."""
        return [DirichletBC(potential_space, value, bc_id)
                for bc_id, value in self.form.dirichlet_bcs]

    # -- The 2-D monopole datum --------------------------------------------------

    def set_monopole_datum(self) -> None:
        """Prepare the 2-D exterior monopole flux datum.

        A 2-D exterior DtN boundary does not determine the `m = 0` mode of
        the trace, because the exterior monopole solution is logarithmic. The
        datum supplies its flux, `-2 G M / R`, with `M` the enclosed mass:
        the net mass of the volume density, the sheets of the boundary form
        and the sheets the caller passes to `update_total_mass`. The
        displacement source carries no net mass. In 3-D nothing is built, not
        even the `Real` space and the volume assemble.

        The enclosed mass is held in `Real` Functions. A number in the form
        would cut the tape (see `update_total_mass`). An unknown with its own
        constraint row would cost a field and a Schur dimension, and would
        not exist on the low-rank path, which has no multipliers. A `Real`
        Function set by a recorded solve is differentiable and is an ordinary
        UFL coefficient, so one expression serves both representations and
        every adjoint block reads it with no further code.

        There are two scalars, because a prescribed `flux` condition carries
        a mass whose contribution to the datum has no `G` in it
        (`monopole_fluxes`).
        """
        #: The 2-D exterior DtN boundaries, which are the only ones whose
        #: monopole the trace does not determine. Empty in 3-D.
        self.monopole_boundaries = [
            bc_id for bc_id, _ in self.form.dtn_boundaries
            if self.potential_mesh.geometric_dimension == 2
            and self.form.boundary_geometry[bc_id][0] == "exterior"]
        self.real_space = None
        self.mesh_volume = None
        self.source_mass = None
        self.cavity_flux = None
        if not self.monopole_boundaries:
            return
        if len(self.monopole_boundaries) > 1:
            raise ValueError(
                f"2-D mesh with {len(self.monopole_boundaries)} exterior DtN "
                f"boundaries {self.monopole_boundaries}: the monopole datum "
                "needs the mass enclosed by the exterior boundary, which is "
                "not defined when there is more than one of them.")
        clash = sorted(
            {bc_id for bc_id, _ in self.form.flux_bcs}.intersection(
                self.monopole_boundaries),
            key=str)
        if clash:
            raise ValueError(
                f"Boundary {clash[0]}: 'flux' and a 2-D exterior 'dtn' both "
                "prescribe the m = 0 normal derivative there - the DtN map "
                "through the enclosed mass, the flux condition directly - so "
                "the two are a double specification of one quantity. Move the "
                "flux condition to an interior boundary or drop it.")
        self.real_space = FunctionSpace(self.potential_mesh, "R", 0)
        #: The value of the plain parent measure, never of `dx_g`, which in a
        #: cross-mesh system is intersected with the density's mesh.
        self.mesh_volume = assemble(Constant(1.0) * self.dx_parent)
        #: `int rho dx + sum_sheets int sigma`, refreshed by `update_total_mass`.
        self.source_mass = Function(self.real_space, name="source_mass")
        #: `sum_flux int (d psi)/(d n) ds` over the prescribed-flux boundaries,
        #: which is `4 pi G` times the mass they imply inside their cavities.
        self.cavity_flux = Function(self.real_space, name="cavity_flux")

    def monopole_fluxes(self) -> dict:
        """`{bc_id: flux}` for the boundary form's `extra_flux`.

        `-2 G M / R` on the 2-D exterior DtN boundary, with `M` the enclosed
        mass. By Gauss a cavity behind a prescribed flux `g` holds the mass
        `int g ds / (4 pi G)`, whose `G` cancels against the `2 G` in front,
        so that piece is written without `G`. It stays exactly
        differentiable when the gravitational constant is a control. The
        cavity term is left out when there is no flux condition, so that the
        form carries no dependency that can never change. Empty in 3-D.
        """
        out = {}
        for bc_id in self.monopole_boundaries:
            _, R = self.form.boundary_geometry[bc_id]
            flux = -2.0 * self.form.G * self.source_mass / R
            if self.form.flux_bcs:
                flux -= self.cavity_flux / (2.0 * np.pi * R)
            out[bc_id] = flux
        return out

    def _on_facet(self, value):
        """A `Real` test function of the parent on a facet of the density's mesh.

        In a cross-mesh system those facets are interior facets of the parent,
        where UFL requires a restriction of every argument; `avg` is exact for
        a `Real` field.
        """
        return interior_facet_integrand(value) if self.cross_mesh else value

    def enclosed_mass_forms(self, extra_masses=()):
        """`(identity, mass, flux)`: the one-by-one systems on the `Real` space.

        `mu` and `nu` are constant in space, so the left-hand side divided by
        `mesh_volume` (the value the same measure assembles to) is the
        identity, and `source_mass` holds a mass, not a mass per unit volume.
        `ensure_constant` is applied to the boundary values, because a Python
        `0.0` folds the integral away and leaves no linear form at all.

        Args:
          extra_masses: `(density, measure)` pairs of sheets that the caller
            writes into the residual itself. `density` is UFL on the
            density's mesh with no quantity of the parent in it (a parent
            quantity would need a restriction on the parent's interior
            facets, which this method gives the `Real` test function only),
            `measure` a facet measure of that mesh, already restricted to its
            tag (intersected with the parent's interior facets in a
            cross-mesh system).

        Returns:
          The identity bilinear form, the mass form (`None` when there is no
          source of mass) and the flux form (`None` without flux conditions).
        """
        mu = TestFunction(self.real_space)
        nu = TrialFunction(self.real_space)
        identity = (nu / self.mesh_volume) * mu * self.dx_parent
        mass = None
        if self.volume_density is not None:
            mass = self.volume_density * mu * self.dx_density
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            term = self.form.sheet_integral(ensure_constant(sigma) * mu, bc_id,
                                            integral_type)
            mass = term if mass is None else mass + term
        for density, measure in extra_masses:
            term = self._on_facet(mu) * density * measure
            mass = term if mass is None else mass + term
        flux = None
        for bc_id, value in self.form.flux_bcs:
            term = ensure_constant(value) * mu * self.form.ds(bc_id)
            flux = term if flux is None else flux + term
        return identity, mass, flux

    def update_total_mass(self, extra_masses=(), *, zero_initial_guess=True) -> None:
        """Refresh the enclosed-mass scalars the datum reads, as recorded solves.

        The caller runs this at the start of every solve, under annotation,
        so that the assembles and the solves are on the tape. The mass must
        be differentiable without becoming an unknown.

        Do not hold these values as numbers. A number cuts the dependency in
        a way that no Taylor test sees: the taped map is then
        self-consistent, so `taylor_test` converges at rate 2 while the
        gradient of the real forward map is wrong (measured at 325 % on both
        DtN representations). A finite difference of new solves sees it.

        Of five ways to put an assembled scalar into a form, two are
        differentiable. Measured on a problem whose right-hand side depends on
        the control only through the scalar, against a reference gradient of
        6.168:

        | spelling                                   | adjoint | tangent |
        | ------------------------------------------ | ------- | ------- |
        | `Constant.assign(assemble(...))`           | 0.0     | fails   |
        | `assemble(...).riesz_representation("l2")` | 0.0     | fails   |
        | `Function(R).assign(assemble(...))`        | 6.168   | fails   |
        | `assemble(...).riesz_representation("L2")` | 6.168   | 6.168   |
        | a small variational problem on `R`         | 6.168   | 6.168   |

        The third has an exact adjoint, but its tangent value is a bare
        `numpy.float64` that `derivative` refuses. The last two are the same
        mass-matrix solve; the explicit one is written here.

        The solve is made new on every call. A cached
        `LinearVariationalSolver` whose right-hand side has a facet integral
        into a `Real` space returns the right value on its first solve and
        wrong values after it in parallel (measured on a unit disc:
        `1.881574` then `0.0` alternately on two ranks). The KSP converges,
        so the result would be a wrong potential with no error.

        Args:
          extra_masses: `(density, measure)` pairs of the caller's sheets that
            the datum counts (`enclosed_mass_forms`).
          zero_initial_guess: `True` starts each solve from zero. A linear
            solve under `ksponly` adds an update to its initial guess, so a
            guess left from the last call puts the round-off of that call
            into the datum. From zero the datum is a function of the present
            sources alone, and a run restarted from a checkpoint reproduces
            it bit for bit; the reset is one recorded assign per scalar.
            `False` starts from the last value, with no reset on the tape;
            the result differs at round-off. `GravitySolver` uses `False`: it
            has no state to restart from, and its recorded solve stays one
            solve per scalar.
        """
        if not self.monopole_boundaries:
            return
        identity, mass, flux = self.enclosed_mass_forms(extra_masses)
        if mass is not None:
            if zero_initial_guess:
                self.source_mass.assign(0.0)
            solve(identity == mass, self.source_mass,
                  solver_parameters=real_scalar_solver_parameters)
        if flux is not None:
            if zero_initial_guess:
                self.cavity_flux.assign(0.0)
            solve(identity == flux, self.cavity_flux,
                  solver_parameters=real_scalar_solver_parameters)

    def check_net_mass(self, extra_scales=()) -> None:
        """Refuse a doubly anchored gauge; warn on a small net mass.

        Run after `update_total_mass`. Nothing on the tape reads the result:
        under annotation its assembles are recorded but reach no functional.
        The coupled GIA solver runs it under `stop_annotating`.

        The scale is the mass the sources move: `int |rho|` of the volume
        density plus `int |sigma|` of every sheet. The test is relative to
        that scale with no floor, so a uniform rescaling of the sources,
        under which the linear problem is invariant, does not change the
        result.

        With a strong `psi` condition the test function `v = 1` is not
        admissible, so the Robin monopole relation and the Dirichlet
        condition are two anchors of one additive constant, and the system is
        over-constrained (measured: 1.1555 relative error on an annulus, the
        same at two refinements). A net mass, or a cavity flux, with a strong
        condition is refused.

        A net mass that is nonzero but very small relative to the scale is
        the signature of leakage from a density or load interface that does
        not conform to cell edges. It warns. The band is narrow on purpose: a
        deliberate net mass is far above it, and a warning on every such
        problem would teach users to ignore it.

        Args:
          extra_scales: `(density, measure)` pairs of every sheet of the
            caller, counted by the datum or not.
        """
        if not self.monopole_boundaries:
            return
        terms = []
        if self.volume_density is not None:
            terms.append(abs(self.volume_density) * self.dx_density)
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            # `ensure_constant`: a Python `0.0` folds `abs(sigma) * ds` away,
            # and an empty form does not assemble to zero.
            terms.append(self.form.sheet_integral(
                abs(ensure_constant(sigma)), bc_id, integral_type))
        for density, measure in extra_scales:
            terms.append(abs(density) * measure)
        scale = 0.0
        for k, term in enumerate(terms):
            value = assemble(term)
            # The sum starts from the first assembled value, not from 0.0, so
            # that under annotation no addition of zero is recorded.
            scale = value if k == 0 else scale + value
        mass = abs(scalar_value(self.source_mass))
        relative = mass / scale if scale > 0.0 else float(mass > 0.0)
        # A prescribed flux is data, not a discretised field, so it cannot
        # leak; it can still anchor the monopole.
        anchored = relative > 1e-8 or scalar_value(self.cavity_flux) != 0.0
        if anchored and self.form.dirichlet_bcs:
            raise ValueError(
                f"Net enclosed mass is nonzero (source mass "
                f"{scalar_value(self.source_mass):.3e}, cavity flux "
                f"{scalar_value(self.cavity_flux):.3e}) on a 2-D mesh that has "
                "both an exterior DtN boundary and a strong 'psi' condition on "
                f"boundary {self.form.dirichlet_bcs[0][0]}. The 2-D monopole "
                "datum fixes the potential's additive constant through the "
                "Robin term, and the Dirichlet condition fixes it again: the "
                "two anchors disagree and the system is over-constrained. Drop "
                "one of them, or use sources with zero net mass.")
        if 1e-8 < relative < 1e-4:
            what = "Net mass" if self.volume_density is not None else "Net sheet mass"
            warn(f"{what} {scalar_value(self.source_mass):.3e} is "
                 f"{relative:.1e} of the sources' own scale: nonzero, but too "
                 "small to look deliberate. This is the signature of leakage "
                 "from a density or load interface that does not conform to "
                 "cell edges; align the mesh with it. If the mass is meant to "
                 "be there, ignore this.")

    # -- The low-rank representation ----------------------------------------------

    def build_mode_rows(self, potential_space):
        """The eliminated mode rows of every DtN boundary, with the strong rows removed.

        Builds `DtNGravityForm.build_mode_rows`, checks that each boundary's
        rows follow the descriptor's mode order (the trace coefficients are
        read by position), and zeroes the entries at the degrees of freedom
        of a strong `psi` condition: those rows are eliminated from the
        assembled operator, so the rank-n update must not write into them
        either.

        Args:
          potential_space: the space of the potential's degrees of freedom,
            the potential space itself or the potential's subspace of a mixed
            space.

        Returns:
          The `BoundaryModeRows` list, which is also `form.mode_rows`. The
          constrained degrees of freedom are stored as `constrained_dofs`.
        """
        mode_rows = self.form.build_mode_rows()
        for (bc_id, dtn), rows in zip(self.form.dtn_boundaries, mode_rows, strict=True):
            side, radius = self.form.boundary_geometry[bc_id]
            expected = [mode.key for mode in dtn.mode_metadata(side, radius)]
            if list(rows.keys) != expected:
                raise RuntimeError(
                    f"boundary {bc_id}: the mode rows are ordered {list(rows.keys)} "
                    f"but the descriptor's order is {expected}.")
        constrained = set()
        for bc in self.dirichlet_bcs(potential_space):
            constrained.update(np.asarray(bc.nodes, dtype=np.int64).tolist())
        apply_dirichlet_to_rows(mode_rows, constrained)
        self.constrained_dofs = constrained
        return mode_rows

    def build_lowrank_operator(self, mixed_space, potential_index, theta):
        """`B = theta_psi C^T W C` on the potential rows of a coupled mixed space.

        Args:
          mixed_space: the mixed space of the coupled solution.
          potential_index: the index of the potential field in it.
          theta: a callable returning `theta_psi` as a number, read at every
            application so that a control on `Lambda` or `B_mu` reaches it.

        Returns:
          The `CoupledLowRankDtN`, also stored as `dtn_operator`.
        """
        mode_rows = self.build_mode_rows(mixed_space.sub(potential_index))
        self.dtn_operator = CoupledLowRankDtN(
            mixed_space, potential_index, mode_rows, theta,
            self.potential_mesh.comm)
        return self.dtn_operator
