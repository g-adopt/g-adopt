r"""The gravitational potential of a coupled solver, as one part.

`GravityPotential` holds everything about the potential `psi` that does not
depend on what drives it: the DtN boundary form, the volume measures of a
potential on a parent mesh coupled to a density on a submesh, the residual
row of the Poisson equation in divergence form, the 2-D monopole datum and
its enclosed-mass bookkeeping, the strong `psi` conditions, and the low-rank
representation of the DtN map.

The coupled self-gravitating GIA solver (`gadopt.gia_solver`) composes it
with the mechanics and the couplings. The interface is written so that
`GravitySolver` can use the same object later: the datum takes the mass
forms and scales of the sheets as arguments instead of knowing the couplings,
and the residual takes the source density and the row scaling as arguments.

## Sign convention

`psi` is minus the Newtonian potential: `grad^2 psi = -4 pi G rho`, the
perturbation gravity is `+grad psi`, and a positive point mass has
`psi = +G M / r`. The boundary form, the monopole flux `-2 G M / R` and the
sheet terms are written in this convention.

## The residual row

    theta_psi [ int grad(psi) . grad(v) dx_g
                - Lambda int rho_0 u . grad(v) dx_m
                + boundary_residual(psi, v, multipliers, monopole flux) ]

The source is the divergence form, so with `v = 1` the row is identically
zero and the discrete perturbation carries exactly zero net mass. Every
interface mass of the mantle is included through the integration by parts.
Sheets that are not part of the mantle (a core, an ocean, a load) are written
into this row by their own couplings, not here.
"""

import numpy as np
from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                       Measure, SpatialCoordinate, TestFunction, TrialFunction,
                       assemble, dot, grad, solve, sqrt)
from warnings import warn

from pyadjoint.tape import stop_annotating

from .dtn_coupled import CoupledLowRankDtN
from .dtn_form import DtNGravityForm
from .dtn_lowrank import apply_dirichlet_to_rows
from .gravity_solver import real_scalar_solver_parameters
from .utility import ensure_constant, interior_facet_integrand

__all__ = ["GravityPotential"]


class GravityPotential:
    """The potential part of a coupled solver: boundary form, residual, datum, low rank.

    Args:
      potential_space: the scalar space of `psi`, on the parent mesh.
      mechanics_mesh: the mesh of the density that drives the potential, a
        `Submesh` of the parent or the parent itself.
      gravity_bcs: the boundary conditions of `psi` in `DtNGravityForm`'s
        dictionary form (`"dtn"`, `"psi"`, `"flux"`, `"sigma"`,
        `"interior_sigma"`). A sheet given here enters the residual through
        the boundary form and the 2-D datum through `enclosed_mass_forms`. A
        caller that writes its sheets into the residual itself passes them to
        the datum as `(density, measure)` pairs instead (`update_total_mass`);
        the coupled GIA solver does that and refuses sheets here.
      Lambda: the self-gravity number `4 pi G rho_bar L / g_bar`. The
        boundary form's gravitational constant is `Lambda / (4 pi)`, so that
        a sheet's `4 pi G sigma` and the volume source agree.
      quad_degree: the boundary quadrature degree of the DtN form, `None`
        for its calibrated default.
      alpha: the Robin shift of the DtN form, `None` for its default.
      representation: `"multiplier"` (one `Real` unknown per DtN mode) or
        `"lowrank"` (the modes eliminated into a rank-n update).
    """

    def __init__(self, potential_space, mechanics_mesh, gravity_bcs, *,
                 Lambda, quad_degree=None, alpha=None,
                 representation="lowrank"):
        if representation not in ("multiplier", "lowrank"):
            raise ValueError(
                f"dtn_representation must be 'multiplier' or 'lowrank', got "
                f"{representation!r}.")
        self.representation = representation
        self.potential_mesh = potential_space.mesh()
        self.mechanics_mesh = mechanics_mesh
        self.cross_mesh = mechanics_mesh is not self.potential_mesh
        self.Lambda = Lambda
        self.form = DtNGravityForm(
            potential_space, gravity_bcs,
            gravitational_constant=Lambda / (4.0 * np.pi),
            quad_degree=quad_degree, alpha=alpha)
        #: `4 pi G`, written as the boundary form writes it.
        self.four_pi_G = 4 * np.pi * self.form.G
        self.set_measures()
        # Checks of the mesh: numbers nothing on the tape reads.
        with stop_annotating():
            self.check_geometry()
        self.set_monopole_datum()
        #: The low-rank operator, built by `build_lowrank_operator`.
        self.dtn_operator = None

    @property
    def n_multipliers(self) -> int:
        """The number of `Real` unknowns this part adds to a mixed space."""
        return 0 if self.representation == "lowrank" else self.form.n_multipliers

    # -- Measures and geometry ------------------------------------------------

    def set_measures(self) -> None:
        """The volume measures of the potential and of the density.

        `dx_g` is the potential's, over the whole parent; `dx_m` the
        density's. Both are intersected with the other mesh when the two
        differ: the arguments of the coupled residual live on a mixed space
        spanning both meshes, and a sub-block that a preconditioner extracts
        and compiles needs every integral to admit both domains. The
        intersection does not restrict the Laplacian to the mantle;
        `check_geometry` asserts that. `dx_parent` is the plain parent
        measure of the enclosed-mass system, whose arguments are `Real`
        fields of the parent alone.
        """
        parent, sub = self.potential_mesh, self.mechanics_mesh
        self.dx_parent = Measure("dx", domain=parent)
        if self.cross_mesh:
            self.dx_g = Measure("dx", domain=parent,
                                intersect_measures=(Measure("dx", domain=sub),))
            self.dx_m = Measure("dx", domain=sub,
                                intersect_measures=(Measure("dx", domain=parent),))
        else:
            self.dx_g = self.dx_parent
            self.dx_m = Measure("dx", domain=sub)

    def check_geometry(self) -> None:
        """Measure every DtN boundary and the cross-mesh measures.

        Each check is for a failure that gives a converged solve and no
        error: a DtN tag that matches nothing or part of a circle, an
        intersected measure that is empty, and a cross-mesh entity map that
        evaluates a parent field at the wrong points.
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
        volume = assemble(Constant(1.0) * Measure("dx", domain=self.mechanics_mesh))
        if not np.isclose(assemble(Constant(1.0) * self.dx_m), volume, rtol=1e-12):
            raise ValueError(
                "The intersected mantle measure does not assemble to the mantle "
                "volume; an empty intersection returns zero silently.")
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
                    f"A parent coefficient integrated over the mantle gives "
                    f"{crossed:.6e} against the mantle volume {volume:.6e}.")

    # -- The residual row -------------------------------------------------------

    def residual(self, psi, v, u, *, density, theta_psi, multipliers=None):
        """The potential row and, on the multiplier path, the DtN constraint rows.

        Args:
          psi, v: the potential and its test function (split components).
          u: the displacement that moves the density.
          density: the reference density `rho_0`.
          theta_psi: the row scaling, which multiplies the constraint rows too.
          multipliers: `None` on the low-rank path; otherwise the
            `(trial, test)` pairs of the multiplier fields, in space order.

        Returns:
          The form.
        """
        if self.representation == "lowrank":
            # `None` writes the Robin shift alone; the modal feedback is added
            # as `theta_psi B0 psi` by the solver's callbacks.
            multipliers = None
        F = dot(grad(psi), grad(v)) * self.dx_g
        F -= self.Lambda * density * dot(u, grad(v)) * self.dx_m
        F += self.form.boundary_residual(psi, v, multipliers,
                                         extra_flux=self.monopole_fluxes())
        return theta_psi * F

    def dirichlet_bcs(self, potential_space):
        """The strong `psi` conditions of the boundary form, on `potential_space`."""
        return [DirichletBC(potential_space, value, bc_id)
                for bc_id, value in self.form.dirichlet_bcs]

    # -- The 2-D monopole datum --------------------------------------------------

    def set_monopole_datum(self) -> None:
        """Prepare the 2-D exterior monopole flux datum.

        A 2-D exterior DtN boundary does not determine the `m = 0` mode of
        the trace, because the exterior monopole solution is logarithmic. The
        datum supplies its flux, `-2 G M / R`, with `M` the enclosed mass. The
        divergence-form volume source carries no net mass, so `M` is the mass
        of the sheets: those of the boundary form and those the caller passes
        to `update_total_mass`. In 3-D nothing is built.
        """
        self.monopole_boundaries = [
            bc_id for bc_id, _ in self.form.dtn_boundaries
            if self.potential_mesh.geometric_dimension == 2
            and self.form.boundary_geometry[bc_id][0] == "exterior"]
        self.real_space = None
        self.source_mass = None
        self.cavity_flux = None
        if not self.monopole_boundaries:
            return
        if len(self.monopole_boundaries) > 1:
            raise ValueError(
                f"2-D mesh with {len(self.monopole_boundaries)} exterior DtN "
                "boundaries: the monopole datum needs the mass enclosed by one "
                "exterior boundary.")
        clash = {bc_id for bc_id, _ in self.form.flux_bcs} & set(self.monopole_boundaries)
        if clash:
            raise ValueError(
                f"Boundary {sorted(clash, key=str)[0]}: 'flux' and a 2-D "
                "exterior 'dtn' both prescribe the m = 0 normal derivative.")
        self.real_space = FunctionSpace(self.potential_mesh, "R", 0)
        self.mesh_volume = assemble(Constant(1.0) * self.dx_parent)
        self.source_mass = Function(self.real_space, name="source_mass")
        self.cavity_flux = Function(self.real_space, name="cavity_flux")

    def monopole_fluxes(self) -> dict:
        """`{bc_id: flux}` for the boundary form's `extra_flux`."""
        out = {}
        for bc_id in self.monopole_boundaries:
            _, R = self.form.boundary_geometry[bc_id]
            flux = -2.0 * self.form.G * self.source_mass / R
            if self.form.flux_bcs:
                flux = flux - self.cavity_flux / (2.0 * np.pi * R)
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

        Args:
          extra_masses: `(density, measure)` pairs of sheets that the caller
            writes into the residual itself. `density` is UFL on the
            density's mesh with no quantity of the parent in it (a parent
            quantity would need a restriction on the parent's interior
            facets, which this method gives the `Real` test function only),
            `measure` a facet measure of that mesh, already restricted to its
            tag (intersected with the parent's interior facets in a
            cross-mesh system).
        """
        mu = TestFunction(self.real_space)
        nu = TrialFunction(self.real_space)
        identity = (nu / self.mesh_volume) * mu * self.dx_parent
        mass = None
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

    def update_total_mass(self, extra_masses=(), extra_scales=()) -> None:
        """Refresh the enclosed-mass scalars the datum reads, as recorded solves.

        Called at the start of every step, so the assembles and the solves
        land on the tape. `GravitySolver.update_total_mass` explains why a
        `Constant.assign(assemble(...))` would cut the tape and why a cached
        solver on this `Real` space is not used.

        Args:
          extra_masses: `(density, measure)` pairs of the caller's sheets that
            the datum counts (`enclosed_mass_forms`).
          extra_scales: `(density, measure)` pairs of every sheet of the
            caller, counted or not, for the scale of `check_net_mass`.
        """
        if not self.monopole_boundaries:
            return
        identity, mass, flux = self.enclosed_mass_forms(extra_masses)
        # Each solve starts from zero. A linear solve under `ksponly` adds an
        # update to its initial guess, so a guess left from the last step
        # puts that step's round-off into the datum; from zero the datum is a
        # function of the present solution alone, and a run restarted from a
        # state reproduces it bit for bit.
        if mass is not None:
            self.source_mass.assign(0.0)
            solve(identity == mass, self.source_mass,
                  solver_parameters=real_scalar_solver_parameters)
        if flux is not None:
            self.cavity_flux.assign(0.0)
            solve(identity == flux, self.cavity_flux,
                  solver_parameters=real_scalar_solver_parameters)
        self.check_net_mass(extra_scales)

    def check_net_mass(self, extra_scales=()) -> None:
        """Refuse a doubly anchored gauge; warn on a small net sheet mass.

        The scale is the mass the sheets move, `sum int |sigma|`. With a
        strong `psi` condition and a nonzero net mass, the Robin monopole
        relation and the Dirichlet condition are two anchors of one constant.
        A diagnostic: nothing on the tape reads it, so it is not recorded.

        Args:
          extra_scales: `(density, measure)` pairs of the caller's sheets.
        """
        if not self.monopole_boundaries:
            return
        with stop_annotating():
            self._check_net_mass(extra_scales)

    def _check_net_mass(self, extra_scales) -> None:
        scale = 0.0
        for bc_id, sigma, integral_type in self.form.sigma_bcs:
            scale += assemble(self.form.sheet_integral(
                abs(ensure_constant(sigma)), bc_id, integral_type))
        for density, measure in extra_scales:
            scale += assemble(abs(density) * measure)
        mass = abs(float(self.source_mass))
        relative = mass / scale if scale > 0.0 else 0.0
        anchored = relative > 1e-8 or float(self.cavity_flux) != 0.0
        if anchored and self.form.dirichlet_bcs:
            raise ValueError(
                "Net enclosed mass is nonzero on a 2-D mesh with an exterior DtN "
                "boundary and a strong 'psi' condition: two anchors of the "
                "potential's additive constant.")
        if 1e-8 < relative < 1e-4:
            warn(f"Net sheet mass {float(self.source_mass):.3e} is "
                 f"{relative:.1e} of the sheets' own scale, which is the "
                 "signature of leakage from a load that does not conform to "
                 "cell edges.")

    # -- The low-rank representation ----------------------------------------------

    def build_lowrank_operator(self, mixed_space, potential_index, theta):
        """`B = theta_psi C^T W C` on the potential rows of `mixed_space`.

        Args:
          mixed_space: the mixed space of the coupled solution.
          potential_index: the index of the potential field in it.
          theta: a callable returning `theta_psi` as a number, read at every
            application so that a control on `Lambda` or `B_mu` reaches it.

        Returns:
          The `CoupledLowRankDtN`, also stored as `dtn_operator`.
        """
        mode_rows = self.form.build_mode_rows()
        for (bc_id, dtn), rows in zip(self.form.dtn_boundaries, mode_rows, strict=True):
            side, radius = self.form.boundary_geometry[bc_id]
            expected = [mode.key for mode in dtn.mode_metadata(side, radius)]
            if list(rows.keys) != expected:
                raise RuntimeError(
                    f"boundary {bc_id}: the mode rows are ordered {list(rows.keys)} "
                    f"but the descriptor's order is {expected}.")
        # The rows of a strong `psi` condition are eliminated from the stock
        # operator, so the update must not write into them either.
        psi_space = mixed_space.sub(potential_index)
        constrained = set()
        for bc_id, value in self.form.dirichlet_bcs:
            constrained.update(np.asarray(
                DirichletBC(psi_space, value, bc_id).nodes, dtype=np.int64).tolist())
        apply_dirichlet_to_rows(mode_rows, constrained)
        self.constrained_dofs = constrained
        self.dtn_operator = CoupledLowRankDtN(
            mixed_space, potential_index, mode_rows, theta,
            self.potential_mesh.comm)
        return self.dtn_operator
