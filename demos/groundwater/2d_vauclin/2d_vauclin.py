# Recharge of a two-dimensional water table
# =========================================
#
# Variably saturated groundwater flow governs a wide range of hydrological
# processes, from infiltration and recharge of shallow aquifers to contaminant
# transport in the vadose zone. A standard two-dimensional benchmark for
# numerical methods in this setting is the water-table recharge problem
# introduced by Vauclin et al. (1979), in which a localised infiltration flux
# imposed at the top of an initially hydrostatic soil column drives an
# unsaturated wetting front downward while deforming the underlying water
# table laterally. Since its publication, the problem has served as a
# reference test in numerous subsequent studies (e.g., Clement et al. 1994;
# Shen and Phanikumar 2010; Xu et al. 2012; Clement et al. 2023), primarily
# as a convergence and stability benchmark for new Richards-equation solvers:
# the geometry is minimal, yet the simultaneous presence of a moving
# saturated/unsaturated interface and a strongly non-linear moisture
# retention curve exercises both the advective (gravity-driven) and diffusive
# (capillary) components of the flow.
#
# The computational domain is a rectangle of 3 m by 2 m. At $t = 0$, the
# region below $z = 0.65$ m is fully saturated, such that the pressure head
# satisfies $h(x, z, 0) = 0.65 - z$, which vanishes at the water table and
# is linearly negative above it. The bottom and left boundaries are closed
# ($\vec{q} \cdot \vec{n} = 0$), the right boundary is held at the initial
# head as a Dirichlet water table, and the top boundary receives water at
# 14.8 cm/hour over the strip $x \le 0.5$ m. The original paper integrates
# the problem to eight hours; here we terminate the simulation at two hours,
# which is sufficient to develop the wetting front and exercise the adaptive
# time-stepping protocol while keeping the run time of the demo modest.
#
# Our tutorial proceeds as follows. We first introduce the Richards equation
# in moisture-content form and define the Haverkamp constitutive relations
# used for the soil hydraulic properties. We then outline the discontinuous
# Galerkin spatial discretisation and the mass-conservative stage-value time
# integration, and describe the three-tier adaptive timestep controller that
# drives the run. The remainder of the file sets up and executes the problem
# in Firedrake.

# Governing equations
# -------------------
#
# Richards equation governs the flow of the liquid phase in an unsaturated
# porous medium, under the simplifying assumption that the non-wetting gas
# phase has negligible density and viscosity relative to the liquid and
# therefore remains stationary at atmospheric pressure. Combining the
# mass-conservation statement for the liquid phase,
# $\partial \theta/\partial t + \nabla \cdot \vec{q} = s$, with the Darcy--Buckingham flux law,
# $\vec{q} = -K(h) \nabla (h + z)$, yields the moisture-content form of
# Richards equation:
#
# $$
#   \frac{\partial \theta}{\partial t}
#     - \nabla \cdot \bigl( K(h) \nabla (h + z) \bigr) = s,
# $$
#
# where $\theta$ is the volumetric water content (dimensionless), $\vec{q}$
# is the volumetric flux [L/T], $h$ is the pressure head [L], $z$ is the
# upward vertical coordinate [L], $K(h)$ is the hydraulic conductivity
# [L/T], and $s$ is a volumetric source term [1/T]. The sum $h + z$ is the
# total hydraulic head, so that gravity enters the flux through the unit
# gradient $\nabla z = \vec{e}_z$.
#
# The water content $\theta$ is not an independent unknown but an algebraic
# function of the pressure head, $\theta = \theta(h)$, prescribed by the
# soil-water retention curve introduced in the following section. Expanding
# the time derivative via the chain rule yields the head-based form:
#
# $$
#   C(h)\, \frac{\partial h}{\partial t}
#     - \nabla \cdot \bigl( K(h) \nabla (h + z) \bigr) = s,
#   \qquad C(h) \equiv \frac{\partial \theta}{\partial h},
# $$
#
# where $C(h)$ [1/L] is the specific moisture capacity. In the fully
# saturated regime, the water content reaches its upper bound $\theta_s$
# and $C$ vanishes, so the time-derivative term disappears and the equation
# becomes locally elliptic. In practice this is accommodated by augmenting
# $C$ with an elastic storage term $S_s\, S(h)$, where $S_s$ is the
# specific storage and $S(h)$ is the effective saturation; the notation
# $C$ is retained for the combined coefficient throughout.
#
# We solve Richards equation in the moisture-content form. Although both
# forms are mathematically equivalent at the continuous level, the former
# conserves mass exactly when discretised with a stage-value time scheme,
# whereas the latter accumulates a systematic mass-balance error owing to
# the chain-rule linearisation of $\partial \theta/\partial t$ as
# $C\,\partial h/\partial t$ (Celia et al. 1990). This distinction is revisited
# in the time-integration section below.

# Soil hydraulic model
# --------------------
#
# Closure of Richards equation requires two empirical relations: the
# soil-water retention curve $\theta(h)$ and the hydraulic conductivity
# function $K(h)$. For the Vauclin benchmark, both take the form
# proposed by Haverkamp et al. (1977),
#
# $$
#   \theta(h) = \theta_r + (\theta_s - \theta_r)
#               \frac{\alpha}{\alpha + |h|^{\beta}},
#   \qquad
#   K(h) = K_s\, \frac{A}{A + |h|^{\gamma}},
# $$
#
# valid in the unsaturated regime $h < 0$, with $\theta = \theta_s$ and
# $K = K_s$ in the saturated regime $h \ge 0$. Here $\theta_r$ and
# $\theta_s$ are the residual and saturated water contents respectively,
# $K_s$ is the saturated hydraulic conductivity, and $\alpha$, $\beta$,
# $A$, $\gamma$ are empirical shape parameters that control the
# transition of each curve away from saturation. Both relations are
# differentiable in $h$, so the specific moisture capacity
# $C(h) = \partial \theta/\partial h$ and the conductivity derivative
# $\partial K/\partial h$ are available in closed form and enter the
# Jacobian of the discrete non-linear system directly.
#
# We adopt the parameter set originally reported by Haverkamp et al.
# (1977) and reproduced in the Vauclin experiment, converted to SI
# units (metres and seconds): $\theta_r = 0$, $\theta_s = 0.30$,
# $K_s = 9.722 \times 10^{-5}$ m/s (equivalent to 35 cm/hr),
# $\alpha = 4.00 \times 10^{4}/100^{\beta}$ m$^{\beta}$, $\beta = 2.90$,
# $A = 2.99 \times 10^{6}/100^{\gamma}$ m$^{\gamma}$, and $\gamma = 5$.
# The specific storage is set to zero, so the saturated soil skeleton
# is treated as incompressible. These numerical values are supplied by
# the `gwassess` companion package, which also handles the unit
# conversion from the centimetre-hour system of the original paper.

# Finite element discretisation
# -----------------------------
#
# We discretise Richards equation in space using the discontinuous
# Galerkin (DG) method. Let $\Omega$ denote the computational domain with
# Lipschitz boundary $\partial \Omega$ and outward unit normal $\vec{n}$,
# and let $V$ be a discrete space of pressure heads with broken
# continuity across inter-element facets. Multiplying the moisture-content
# form by a test function $v \in V$, integrating element-wise, and
# applying integration by parts to the divergence term yields the weak
# residual
#
# $$
#   \int_{\Omega} \frac{\partial \theta(h)}{\partial t}\, v\, dx
#   + \int_{\Omega} K(h)\, \nabla h \cdot \nabla v\, dx
#   + \int_{\Omega} K(h)\, \frac{\partial v}{\partial z}\, dx
#   - \int_{\partial \Omega}
#       \bigl[ K(h)\, \nabla (h + z) \cdot \vec{n} \bigr] v\, ds
#   = \int_{\Omega} s\, v\, dx,
# $$
#
# where the separation into the diffusive term $K\, \nabla h \cdot \nabla v$
# and the gravitational term $K\, \partial v/\partial z$ follows from
# $\nabla (h + z) = \nabla h + \vec{e}_z$ and identifies each component
# of the flux with its natural numerical-flux treatment. On discontinuous
# function spaces the surface integral decomposes into an interior-facet
# contribution, where appropriate numerical fluxes are imposed across
# element boundaries, and a boundary contribution that implements the
# prescribed boundary conditions.
#
# For the diffusive term we adopt the symmetric interior penalty Galerkin
# (SIPG) formulation of Arnold et al. (2002), in which the interior-facet
# integral is replaced by a symmetrised combination of the average normal
# flux and a penalty term proportional to the jump in $h$; the penalty
# scales as $(p+1)^2 / h_F$, with $h_F$ a characteristic facet length,
# which guarantees coercivity of the discrete bilinear form. The
# gravitational term is treated with an upwind numerical flux: on each
# facet the value of $K$ is taken from the upstream side with respect to
# the gravity direction $\vec{e}_z$, so that the direction of propagation
# of the infiltration front is respected by the discrete operator.
# Dirichlet conditions on $h$ are imposed weakly via a Nitsche-style term
# consistent with the SIPG symmetrisation, while Neumann/flux conditions
# replace the diffusive and gravitational facet integrals on $\partial\Omega$
# by the prescribed normal flux directly.
#
# For the Vauclin benchmark we employ a quadrilateral mesh with $p = 2$
# discontinuous Galerkin elements ($\mathrm{DQ}_2$) for the pressure
# head. This choice yields third-order spatial convergence for smooth
# solutions and preserves local mass balance owing to the element-wise
# divergence theorem. Full derivation of the weak form and of the
# associated facet integrals is provided in the accompanying paper.

# Time integration and mass conservation
# --------------------------------------
#
# For time discretisation we employ a second-order stiffly accurate
# diagonally implicit Runge--Kutta scheme (DIRK22), applied through the
# Irksome framework (Farrell et al. 2021) that is integrated into
# G-ADOPT. Stiffly accurate schemes are appropriate for the stiff
# regime near saturation fronts, in which the discrete Jacobian becomes
# increasingly ill-conditioned as $C(h)$ approaches zero and the
# character of the equation transitions from parabolic to locally
# elliptic. Richards equation raises an additional complication,
# specific to its moisture-content form: the time derivative of the
# dependent variable is not the unknown $h$ itself but the non-linear
# composite $\theta(h)$, which introduces an ambiguity in how the
# discrete time derivative is interpreted.
#
# Two interpretations are possible at each stage of the Runge--Kutta
# scheme. The derivative interpretation applies the chain rule
# analytically and advances the discrete system for the variable $h$,
#
# $$
#   C(h^*)\, \frac{h^{n+1} - h^{n}}{\Delta t} \approx
#   \frac{\theta(h^{n+1}) - \theta(h^{n})}{\Delta t},
# $$
#
# which incurs a linearisation error proportional to $\partial^2 \theta/\partial h^2$
# per step; this error does not cancel in time and
# accumulates as a systematic drift in the discrete mass balance. The
# stage-value interpretation instead advances the composite quantity
# $\theta(h)$ directly, computing the finite difference
# $(\theta(h^{n+1}) - \theta(h^{n}))/\Delta t$ exactly, and conserves
# mass to solver tolerance (Farrell et al. 2021). In this tutorial we
# adopt the latter, which is the default for ``RichardsSolver`` in
# G-ADOPT and requires the Butcher tableau to be stiffly accurate;
# this requirement is satisfied by DIRK22, as well as by Backward
# Euler, RadauIIA, and Crank--Nicolson.
#
# At each time step the non-linear stage equations are linearised with
# Newton's method, and the resulting linear systems are solved with a
# direct LU factorisation, which is sufficient for this two-dimensional
# problem. For fully three-dimensional applications a Krylov method
# with multigrid preconditioning is available; an example is given in
# the companion Cockett benchmark.

# Adaptive timestepping
# ---------------------
#
# The numerical difficulty of Richards-equation problems is strongly
# heterogeneous in time. During quiescent phases the solution evolves
# smoothly and Newton iterations converge quickly, permitting large
# time steps; when an infiltration front arrives, however, $\theta(h)$
# becomes sharply non-linear in a thin band of cells, the Jacobian
# becomes stiff, and Newton fails to converge or requires many
# iterations. A fixed time step chosen for the front regime incurs
# excessive computational cost throughout the quiescent phases, while
# one chosen for the smooth regime risks non-convergence at the front.
# We therefore employ an adaptive time-step controller,
# ``RichardsTimestepAdaptor``, that adjusts $\Delta t$ between outer
# solves in response to three physically motivated signals.
#
# The first two tiers impose a Courant--Friedrichs--Lewy (CFL) ceiling
# derived from the local flow state. The advective tier bounds $\Delta t$
# through the gravity-driven Darcy flux $\vec{q}_g = K(h)\,\vec{e}_z$,
# requiring that a fluid parcel does not traverse more
# than a prescribed fraction of a cell per step,
#
# $$
#   \Delta t_{\mathrm{adv}} =
#   \frac{\mathrm{CFL}_{\mathrm{adv}}}{\max_{\Omega} |J^{-1} \vec{q}_g|},
# $$
#
# where $J$ is the element Jacobian. The diffusive tier bounds $\Delta t$
# through the hydraulic diffusivity $D(h) = K(h)/C(h)$,
#
# $$
#   \Delta t_{\mathrm{diff}} =
#   \frac{\mathrm{CFL}_{\mathrm{diff}}}{2 d \cdot \max_{\Omega} D(h)/h_e^2},
# $$
#
# with $d$ the spatial dimension and $h_e$ the element diameter. This
# expression is a classical explicit-scheme stability bound and is
# therefore an accuracy heuristic, rather than a stability requirement,
# for the implicit DIRK22 scheme used here. The third tier modulates
# the allowed growth rate between steps according to the iteration
# count of the previous Newton solve: when Newton converges within a
# target number of iterations the step is allowed to grow by a factor
# ``snes_scale_up`` $> 1$, whereas when the iteration count is
# exceeded the step is forced to shrink by a factor
# ``snes_scale_down`` $< 1$, independently of the CFL tier bounds.
#
# The diffusive tier requires additional care for problems with dry
# initial conditions. In regions of low saturation,
# $C(h) = \partial \theta/\partial h$ becomes very small, so that $D(h) = K(h)/C(h)$
# diverges and the diffusive bound approaches zero. For implicit
# schemes this singularity is not a stability concern, so we cap
# $\Delta t_{\mathrm{diff}}$ from below through the parameter
# ``minimum_diffusive_dt``; in the present setup we set it large
# enough to suppress the diffusive tier and leave $\Delta t$ under
# the joint control of the advective and SNES tiers.
#
# The adaptor mutates the Firedrake ``Constant`` that stores $\Delta t$
# without altering the solver's internal state, so that the
# stage-value formulation remains intact and the mass-conservation
# property introduced in the previous section is preserved across
# varying time steps. The outer loop implements the
# convergence-recovery protocol: when Newton diverges, the pre-step
# state is restored, $\Delta t$ is halved, and the solve is
# re-attempted; each retry is reported back to the adaptor via
# ``record_retry()``, which triggers a two-step cool-down during
# which the step size is held constant rather than grown. Without
# the cool-down, the SNES iteration count on the first
# post-recovery step would be artificially small, because it would
# reflect the reduced $\Delta t$ rather than the underlying
# physics, and the adaptor would immediately grow $\Delta t$ back
# into the regime in which convergence was lost.

from gadopt import *
from firedrake.exceptions import ConvergenceError
import gwassess

# The reference problem setup is obtained from the ``gwassess`` companion
# package, which provides the domain dimensions, boundary-condition
# values, and Haverkamp parameters for the Vauclin benchmark in SI units.

vauclin_solution = gwassess.VauclinRichardsSolution2D()
Lx = vauclin_solution.Lx
Ly = vauclin_solution.Ly

# The computational mesh is a regular $46 \times 31$ grid of
# quadrilateral elements covering the $3~\mathrm{m} \times 2~\mathrm{m}$
# domain. The mesh is marked as Cartesian, indicating that gravity acts
# in the negative $z$-direction; non-Cartesian meshes in G-ADOPT instead
# assume a radially inward gravity. The pressure head $h$ is represented
# in a discontinuous piecewise-quadratic space ($\mathrm{DQ}_2$), and an
# auxiliary vector space $W$ of the same polynomial degree is declared
# for the post-processed Darcy flux used in the diagnostic output.

nodes_x, nodes_y = 46, 31
mesh = RectangleMesh(nodes_x, nodes_y, Lx, Ly, name="mesh", quadrilateral=True)
mesh.cartesian = True
X = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DQ", 2)
W = VectorFunctionSpace(mesh, "DQ", 2)

# The constitutive model is instantiated from the parameter dictionary
# returned by ``gwassess`` and wrapped in a ``HaverkampCurve`` object,
# which exposes the moisture content $\theta(h)$, the relative
# permeability $K(h)/K_s$, and the specific moisture capacity $C(h)$ as
# differentiable UFL expressions for direct use in residual and Jacobian
# assembly.

soil_params = vauclin_solution.get_soil_parameters()

soil_curve = HaverkampCurve(
    theta_r=soil_params['theta_r'],
    theta_s=soil_params['theta_s'],
    Ks=soil_params['Ks'],
    Ss=soil_params['Ss'],
    alpha=soil_params['alpha'],
    beta=soil_params['beta'],
    A=soil_params['A'],
    gamma=soil_params['gamma'],
)

moisture_content = soil_curve.moisture_content
hydraulic_conductivity = soil_curve.hydraulic_conductivity

# The initial pressure head is the hydrostatic profile
# $h(x, z, 0) = 0.65 - 1.001\, z$. The factor $1.001$, rather than exactly unity,
# ensures that the upper portion of the domain starts marginally dry
# rather than at the saturation boundary; this avoids an artificial
# singularity in $C(h)$ at $t = 0$. Two Firedrake ``Function`` objects
# are allocated: $h$ for the current solution, and $h_{\mathrm{old}}$
# as a buffer for the pre-step state that is used by the retry
# protocol.

water_table_height = vauclin_solution.WATER_TABLE_HEIGHT
h_ic = Function(V, name="InitialCondition").interpolate(
    water_table_height - 1.001 * X[1]
)
h = Function(V, name="PressureHead").interpolate(h_ic)
h_old = Function(V, name="PreviousSolution").interpolate(h_ic)

theta = Function(V, name="MoistureContent").interpolate(moisture_content(h))
q = Function(W, name="VolumetricFlux")
h_initial_plot = Function(V, name="InitialPressureHead").interpolate(h)
theta_initial_plot = Function(V, name="InitialMoistureContent").interpolate(theta)

# The top infiltration flux is smoothed in both space and time. A
# hyperbolic-tangent temporal ramp of the form $\tanh(\omega t)$, with
# $\omega = 1.25 \times 10^{-4}~\mathrm{s}^{-1}$, brings the flux from
# zero to its nominal value over the first few thousand seconds and
# avoids a sudden initial transient, while a pair of hyperbolic tangents
# in $x$ localises the infiltration to the strip $|x| \le 0.5$ m with a
# smooth transition region of width $\sim 0.1$ m on either side.
# Boundary conditions are assembled into a dictionary indexed by
# boundary identifier: the top receives the infiltration flux, the right
# is held at the hydrostatic Dirichlet head, and the bottom and left are
# closed with a zero normal flux.

time_var = Constant(0.0)
infiltration_rate = vauclin_solution.INFILTRATION_RATE
infiltration_width = vauclin_solution.INFILTRATION_WIDTH

top_flux = tanh(0.000125 * time_var) * infiltration_rate * (
    0.5 * (1 + tanh(10 * (X[0] + infiltration_width)))
    - 0.5 * (1 + tanh(10 * (X[0] - infiltration_width)))
)

boundary_ids = get_boundary_ids(mesh)
richards_bcs = {
    boundary_ids.left: {'flux': 0.0},
    boundary_ids.right: {'h': h_ic},
    boundary_ids.bottom: {'flux': 0.0},
    boundary_ids.top: {'flux': top_flux},
}

# With the constitutive model and boundary data in place, we instantiate
# the solver. ``RichardsSolver`` accepts the solution field $h$, the
# soil curve, a Firedrake ``Constant`` holding the time step $\Delta t$,
# and the time-stepper class. We request a direct LU factorisation of
# the linear systems, which is adequate for this two-dimensional
# problem, and set the quadrature degree to five, which integrates the
# Haverkamp non-linearities accurately while avoiding unnecessary
# over-integration.

dt = Constant(10.0)
richards_solver = RichardsSolver(
    h,
    soil_curve,
    delta_t=dt,
    timestepper=DIRK22,
    bcs=richards_bcs,
    solver_parameters="direct",
    quad_degree=5,
)

# The adaptive time-step controller is configured with a lower bound of
# $0.1$ s, an upper bound of $300$ s, and a large value of
# ``minimum_diffusive_dt`` ($10^6$ s) that effectively disables the
# diffusive tier, as discussed in the preceding section. A two-step
# cool-down following each retry suppresses premature regrowth of
# $\Delta t$.

adaptor = RichardsTimestepAdaptor(
    dt,
    richards_solver,
    target_cfl=0.5,
    target_diffusive_cfl=0.5,
    maximum_timestep=300.0,
    minimum_timestep=0.1,
    minimum_diffusive_dt=1e6,
    post_retry_cooldown=2,
)

# For the diagnostic output, we declare quadrature measures of degree
# five and compute the initial total moisture content
# $M_0 = \int_\Omega \theta(h^0)\, dx$, which serves as the reference against
# which the time-integrated boundary flux is later compared. A
# ``ParameterLog`` writes a tabular record of diagnostic quantities,
# and a ``VTKFile`` collects snapshots of the pressure head, the
# moisture content, and the post-processed volumetric flux for
# visualisation.

ds_mesh = Measure("ds", domain=mesh, metadata={"quadrature_degree": 5})
dx_mesh = Measure("dx", domain=mesh, metadata={"quadrature_degree": 5})

initial_mass = assemble(moisture_content(h) * dx_mesh)
external_flux = 0

plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt min_h max_h mass_balance")

output = VTKFile("vauclin.pvd")
output.write(h, theta, q, time=0)

# We run the simulation for a wall-clock duration of $7200$ s (two
# hours). The main time loop performs the sequence: save the pre-step
# solution, update the time coordinate that enters the top-flux
# expression, query the adaptor for the next $\Delta t$, and attempt
# the non-linear solve. On convergence failure the loop halves $\Delta t$
# and retries up to four times, reporting each failure back to the
# adaptor as described above; after five unsuccessful attempts the
# exception is re-raised so that a genuine pathology is visible to the
# caller rather than silently absorbed.

t_final = 7200.0  # 2 hours
time = 0.0
timestep_count = 0
max_retries = 4

while time < t_final:
    h_old.assign(h)
    time_var.assign(time)
    adaptor.update_timestep()

    for attempt in range(max_retries + 1):
        try:
            richards_solver.solve()
            break
        except ConvergenceError:
            if attempt == max_retries:
                raise
            h.assign(h_old)
            dt.assign(max(float(dt) * 0.5, 0.1))
            adaptor.record_retry()

    time += float(dt)
    timestep_count += 1

    # After a successful step, the volumetric flux is reconstructed from
    # the midpoint conductivity and the gradient of the total head, and
    # integrated against the outward facet normal to accumulate the
    # cumulative boundary flux. The diagnostic ratio
    # $(M(t) - M_0) / \int_0^t \int_{\partial\Omega} \vec{q} \cdot \vec{n}\, ds\, dt$
    # would approach unity if the interpolated flux $\vec{q} = -K \nabla(h + z)$
    # were identical to the discrete flux assembled by
    # the solver; in practice it deviates by order $1\,\%$, which
    # reflects the projection error of the post-processing step rather
    # than any loss of mass in the discrete system. The underlying
    # stage-value integration conserves the discrete mass residual to
    # solver tolerance.
    K = hydraulic_conductivity((h + h_old) / 2)
    q.interpolate(-K * grad((h + h_old) / 2 + X[1]))
    external_flux += assemble(float(dt) * dot(q, -FacetNormal(mesh)) * ds_mesh)

    current_mass = assemble(moisture_content(h) * dx_mesh)
    mass_balance = (current_mass - initial_mass) / external_flux if external_flux != 0 else 0

    min_h = h.dat.data.min()
    max_h = h.dat.data.max()

    plog.log_str(
        f"{timestep_count} {time} {float(dt)} {min_h} {max_h} {mass_balance}"
    )

    if timestep_count % 2 == 0:
        theta.interpolate(moisture_content(h))
        output.write(h, theta, q, time=time)

plog.close()

# +
h_final_plot = Function(V, name="FinalPressureHead").interpolate(h)
theta.interpolate(moisture_content(h))
theta_final_plot = Function(V, name="FinalMoistureContent").interpolate(theta)

h_plot_min = min(h_initial_plot.dat.data_ro.min(), h_final_plot.dat.data_ro.min())
h_plot_max = max(h_initial_plot.dat.data_ro.max(), h_final_plot.dat.data_ro.max())
theta_plot_min = min(theta_initial_plot.dat.data_ro.min(), theta_final_plot.dat.data_ro.min())
theta_plot_max = max(theta_initial_plot.dat.data_ro.max(), theta_final_plot.dat.data_ro.max())
# -

# We visualise the initial and final states with two-panel Firedrake-native
# plots for pressure head and moisture content. The colour limits are fixed
# across time to make changes directly comparable.
#
# region tags=["active-ipynb"]
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
#
# h_initial_collection = tripcolor(
#     h_initial_plot,
#     axes=axes[0],
#     cmap="RdBu_r",
#     vmin=h_plot_min,
#     vmax=h_plot_max,
# )
# axes[0].set_title("Initial pressure head")
# fig.colorbar(h_initial_collection, ax=axes[0], label="h [m]")
#
# theta_initial_collection = tripcolor(
#     theta_initial_plot,
#     axes=axes[1],
#     cmap="viridis",
#     vmin=theta_plot_min,
#     vmax=theta_plot_max,
# )
# axes[1].set_title("Initial moisture content")
# fig.colorbar(theta_initial_collection, ax=axes[1], label=r"$\theta$ [-]")
# endregion
#
# region tags=["active-ipynb"]
# fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
#
# h_final_collection = tripcolor(
#     h_final_plot,
#     axes=axes[0],
#     cmap="RdBu_r",
#     vmin=h_plot_min,
#     vmax=h_plot_max,
# )
# axes[0].set_title("Final pressure head")
# fig.colorbar(h_final_collection, ax=axes[0], label="h [m]")
#
# theta_final_collection = tripcolor(
#     theta_final_plot,
#     axes=axes[1],
#     cmap="viridis",
#     vmin=theta_plot_min,
#     vmax=theta_plot_max,
# )
# axes[1].set_title("Final moisture content")
# fig.colorbar(theta_final_collection, ax=axes[1], label=r"$\theta$ [-]")
# endregion
# +
PETSc.Sys.Print(f"Simulation complete: {timestep_count} timesteps")
PETSc.Sys.Print(f"Final min_h={min_h:.4f}, max_h={max_h:.4f}, mass_balance={mass_balance:.6f}")
# -
