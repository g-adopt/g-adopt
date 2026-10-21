# Creep and recovery: from Maxwell to Burgers rheology
# ==================================================
#
# A Maxwell body combines instantaneous elastic deformation with permanent
# viscous creep. A Burgers body adds a third response: deformation that builds
# up over time under load, then gradually recovers after unloading. This
# tutorial shows how to model that **recoverable transient strain** using
# `BurgersApproximation` and `BurgersSolver`.
#
# We stretch a small 2-D block horizontally while compressing it vertically,
# then remove the load. The stress is uniform, so we can see the material
# response directly. This is a pure-shear creep experiment with gravity
# switched off. The [GIA base case](base_case.ipynb) shows how displacement,
# gravity and a spatially varying ice load interact in a larger problem.
#
# What changes when we add a Kelvin element?
# -----------------------------------------
#
# For small deformations, the deviatoric strain is
#
# $$
# \boldsymbol e_d = \operatorname{sym}(\nabla\boldsymbol u)
#                  - \tfrac13(\nabla\cdot\boldsymbol u)\boldsymbol I.
# $$
#
# A Burgers body is an elastic spring, a Kelvin element (a spring and dashpot
# in parallel), and a permanent-creep dashpot, connected **in series**. Their
# strains add, and they carry the same deviatoric stress:
#
# $$
# \boldsymbol e_d = \boldsymbol e_e+\boldsymbol e_t+\boldsymbol e_v,
# \qquad
# \boldsymbol s = 2\mu\boldsymbol e_e
#              = 2\mu(\boldsymbol e_d-\boldsymbol e_t-\boldsymbol e_v).
# $$
#
# The two internal strains evolve according to
#
# $$
# \dot{\boldsymbol e}_t
#   = \frac{\boldsymbol s-2\mu_t\boldsymbol e_t}{2\eta_t},
# \qquad
# \dot{\boldsymbol e}_v = \frac{\boldsymbol s}{2\eta}.
# $$
#
# Under constant stress, the Kelvin strain approaches
# $\boldsymbol s/(2\mu_t)$ on the **retardation time** $\tau_t=\eta_t/\mu_t$.
# After stress removal it decays on that same timescale. Permanent creep
# continues accumulating while stress is applied. Setting the Kelvin strain
# to zero leaves the Maxwell spring and dashpot.
#
# We compare linear Maxwell and Burgers bodies with the **same instantaneous
# shear modulus $\mu$ and steady viscosity $\eta$**. Their initial elastic
# response and long-time creep rate therefore agree. Burgers has additional
# deformation during loading and additional recovery during unloading.
# The bulk response remains elastic in both models.

from gadopt import *
import numpy as np
import matplotlib.pyplot as plt

# + tags=["active-ipynb"]
# %matplotlib inline
# -

# A small, homogeneous block
# --------------------------
#
# All quantities below are nondimensional. We use $\mu_0$ as the stress scale,
# $\eta_0/\mu_0$ as the time scale, and the block size as the length scale.
# For $\mu_0=70$ GPa and $\eta_0=10^{21}$ Pa s, one time unit is about 453 years.
# The applied stress, $10^{-5}\mu_0$, keeps strains small. Its amplitude does
# not affect the shape of the linear response.
#
# The mesh stays fixed: $\boldsymbol u$ is total displacement from the initial
# configuration. A linear displacement field represents this homogeneous
# experiment exactly, so continuous linear displacement and discontinuous
# constant tensor fields are sufficient here. Spatially varying problems
# generally need a finer mesh and higher-order spaces.

# +
mesh = UnitSquareMesh(4, 4)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)
V = VectorFunctionSpace(mesh, "CG", 1)
S = TensorFunctionSpace(mesh, "DG", 0)

mu, eta = 1.0, 1.0
mu_t, eta_t = 1.0, 0.3
stress_amplitude = 1e-5
transition_stress = 1e-5
exponent = 1
# -

# The Burgers solution has three fields, in this order: displacement,
# recoverable transient strain and permanent viscous strain. A new `Function`
# starts at zero, giving an undeformed, unloaded initial state. The two tensor
# fields are **physical strain contributions**, rather than individual
# Maxwell relaxation modes.

Z = MixedFunctionSpace([V, S, S])
z = Function(Z)
u, transient, permanent = z.subfunctions
u.rename("displacement")
transient.rename("transient strain")
permanent.rename("permanent strain")

# Prescribing a loading cycle
# ---------------------------
#
# We ramp the stress up between $t=0$ and $0.1$, hold it until $t=0.9$, then
# remove it by $t=1$. Recovery continues until $t=3$. Short ramps resolve the
# elastic response without introducing instantaneous jumps into the plots.
#
# On the right boundary the traction is $(\sigma,0)$; on the top it is
# $(0,-\sigma)$. The left and bottom are symmetry boundaries with zero normal
# displacement. These conditions give a uniform stress with $s_{xx}=\sigma$,
# $s_{yy}=-\sigma$ and zero shear stress. The displacement has the form
# $\boldsymbol u=(a(t)x,-a(t)y)$, so there is no volume change. The omitted
# out-of-plane deviatoric stress is zero in this particular 2-D experiment.

traction = Constant(0)
bcs = {
    boundary.left: {"ux": 0},
    boundary.bottom: {"uy": 0},
    boundary.right: {"stress": as_vector((traction, 0))},
    boundary.top: {"stress": as_vector((0, -traction))},
}
rise, unload_start, unload_end, end = 0.1, 0.9, 1.0, 3.0
dt = Constant(0.005)

# Choosing the approximation and solver
# -------------------------------------
#
# `shear_modulus` and `viscosity` describe the elastic spring and permanent
# dashpot. The two `transient_*` parameters describe the Kelvin element.
# Here $\tau_t=0.3$ time units, about 136 years with the dimensional scales
# above. `exponent=1` selects linear permanent creep. We set `B_mu=0` to remove
# gravitational restoring forces and isolate the constitutive response.
# Density and bulk modulus are required by the compressible approximation;
# their values do not influence this volume-preserving, gravity-free solution.

approximation = BurgersApproximation(
    bulk_modulus=5, density=1, shear_modulus=mu, viscosity=eta,
    transient_shear_modulus=mu_t, transient_viscosity=eta_t,
    exponent=exponent, transition_stress=transition_stress, B_mu=0,
)

# `BurgersSolver` solves for all three fields together using backward Euler.
# Stress, transient strain and permanent strain are evaluated at the new time
# level. A successful `solve()` also updates the stored history. Initialise
# the solution before constructing the solver, including when restarting.
#
# This small example uses the solver's default direct linear solver. The
# timestep resolves the Kelvin time with 60 steps. Implicit time integration
# provides stability; accuracy still needs checking by reducing `dt`.

solver = BurgersSolver(z, approximation, dt=dt, bcs=bcs)

# For comparison, the existing Maxwell interface needs displacement and one
# internal-variable tensor. Both models see exactly the same boundary load.

u_maxwell = Function(V, name="Maxwell displacement")
m_maxwell = Function(S)
maxwell_approximation = MaxwellApproximation(
    bulk_modulus=5, density=1, shear_modulus=mu, viscosity=eta, B_mu=0,
)
maxwell_solver = InternalVariableSolver(
    u_maxwell, maxwell_approximation, internal_variables=m_maxwell,
    dt=dt, bcs=bcs,
)

# Solving and recording the strain contributions
# -----------------------------------------------
#
# We measure $e_{xx}$, the horizontal extension. Positive values mean extension.
# Since this field is uniform, its domain average also gives its value at any
# point in the block. Elastic strain is the total minus the two internal
# strains; it is not an additional solution field.

# +
total_strain = approximation.deviatoric_strain(u)[0, 0]
elastic_strain = total_strain - transient[0, 0] - permanent[0, 0]
maxwell_strain = maxwell_approximation.deviatoric_strain(u_maxwell)[0, 0]
area = assemble(Constant(1) * dx(domain=mesh))
history = []

for step in range(round(end / float(dt)) + 1):
    time = step * float(dt)
    load_fraction = np.interp(time, [0, rise, unload_start, unload_end, end], [0, 1, 1, 0, 0])
    traction.assign(stress_amplitude * load_fraction)
    if step > 0:
        solver.solve()
        maxwell_solver.solve()
    strains = [assemble(field * dx) / area for field in
               (maxwell_strain, total_strain, elastic_strain, transient[0, 0], permanent[0, 0])]
    history.append([time, float(traction), *strains])

history = np.asarray(history)
# -

# Reading the results
# --------------------
#
# The top panel shows the stress history. The middle panel compares total
# strain; the bottom separates the Burgers contributions. All panels start
# at time zero and use a linear time axis. The curves report signed strain,
# rather than percentages of deformation.

if mesh.comm.rank == 0:
    np.savetxt("strains.csv", history, delimiter=",",
               header="time,stress,maxwell,total,elastic,transient,permanent", comments="")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    times = history[:, 0]
    axes[0].plot(times, history[:, 1] / stress_amplitude, color="black")
    axes[0].set_ylabel("Applied stress / peak stress")
    axes[1].plot(times, history[:, 2] * 1e6, "--", label="Maxwell")
    axes[1].plot(times, history[:, 3] * 1e6, label="Burgers")
    axes[1].set_ylabel("Total strain (microstrain)")
    for column, label in ((4, "Elastic"), (5, "Recoverable transient"), (6, "Permanent creep")):
        axes[2].plot(times, history[:, column] * 1e6, label=label)
    axes[2].set(ylabel="Burgers strain (microstrain)", xlabel=r"Time / $(\eta_0/\mu_0)$")
    for ax in axes:
        ax.axvline(unload_end, color="0.7", linewidth=1)
        ax.set_xlim(0, end)
        ax.grid(alpha=0.25)
    axes[1].legend()
    axes[2].legend()
    fig.savefig("creep.png", dpi=150)
    plt.show()

# During the hold, both bodies accumulate permanent strain at the same rate,
# $\sigma/(2\eta)$. Burgers also accumulates Kelvin strain. Once this transient
# response saturates, its total creep rate approaches the Maxwell rate.
#
# Elastic strain disappears as the load is removed. Maxwell then retains its
# permanent strain, while Burgers continues recovering as Kelvin strain decays.
# The two curves approach the **same permanent strain**, because the linear
# dashpots experienced the same stress history and have the same viscosity.
#
# This retained strain is a property of the zero-stress material experiment.
# In GIA, gravity and spatial stress redistribution can drive further creep
# after the external ice load has vanished. Zero applied load does not generally
# mean zero internal stress, or that surface recovery is complete.

# Making permanent creep nonlinear
# ---------------------------------
#
# Change `exponent` to `3` and rerun from the beginning. For $n>1$, with zero
# background stress, the permanent-creep law is
#
# $$
# \dot{\boldsymbol e}_v = \frac{\boldsymbol s}{2\eta}
#    \left[1+\left(\frac{\sigma_{\rm II}}{\sigma_*}\right)^{n-1}\right],
# \qquad \sigma_{\rm II}=\sqrt{\tfrac12\boldsymbol s:\boldsymbol s}.
# $$
#
# Here $\sigma_{\rm II}=|\sigma|$ and `transition_stress` is $\sigma_*$.
# The elastic spring and Kelvin law stay linear; only permanent creep changes.
# At the chosen peak stress $\sigma_{\rm II}=\sigma_*$, its rate is twice the
# linear rate. `viscosity` is now the low-stress reference, so the nonlinear
# Burgers body and linear Maxwell body have different steady viscosities at
# finite stress. Their residual strains will differ. `exponent=1` explicitly
# selects the linear law used in the main comparison.
#
# Compare the Kelvin curves: they agree here because stress is prescribed.
# Under a spatially varying ice load, nonlinear creep also changes the stress
# distribution and can therefore change the Kelvin response. Applying a
# nonlinear law to Maxwell relaxation modes is not generally equivalent to
# making only the physical permanent-creep dashpot nonlinear.

# + [markdown] tags=["exercise"]
# Exercises
# ----------
#
# 1. Halve `eta_t` and rerun. The Kelvin response becomes faster, while its
#    constant-stress limiting strain stays the same. Change `mu_t` instead:
#    both that limiting strain and the retardation time change.
# 2. Increase `mu_t` substantially. Kelvin compliance becomes small and the
#    Burgers displacement approaches the Maxwell displacement. Reduce `dt`
#    as needed to resolve any shorter transient you want to inspect.
# 3. With `exponent=3`, double `stress_amplitude` while keeping
#    `transition_stress` fixed. Does permanent strain double? Compare this
#    with the linear case, and check which contributions still scale linearly.
# 4. Halve `dt` and compare the loading and recovery curves at matching times.
#    Backward Euler is first-order accurate: numerical errors should decrease
#    as the timestep is reduced. This uniform experiment has no spatial
#    discretisation error; a GIA calculation also needs mesh and domain checks.
