import os
import matplotlib.pyplot as plt
from firedrake import *
from mpi4py import MPI

output_path = os.getcwd()

name = f"submesh-withurfDG-2d-internalvariable-dispvel-1dvisc"

full_mesh = Mesh("unstructured_annulus_refined_surface_gravity.msh")
DG0_full = FunctionSpace(full_mesh, "DG", 0)
X = SpatialCoordinate(full_mesh)
r = sqrt(X[0] ** 2 + X[1] ** 2)
# Heaviside step function in mantle
radius_grav = 31855e3
radius_earth = 6371e3
grav_mesh_depth = radius_grav - radius_earth
radius_core = 3480e3
depth = radius_earth - radius_core
Rg_tilde = radius_grav / depth
Re_tilde = radius_earth / depth
Rc_tilde = radius_core / depth
F_mantle = Function(DG0_full).interpolate(conditional(And(r >= Rc_tilde, r <= Re_tilde), 1, 0))
F_rest = Function(DG0_full).interpolate(conditional(r >= 0, 1, 0))
mesh = RelabeledMesh(full_mesh, [F_mantle, F_rest], [98, 99])
subm = Submesh(mesh, 2, 98)

Rg = 31855e3
Re = 6371e3
grav_mesh_depth = Rg - Re
Rc = 3480e3
D = Re - Rc


Rc_id = 1
Re_id = 2
Rg_id = 3

PETSc.Sys.Print("Area of annulus: ", assemble(Constant(1) * dx(domain=subm)))
PETSc.Sys.Print("Length of top: ", assemble(Constant(1) * ds(Re_id, domain=subm)))
PETSc.Sys.Print("Length of bottom: ", assemble(Constant(1) * ds(Rc_id, domain=subm)))

PETSc.Sys.Print("Area of annulus gravity mesh: ", assemble(Constant(1) * dx(domain=mesh)))
PETSc.Sys.Print("Length of top gravity mesh: ", assemble(Constant(1) * ds(Rg_id, domain=mesh)))
PETSc.Sys.Print("Length of bottom gravity mesh: ", assemble(Constant(1) * ds(Rc_id, domain=mesh)))

# Set up function spaces:
V = VectorFunctionSpace(subm, "CG", 2)  # Displacement function space (vector)
V_grav = FunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
# V_grav = FunctionSpace(mesh, "CG", 2)  # Displacement function space (vector)
S = TensorFunctionSpace(subm, "DG", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG0 = FunctionSpace(subm, "DG", 0)  # (Discontinuous) Stress tensor function space (tensor)
DG1 = FunctionSpace(subm, "DG", 1)  # (Discontinuous) Stress tensor function space (tensor)
DG1_full = FunctionSpace(mesh, "DG", 1)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(subm, "R", 0)  # Real function space (for constants)

Z = MixedFunctionSpace([V, S, V_grav])  # Mixed function space.
z = Function(Z)  # A field over the mixed function space Z.
# Function to store the solutions:
u, m, phi = split(z)  # Returns symbolic UFL expression for u and m
# Next rename for output:
z.subfunctions[0].rename("Displacement")
z.subfunctions[1].rename("Internal variable")
z.subfunctions[2].rename("Gravitational Potential")

X = SpatialCoordinate(subm)
X_grav = SpatialCoordinate(mesh)

density_values = [3037]  # , 3438, 3871, 4978]
shear_modulus_values = ([0.50605e11],)  # 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e21]  # [1e25, 1e21, 1e21, 2e21]

density_scale = 4500
shear_modulus_scale = 1e11
viscosity_scale = 1e21

density_values_tilde = np.array(density_values) / density_scale
shear_modulus_values_tilde = np.array(shear_modulus_values) / shear_modulus_scale
viscosity_values_tilde = np.array(viscosity_values) / viscosity_scale

density = Function(DG0, name="density").assign(1)
# initialise_background_field(density, density_values_tilde)

shear_modulus = Function(DG0, name="shear modulus").assign(1)
# initialise_background_field(shear_modulus, shear_modulus_values_tilde)

# if Pseudo incompressible set bulk modulus to a constant...
# Otherwise use same jumps from shear modulus multiplied by a factor

bulk_modulus = Function(DG0, name="bulk modulus").assign(1)
# initialise_background_field(bulk_modulus, shear_modulus_values_tilde)

background_viscosity = Function(DG1, name="background viscosity").assign(1)
# initialise_background_field(background_viscosity, viscosity_values_tilde)

grad_phi = grad(phi)

bulk_shear_ratio = Constant(1.94)


# Defined lateral viscosity regions
def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
    arg = (
        ((x - mu_x) / sigma_x) ** 2
        - 2 * rho * ((x - mu_x) / sigma_x) * ((y - mu_y) / sigma_y)
        + ((y - mu_y) / sigma_y) ** 2
    )
    numerator = exp(-1 / (2 * (1 - rho**2)) * arg)
    if normalised_area:
        denominator = 2 * pi * sigma_x * sigma_y * (1 - rho**2) ** 0.5
    else:
        denominator = 1
    return numerator / denominator


def setup_heterogenous_viscosity(viscosity):
    heterogenous_viscosity_field = Function(viscosity.function_space(), name="viscosity")
    antarctica_x, antarctica_y = -2e6 / D, -5.5e6 / D

    low_visc = 1e20 / viscosity_scale
    high_visc = 1e22 / viscosity_scale

    low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6 / D, 0.5e6 / D, -0.4)
    heterogenous_viscosity_field.interpolate(
        low_visc * low_viscosity_antarctica + viscosity * (1 - low_viscosity_antarctica)
    )

    llsvp1_x, llsvp1_y = 3.5e6 / D, 0
    llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6 / D, 1e6 / D, 0)
    heterogenous_viscosity_field.interpolate(low_visc * llsvp1 + heterogenous_viscosity_field * (1 - llsvp1))

    llsvp2_x, llsvp2_y = -3.5e6 / D, 0
    llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6 / D, 1e6 / D, 0)
    heterogenous_viscosity_field.interpolate(low_visc * llsvp2 + heterogenous_viscosity_field * (1 - llsvp2))

    slab_x, slab_y = 3e6 / D, 4.5e6 / D
    slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6 / D, 0.35e6 / D, 0.7)
    heterogenous_viscosity_field.interpolate(high_visc * slab + heterogenous_viscosity_field * (1 - slab))

    high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6 / D
    high_viscosity_craton = bivariate_gaussian(
        X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6 / D, 0.5e6 / D, 0.2
    )
    heterogenous_viscosity_field.interpolate(
        high_visc * high_viscosity_craton + heterogenous_viscosity_field * (1 - high_viscosity_craton)
    )

    return heterogenous_viscosity_field


viscosity = setup_heterogenous_viscosity(background_viscosity)

year_in_seconds = 8.64e4 * 365.25
characteristic_maxwell_time = viscosity_scale / shear_modulus_scale

Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds / characteristic_maxwell_time)

dt_years = 100
dt = Constant(dt_years * year_in_seconds / characteristic_maxwell_time)
Tend_years = 10e3
Tend = Constant(Tend_years * year_in_seconds / characteristic_maxwell_time)
dt_out_years = dt_years
dt_out = Constant(dt_out_years * year_in_seconds / characteristic_maxwell_time)

max_timesteps = round((Tend - Tstart * year_in_seconds / characteristic_maxwell_time) / dt)
PETSc.Sys.Print("max timesteps: ", max_timesteps)

output_frequency = round(dt_out / dt)
PETSc.Sys.Print("output_frequency:", output_frequency)
PETSc.Sys.Print(f"dt: {float(dt)} maxwell times")
PETSc.Sys.Print(f"dt: {float(dt * characteristic_maxwell_time / year_in_seconds)} years")
PETSc.Sys.Print(f"Simulation start time: {Tstart} maxwell times")
PETSc.Sys.Print(f"Simulation end time: {Tend} maxwell times")
PETSc.Sys.Print(f"Simulation end time: {float(Tend * characteristic_maxwell_time / year_in_seconds)} years")

rho_ice = 931 / density_scale
g = 9.815
Vi = Constant(density_scale * D * g / shear_modulus_scale)
PETSc.Sys.Print("Ratio of buoyancy/shear = rho g D / mu = ", float(Vi))
Hice1 = 1000 / D
Hice2 = 2000 / D
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2 * pi / 360) * 10  # Disk half width in radians
disc_halfwidth2 = (2 * pi / 360) * 20  # Disk half width in radians
surface_dx_smooth = 200 * 1e3
# ncells_smooth = 2*pi*radius_values[0] / surface_dx_smooth
ncells_smooth = 20
surface_resolution_radians_smooth = 2 * pi / ncells_smooth
colatitude = atan2(X[0], X[1])
disc1_centre = (2 * pi / 360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5 * (1 - tanh((abs(colatitude - disc1_centre) - disc_halfwidth1) / (2 * surface_resolution_radians_smooth)))
disc2 = 0.5 * (
    1 - tanh((abs(abs(colatitude) - disc2_centre) - disc_halfwidth2) / (2 * surface_resolution_radians_smooth))
)

ice_load = Vi * rho_ice * (Hice1 * disc1 + Hice2 * disc2)

stokes_bcs = {
    Rc_id: {"un": 0},
    Re_id: {"normal_stress": ice_load, "free_surface": {}},
}

direct_stokes_solver_parameters = {
    "snes_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_converged_reason": None,
}

iterative_parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-7,
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "symmetric_multiplicative",
    "fieldsplit_0_ksp_converged_reason": None,
    "fieldsplit_0_ksp_monitor": None,
    "fieldsplit_0_ksp_type": "gmres",
    "fieldsplit_0_pc_type": "python",
    #                        "fieldsplit_0_pc_python_type": "gadopt.SPDAssembledPC",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "gamg",
    "fieldsplit_0_assembled_mg_levels_pc_type": "sor",
    "fieldsplit_0_ksp_rtol": 1e-5,
    "fieldsplit_0_assembled_pc_gamg_threshold": 0.01,
    "fieldsplit_0_assembled_pc_gamg_square_graph": 100,
    "fieldsplit_0_assembled_pc_gamg_coarse_eq_limit": 1000,
    "fieldsplit_0_assembled_pc_gamg_mis_k_minimum_degree_ordering": True,
    "fieldsplit_1_ksp_converged_reason": None,
    "fieldsplit_1_ksp_monitor": None,
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_1_assembled_pc_type": "sor",
    "fieldsplit_1_ksp_rtol": 1e-5,
    # "fieldsplit_2_ksp_converged_reason": None,
    # "fieldsplit_2_ksp_monitor": None,
    # "fieldsplit_2_ksp_type": "preonly",
    "fieldsplit_2_ksp_type": "cg",
    "fieldsplit_2_ksp_converged_reason": None,
    "fieldsplit_2_ksp_monitor": None,
    # "fieldsplit_2_pc_type": "lu",
    "fieldsplit_2_pc_type": "python",
    "fieldsplit_2_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_2_assembled_pc_type": "lu",
    "fieldsplit_2_ksp_rtol": 1e-5,
    # "fieldsplit_2_assembled_pc_type": "lu",
    # "fieldsplit_2_pc_factor_mat_solver_type": "mumps",
    # "fieldsplit_2_snes_type": "newtonls",
    # "fieldsplit_2_snes_linesearch_type": "l2",
    # "fieldsplit_2_snes_max_it": 100,
    # "fieldsplit_2_snes_atol": 1e-10,
    # "fieldsplit_2_snes_rtol": 1e-5,
    # "fieldsplit_2_snes_converged_reason": None,
}

# Nullspaces and near-nullspaces:
x_rotV = Function(V).interpolate(as_vector((-X[1], X[0])))
# V_nullspace = VectorSpaceBasis([x_rotV])
# V_nullspace.orthonormalize()
### rotational = False
V_nullspace = Z.subspaces[0]
p_nullspace = VectorSpaceBasis(constant=True)  # Constant nullspace for pressure n
Z_nullspace = MixedVectorSpaceBasis(Z, [V_nullspace, p_nullspace, Z.sub(2)])  # Setting mixed nullspace

# Generating near_nullspaces for GAMG:
nns_x = Function(V).interpolate(Constant([1.0, 0.0]))
nns_y = Function(V).interpolate(Constant([0.0, 1.0]))
V_near_nullspace = VectorSpaceBasis([nns_x, nns_y, x_rotV])
V_near_nullspace.orthonormalize()
Z_near_nullspace = MixedVectorSpaceBasis(Z, [V_near_nullspace, Z.sub(1), Z.sub(2)])

### Measures
dx_all = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", domain=subm),))
dx_m = Measure("dx", domain=subm, intersect_measures=(Measure("dx", domain=mesh),))
dS_m = Measure("dS", domain=subm, intersect_measures=(Measure("ds", domain=mesh),))
dS_full = Measure("dS", domain=mesh, intersect_measures=(Measure("dS", domain=subm),))
ds_mt = Measure("ds", domain=subm, intersect_measures=(Measure("dS", domain=mesh),))
ds_mb = Measure("ds", domain=subm, intersect_measures=(Measure("ds", domain=mesh),))


def upward_normal(mesh):
    X = SpatialCoordinate(mesh)
    r = sqrt(sum([x**2 for x in X]))
    return X / r

def vc(u):
    mesh = u.function_space().mesh().unique()
    n = upward_normal(mesh)
    return sum([n_i * u_i for n_i, u_i in zip(n, u)])


def stress(u, m_list):
    dim = len(u)
    div_u = div(u) * Identity(dim)
    e = sym(grad(u))
    d = e - 1 / 3 * tr(e) * Identity(dim)
    stress = bulk_shear_ratio * bulk_modulus * div_u + 2 * len(m_list) * shear_modulus * d
    for m in m_list:
        stress -= 2 * shear_modulus * m
    return stress


def deviatoric_strain(u):
    dim = len(u)
    e = sym(grad(u))
    return e - 1 / 3 * tr(e) * Identity(dim)


def buoyancy(displacement):
    # Buoyancy term rho1, coming from linearisation and integrating the continuity equation w.r.t time
    # accounts for advection of density in the absence of an evolution equation for temperature
    return -Vi * g * -inner(displacement, grad(density))


mu = shear_modulus
stress = stress(u, [m])
source_mantle = buoyancy(u) * upward_normal(subm)
g = 9.185  # need to remove this from Vi
source_grav = -Vi * density / g * grad_phi
source = source_mantle + source_grav
strain = deviatoric_strain(u)
###u_r = vc(u)
u_r = sum([n_i * u_i for n_i, u_i in zip(upward_normal(subm), u)])
maxwell_time = viscosity / shear_modulus
scaling_factors = [1, -0.5]
n = FacetNormal(subm)
n_full = FacetNormal(mesh)
test_funcs = TestFunctions(Z)


def hydrostatic_prestress_advection(u_r):
    return Vi * density * u_r


### Boundary conditions
weak_bcs = {1: {"un": 0}, 2: {"normal_stress": ice_load + hydrostatic_prestress_advection(u_r)}}
free_surface_dict = {2: {}}


def advection_hydrostatic_prestress_term(trial):
    # Advection of background hydrostatic pressure used in linearised
    # GIA simulations where
    rho0 = density
    g = Constant(1)
    u_r = sum([n_i * u_i for n_i, u_i in zip(upward_normal(subm), trial)])
    # Only include jump term for discontinuous density spaces?
    # if is_continuous(rho0.function_space()):
    #    F = 0
    # else:
    # change ds for extruded mesh? maybe not a good idea?
    #        if type(rho0.function_space()._mesh) is ExtrudedMeshTopology:
    #            dS = dS_h
    F = Vi("+") * jump(rho0) * u_r("+") * g("+") * dot(test_funcs[0]("+"), n("+")) * dS_m(degree=6)
    return -F


def momentum_source_term(trial):
    # F = -dot(test_funcs[0], source_mantle) * dx_m(degree=6) + dot(test_funcs[0], source_grav) * dx_m(degree=6)
    F = -dot(test_funcs[0], source) * dx_m(degree=6)

    return -F


def viscosity_term(trial):
    r"""Viscosity term $-nabla * (mu nabla u)$ in the momentum equation.

    Using the symmetric interior penalty method (Epshteyn & Rivière, 2007), the weak
    form becomes

    $$
    {:( -int_Omega nabla * (mu grad u) phi dx , = , int_Omega mu (grad phi) * (grad u) dx ),
      ( , - , int_(cc"I" uu cc"I"_v) "jump"(phi bb n) * "avg"(mu grad u) dS
          -   int_(cc"I" uu cc"I"_v) "jump"(u bb n) * "avg"(mu grad phi) dS ),
      ( , + , int_(cc"I" uu cc"I"_v) sigma "avg"(mu) "jump"(u bb n) * "jump"(phi bb n) dS )
    :}
    $$

    where σ is a penalty parameter.

    Epshteyn, Y., & Rivière, B. (2007).
    Estimation of penalty parameters for symmetric interior penalty Galerkin methods.
    Journal of Computational and Applied Mathematics, 206(2), 843-872.
    """
    dim = 2
    identity = Identity(dim)
    compressible_stress = True

    ### stress = stress
    F = inner(nabla_grad(test_funcs[0]), stress) * dx_m(degree=6)

    sigma = 36.0
    sigma *= FacetArea(subm) / CellVolume(subm)

    # NOTE: Unspecified boundaries result in free stress (i.e. free in all directions).
    # NOTE: "un" can be combined with "stress" provided the stress component is
    # tangential (e.g. no normal flow with wind)
    for bc_id, bc in weak_bcs.items():
        if "u" in bc and any(bc_type in bc for bc_type in ["stress", "un"]):
            raise ValueError('"stress" or "un" cannot be specified if "u" is already given.')
        if "normal_stress" in bc and any(bc_type in bc for bc_type in ["u", "un"]):
            raise ValueError('"u" or "un" cannot be specified if "normal_stress" is already given.')

        if "un" in bc:
            trial_tensor_jump = outer(n, n) * (dot(n, trial) - bc["un"])
            if compressible_stress:
                trial_tensor_jump -= 2 / 3 * identity * (dot(n, trial) - bc["un"])
            trial_tensor_jump += transpose(trial_tensor_jump)
            # Terms below are similar to the above terms for the DG dS integrals.
            F += 2 * sigma * inner(outer(n, test_funcs[0]), mu * trial_tensor_jump) * ds_mb(bc_id, degree=6)
            F -= inner(mu * nabla_grad(test_funcs[0]), trial_tensor_jump) * ds_mb(bc_id, degree=6)
            # We only keep the normal part of stress; the tangential part is assumed to
            # be zero stress (i.e. free slip) or prescribed via "stress".
            F -= dot(n, test_funcs[0]) * dot(n, dot(stress, n)) * ds_mb(bc_id, degree=6)

            # Hack in bulk compressibility part of un bc - Fix in gadopt...
            # is this physics or discretisation?
            trial_tensor_jump = identity * (dot(n, trial) - bc["un"])
            trial_tensor_jump += transpose(trial_tensor_jump)
            bulk = bulk_modulus * bulk_shear_ratio
            # Terms below are similar to the above terms for the DG dS integrals.
            F += 2 * sigma * inner(outer(n, test_funcs[0]), bulk * trial_tensor_jump) * ds_mb(bc_id, degree=6)
            F -= inner(bulk * nabla_grad(test_funcs[0]), trial_tensor_jump) * ds_mb(bc_id, degree=6)
            # We only keep the normal part of stress; the tangential part is assumed to
            # be zero stress (i.e. free slip) or prescribed via "stress".
            F -= dot(n, test_funcs[0]) * dot(n, dot(stress, n)) * ds_mb(bc_id, degree=6)

        if "stress" in bc:  # a momentum flux, a.k.a. "force"
            # Here we need only the third term because we assume jump_u = 0
            # (u_ext = trial) and stress = n . (mu . stress_tensor).
            F -= dot(test_funcs[0], bc["stress"]) * ds_mb(bc_id, degree=6)

        if "normal_stress" in bc:
            F += dot(test_funcs[0], bc["normal_stress"] * n) * ds_mb(bc_id, degree=6)

    return -F


source_scalar = strain / maxwell_time
sink_coeff = 1 / maxwell_time


def source_term(trial):
    r"""Scalar source term `s_T`."""
    F = -inner(test_funcs[1], source_scalar) * dx_m(degree=6)

    return -F


def sink_term(trial):
    r"""Scalar sink term `\alpha_T T`."""
    # Implement sink term implicitly at current time step.
    F = inner(test_funcs[1], sink_coeff * trial) * dx_m(degree=6)

    return -F


def mass_term(trial):
    """UFL form for the mass term used in the time discretisation.

    Args:
        eq:
          G-ADOPT Equation.
        trial:
          Firedrake trial function.

    Returns:
        The UFL form associated with the mass term of the equation.

    """
    return inner(test_funcs[1], trial) * dx_m(degree=6)


residual_terms_internal_variable = [source_term, sink_term]
solution_old = Function(Z)
_, mold, _ = split(solution_old)

### Gravity boundary condition
grav_bcs = [DirichletBC(Z.sub(2), 0.0, Rg_id)]

##################
L_grav = -inner(grad(test_funcs[2]), grad(phi)) * dx_all(degree=6)  # - v * dot(grad(phi), FacetNormal(mesh)) * ds(4)
rho1_grav = -dot(u, grad(density)) - density * div(u)
R_grav = 4 * pi * rho1_grav * test_funcs[2] * dx_m(degree=6) + Constant(0) * test_funcs[2] * dx_all(degree=6)
F_grav = L_grav - R_grav
##################

F = (
    -(advection_hydrostatic_prestress_term(u) + momentum_source_term(u) + viscosity_term(u))
    - 0.5 * mass_term((m - mold) / dt)
    + 0.5 * (source_term(m) + sink_term(m))
) - F_grav

problem = NonlinearVariationalProblem(F, z, bcs=grav_bcs, J=None)
# problem = NonlinearVariationalProblem(F, z, J=None)
coupled_solver = NonlinearVariationalSolver(
    problem,
    solver_parameters=iterative_parameters,
    #nullspace=Z_nullspace,
    #near_nullspace=Z_near_nullspace,
    appctx={"mu": mu},
)

grav_output_file = VTKFile("grav_iterative.pvd")
output_file = VTKFile(f"out_iterative.pvd")
G = VectorFunctionSpace(mesh, "CG", 2)
g = Function(G,name="g")

vertical_displacement = Function(V.sub(1), name="radial displacement")
bc_displacement = DirichletBC(vertical_displacement.function_space(), 0, Re_id)

for timestep in range(1, max_timesteps + 1):
    # for timestep in range(1,3):
    # update time first so that ice load begins
    time.assign(time + dt)
    coupled_solver.solve()
    # solution_old.assign(z)
    vertical_displacement.interpolate(vc(z.subfunctions[0])*D)
    displacement_z_min = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].min(initial=0)
    displacement_min = vertical_displacement.comm.allreduce(displacement_z_min, MPI.MIN)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    PETSc.Sys.Print("Greatest (-ve) displacement", displacement_min)
    displacement_z_max = vertical_displacement.dat.data_ro_with_halos[bc_displacement.nodes].max(initial=0)
    displacement_max = vertical_displacement.comm.allreduce(displacement_z_max, MPI.MAX)  # Minimum displacement at surface (should be top left corner with greatest (-ve) deflection due to ice loading
    PETSc.Sys.Print("Greatest (+ve) displacement", displacement_max)

    solution_old.sub(0).assign(z.sub(0))
    solution_old.sub(1).assign(z.sub(1))
    solution_old.sub(2).assign(z.sub(2))

    g.interpolate(-grad(z.sub(2)))
    grav_output_file.write(z.sub(2),g)
    output_file.write(z.sub(0), z.sub(1), vertical_displacement)
