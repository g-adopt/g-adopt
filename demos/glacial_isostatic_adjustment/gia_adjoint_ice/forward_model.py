# This script defines the forward viscoelastic loading problem. Adjoint GIA modelling and ice inversions in an annulus

from gadopt import *
from gadopt.utility import step_func, vertical_component
import pyvista as pv
import matplotlib.pyplot as plt




def forward_model(mesh, normalised_ice_thickness):
    def initialise_background_field(field, background_values, X, vertical_tanh_width=40e3):
        profile = background_values[0]
        sharpness = 1 / vertical_tanh_width
        depth = sqrt(X[0]**2 + X[1]**2)-radius_values[0]
        for i in range(1, len(background_values)):
            centre = radius_values[i] - radius_values[0]
            mag = background_values[i] - background_values[i-1]
            profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

        field.interpolate(profile)


    def bivariate_gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalised_area=False):
        arg = ((x-mu_x)/sigma_x)**2 - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y) + ((y-mu_y)/sigma_y)**2
        numerator = exp(-1/(2*(1-rho**2))*arg)
        if normalised_area:
            denominator = 2*pi*sigma_x*sigma_y*(1-rho**2)**0.5
        else:
            denominator = 1
        return numerator / denominator


    def setup_heterogenous_viscosity(viscosity):
        heterogenous_viscosity_field = Function(viscosity.function_space(), name='viscosity')
        antarctica_x, antarctica_y = -2e6, -5.5e6

        low_viscosity_antarctica = bivariate_gaussian(X[0], X[1], antarctica_x, antarctica_y, 1.5e6, 0.5e6, -0.4)
        heterogenous_viscosity_field.interpolate(-3*low_viscosity_antarctica + viscosity * (1-low_viscosity_antarctica))

        llsvp1_x, llsvp1_y = 3.5e6, 0
        llsvp1 = bivariate_gaussian(X[0], X[1], llsvp1_x, llsvp1_y, 0.75e6, 1e6, 0)
        heterogenous_viscosity_field.interpolate(-3*llsvp1 + heterogenous_viscosity_field * (1-llsvp1))

        llsvp2_x, llsvp2_y = -3.5e6, 0
        llsvp2 = bivariate_gaussian(X[0], X[1], llsvp2_x, llsvp2_y, 0.75e6, 1e6, 0)
        heterogenous_viscosity_field.interpolate(-3*llsvp2 + heterogenous_viscosity_field * (1-llsvp2))

        slab_x, slab_y = 3e6, 4.5e6
        slab = bivariate_gaussian(X[0], X[1], slab_x, slab_y, 0.7e6, 0.35e6, 0.7)
        heterogenous_viscosity_field.interpolate(-1*slab + heterogenous_viscosity_field * (1-slab))

        high_viscosity_craton_x, high_viscosity_craton_y = 0, 6.2e6
        high_viscosity_craton = bivariate_gaussian(X[0], X[1], high_viscosity_craton_x, high_viscosity_craton_y, 1.5e6, 0.5e6, 0.2)
        heterogenous_viscosity_field.interpolate(-1*high_viscosity_craton + heterogenous_viscosity_field * (1-high_viscosity_craton))

        return heterogenous_viscosity_field
    # Let's use the same triangular mesh from the *gia_2d_cylindrical* demo with a target resolution of 200km near the surface of the Earth coarsening to 500 km in the interior.

    # Set up geometry:
    D = 2891e3  # Depth of domain in m
    bottom_id, top_id = 1, 2
    mesh.cartesian = False

    # We next set up the function spaces, and specify functions to hold our solutions. As our mesh is now made up of triangles instead of quadrilaterals, the syntax for defining our finite elements changes slighty. We need to specify *Continuous Galerkin* elements, i.e. replace `Q` with `CG` instead.

    # +
    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    TP1 = TensorFunctionSpace(mesh, "DG", 2)  # (Discontinuous) Stress tensor function space (tensor)
    R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    z = Function(Z)  # A field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p
    z.subfunctions[0].rename("Incremental Displacement")
    z.subfunctions[1].rename("Pressure")

    displacement = Function(V, name="displacement").assign(0)
    stress_old = Function(TP1, name="stress_old").assign(0)
    # -

    # We can output function space information, for example the number of degrees
    # of freedom (DOF).

    # Output function space information:
    log("Number of Incremental Displacement DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

    X = SpatialCoordinate(mesh)

    # Now we can set up the background profiles for the material properties. In this case the density and shear modulus vary in the vertical direction. We will approximate the series of layers using a smooth tanh function with a width of 40 km.

    # +
    # layer properties from spada et al 2011
    radius_values = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
    density_values = [3037, 3438, 3871, 4978]
    shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
    viscosity_values = [2, -2, -2, -1.698970004]  # viscosity = 1e23 * 10**viscosity_values
    # N.b. that we have modified the viscosity of the Lithosphere viscosity from
    # Spada et al 2011 because we are using coarse grid resolution


    density = Function(W, name="density")
    initialise_background_field(density, density_values, X)

    shear_modulus = Function(W, name="shear modulus")
    initialise_background_field(shear_modulus, shear_modulus_values, X)


    # Next let's initialise the viscosity field. In this tutorial we are going to make things a bit more interesting by using a laterally varying viscosity field. We'll put some regions of low viscosity near the South Pole (inspired by West Antarctica) as well as in the lower mantle. We've also put some relatively higher patches of mantle in the northern hemisphere to represent a downgoing slab.


    normalised_viscosity = Function(W, name="Normalised viscosity")
    initialise_background_field(normalised_viscosity, viscosity_values, X)
    normalised_viscosity = setup_heterogenous_viscosity(normalised_viscosity)

    viscosity = Function(normalised_viscosity, name="viscosity").interpolate(1e23*10**normalised_viscosity)


    # Now let's setup the ice load. For this tutorial we will have two synthetic ice sheets. Let's put one a larger one over the South Pole, with a total horizontal extent of 40 $^\circ$ and a maximum thickness of 2 km, and a smaller one offset from the North Pole with a width of 20 $^\circ$ and a maximum thickness of 1 km. To simplify things let's keep the ice load fixed in time.

    # +
    rho_ice = 931
    g = 9.8125

    Hice = 1000
    year_in_seconds = Constant(3600 * 24 * 365.25)
    ice_load = Function(W).interpolate(rho_ice * g * Hice * normalised_ice_thickness)

    # -


    # Let's setup the timestepping parameters with a timestep of 200 years and an output frequency of 1000 years.

    # +
    # Timestepping parameters
    Tstart = 0
    time = Function(R).assign(Tstart * year_in_seconds)

    dt_years = 1e3
    dt = Constant(dt_years * year_in_seconds)
    Tend_years = 10e3
    Tend = Constant(Tend_years * year_in_seconds)
    dt_out_years = 1e3
    dt_out = Constant(dt_out_years * year_in_seconds)

    max_timesteps = round((Tend - Tstart * year_in_seconds) / dt)
    log("max timesteps: ", max_timesteps)

    dump_period = round(dt_out / dt)
    log("dump_period:", dump_period)
    log(f"dt: {float(dt / year_in_seconds)} years")
    log(f"Simulation start time: {Tstart} years")

    do_write = True
    # -

    # We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
    #
    # For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as indicating this is a free surface.
    #
    # The `delta_rho_fs` option accounts for the density contrast across the free surface whether there is ice or air above a particular region of the mantle.

    # Setup boundary conditions
    stokes_bcs = {top_id: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - rho_ice*normalised_ice_thickness}},
                  bottom_id: {'un': 0}
                  }


    # We also need to specify a G-ADOPT approximation which sets up the various parameters and fields needed for the viscoelastic loading problem.


    approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, viscosity, g=g)

    # We finally come to solving the variational problem, with solver
    # objects for the Stokes system created. We pass in the solution fields `z` and various fields needed for the solve along with the approximation, timestep and boundary conditions.
    #

    stokes_solver = ViscoelasticStokesSolver(z, stress_old, displacement, approximation,
                                             dt, bcs=stokes_bcs)

    # We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

    # +
    if do_write:
        # Create a velocity function for plotting
        velocity = Function(V, name="velocity")
        velocity.interpolate(z.subfunctions[0]/dt)

        # Create output file
        output_file = VTKFile("output.pvd")
        output_file.write(*z.subfunctions, displacement, velocity)

    plog = ParameterLog("params.log", mesh)
    plog.log_str(
        "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
    )

    checkpoint_filename = "viscoelastic_loading-chk.h5"

    gd = GeodynamicalDiagnostics(z, density, bottom_id, top_id)

    # Initialise a (scalar!) function for logging vertical displacement
    U = FunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (scalar)
    vertical_displacement = Function(U, name="Vertical displacement")
    # -

    # Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter. At each step we call `solve` to calculate the incremental displacement and pressure fields. This will update the displacement at the surface and stress values accounting for the time dependent Maxwell consitutive equation.

    for timestep in range(max_timesteps+1):

        stokes_solver.solve()

        time.assign(time+dt)

        if timestep % dump_period == 0:
            # First output step is after one solve i.e. roughly elastic displacement
            # provided dt < maxwell time.
            log("timestep", timestep)

            if do_write:
                velocity.interpolate(z.subfunctions[0]/dt)
                output_file.write(*z.subfunctions, displacement, velocity)

            with CheckpointFile(checkpoint_filename, "w") as checkpoint:
                checkpoint.save_function(z, name="Stokes")
                checkpoint.save_function(displacement, name="Displacement")
                checkpoint.save_function(stress_old, name="Deviatoric stress")

        vertical_displacement.interpolate(vertical_component(displacement))

        # Log diagnostics:
        plog.log_str(
            f"{timestep} {float(time)} {float(dt)} "
            f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(top_id)} "
            f"{vertical_displacement.dat.data.min()} {vertical_displacement.dat.data.max()}"
        )

    return z.subfunctions[0], displacement

