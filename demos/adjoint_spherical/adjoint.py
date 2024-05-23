from gadopt import *
from gadopt.inverse import *
from gadopt.gplates import GplatesFunction, pyGplatesConnector
import numpy as np
from firedrake.adjoint_utils import blocks
from pyadjoint import stop_annotating 
from pathlib import Path
from pykdtree.kdtree import KDTree
from pyshtools import SHCoeffs
from tfinterpy.idw import IDW

# Quadrature degree:
dx = dx(degree=6)

# Quadrature degree:
dx = dx(degree=6)

# Projection solver parameters for nullspaces:
iterative_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}

LinearSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
NonlinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearVariationalSolver.DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}
LinearSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
LinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters
NonlinearVariationalSolver.DEFAULT_KSP_PARAMETERS = iterative_solver_parameters

blocks.solving.Block.evaluate_adj = collect_garbage(blocks.solving.Block.evaluate_adj)
blocks.solving.Block.recompute = collect_garbage(blocks.solving.Block.recompute)

# timer decorator for fwd and derivative calls.
ReducedFunctional.__call__ = collect_garbage(
    timer_decorator(ReducedFunctional.__call__)
)
ReducedFunctional.derivative = collect_garbage(
    timer_decorator(ReducedFunctional.derivative)
)

# Set up geometry:
rmax = 2.208
rmin = 1.208


def __main__():
    my_taylor_test()


def conduct_inversion():
    Tic, reduced_functional = forward_problem()

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Tic.function_space(), name="Lower bound temperature")
    T_ub = Function(Tic.function_space(), name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)
    
    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
    
    # Establish a LinMore Optimiser
    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint",
    )
    # Add the callback function to optimisation
    optimiser.add_callback(callback)
    # run the optimisation
    optimiser.run()


def conduct_taylor_test():
    Tic, reduced_functional = forward_problem()
    log("Reduced Functional Repeat: ", reduced_functional([Tic]))
    Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
    Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
    _ = taylor_test(reduced_functional, Tic, Delta_temp)


@collect_garbage
def forward_problem():
    # Section:
    # Enable writing intermediary adjoint fields to disk
    enable_disk_checkpointing()


    base_path = Path(__file__).resolve()

    with CheckpointFile(str(base_path / "spherical_mesh.h5"), "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # Load mesh
    with CheckpointFile(str(base_path / "linear_LLNLG3G_SLB_Q5_smooth_2.0_101.h5"), "r") as fi:
        Tobs = fi.load_function(mesh, name="Tobs")  # reference tomography temperature
        Tave = fi.load_function(mesh, name="AverageTemperature")  # 1-D geotherm

    # Boundary markers to top and bottom
    bottom_id, top_id = "bottom", "top"

    # For accessing the coordinates
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # Initial Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.
    R = FunctionSpace(mesh, "R", 0) # Function space for constants 

    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    z = Function(Z)
    u, p = split(z)

    # Set up temperature field and initialise:
    Tic = Function(Q1, name="Tic")
    T = Function(Q, name="Temperature")
    mu = Function(W, name="Viscosity")
    assign_1d_profile(mu, str(base_path.parents[1] / "gplates_global/mu2_radial.rad"))

    T0 = Constant(0.091)  # Non-dimensional surface temperature
    Di = Constant(0.5)  # Dissipation number.
    H_int = Constant(10.0)  # Internal heating

    # Initial time step
    delta_t = Function(R, name="delta_t").assign(2.0e-6)
    Tic.assign(Tobs)

    # getting the filenames for the reconstruction model
    plate_reconstruction_files = get_plate_reconstruction_info()

    plate_reconstruction_model = pyGplatesConnector(
        rotation_filenames=plate_rec_files["rotation_filenames"],
        topology_filenames=plate_rec_files["topology_filenames"],
        nneighbours=4,
        nseeds=1e5,
        scaling_factor=1.0,
        oldest_age=1000,
        delta_t=1.0
    )

    # Top velocity boundary condition
    gplates_velocities = GplatesFunction(
        V,
        gplates_connector=plate_reconstruction_model,
        top_boundary_marker=top_id,
        name="GPlates_Velocity"
    )

    # Setup Equations Stokes related constants
    Ra = Constant(1.0e7)  # Rayleigh number
    Di = Constant(0.5)  # Dissipation number.

    # Compressible reference state:
    rho_0, alpha = 1.0, 1.0
    weight = r-rmin
    rhobar = Function(Q, name="CompRefDensity").interpolate(rho_0 * exp(((1.0 - weight) * Di) / alpha))
    Tbar = Function(Q, name="CompRefTemperature").interpolate(T0 * exp((1.0 - weight) * Di) - T0)
    alphabar = Function(Q, name="IsobaricThermalExpansivity").assign(1.0)
    cpbar = Function(Q, name="IsobaricSpecificHeatCapacity").assign(1.0)
    chibar = Function(Q, name="IsothermalBulkModulus").assign(1.0)

    # We use TALA for approximation
    approximation = TruncatedAnelasticLiquidApproximation(
        Ra, Di, rho=rhobar, Tbar=Tbar,
        alpha=alphabar, chi=chibar, cp=cpbar)

    # Section: Setting up nullspaces
    # Nullspaces for stokes contains only a constant nullspace for pressure, as the top boundary is
    # imposed. The nullspace is generate with closed=True(for pressure) and rotational=False
    # as there are no rotational nullspace for velocity.
    # .. note: For compressible formulations we only provide `transpose_nullspace`
    Z_nullspace = create_stokes_nullspace(
        Z, closed=True, rotational=False)
    # The near nullspaces gor gamg always include rotational and translational modes
    Z_near_nullspace = create_stokes_nullspace(
        Z, closed=False, rotational=True, translations=[0, 1, 2])

    # Section: Setting boundary conditions
    # Temperature boundary conditions (constant)
    # for the top and bottom boundaries
    temp_bcs = {
        bottom_id: {'T': 1.0 - (T0*exp(Di) - T0)},
        top_id: {'T': 0.0},
    }
    # Velocity boundary conditions
    stokes_bcs = {
        top_id: {'u': gplates_velocities},
        bottom_id: {'un': 0},
    }

    # Constructing Energy and Stokes solver
    energy_solver = EnergySolver(
        T, u, approximation, delta_t,
        ImplicitMidpoint, bcs=temp_bcs, su_advection=True)
    energy_solver.fields['source'] = rhobar * H_int
    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                                 cartesian=False, constant_jacobian=True,
                                 nullspace=Z_nullspace,
                                 transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # tweaking solver parameters
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-4
    stokes_solver.solver_parameters['snes_rtol'] = 1e-2
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-3
    stokes_solver.solver_parameters['fieldsplit_0']['assembled_pc_gamg_threshold'] = -1
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-2

    # non-dimensionalised time for present geologic day (0)
    ndtime_now = plate_reconstruction_model.age2ndtime(0.)

    # non-dimensionalised time for 10 Myrs ago
    time = plate_reconstruction_model.age2ndtime(25.)

    # Write output files in VTK format:
    u_, p_ = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")

    # Defining control
    control = Control(Tic)

    # project the initial condition from Q1 to Q2, and imposing
    # boundary conditions
    project(
        Tic,
        T,
        solver_parameters=iterative_solver_parameters,
        forward_kwargs={"solver_parameters": iterative_solver_parameters},
        adj_kwargs={"solver_parameters": iterative_solver_parameters},
        bcs=energy_solver.strong_bcs,
    )

    # timestep counter
    timestep_index = 0

    # Now perform the time loop:
    while time < ndtime_now:
        if timestep_index % 2 == 0:
            # Update surface velocities
            gplates_velocities.update_plate_reconstruction(time)

            # Solve Stokes sytem
            stokes_solver.solve()

        # Make sure we are not going past present day
        if ndtime_now - time < float(delta_t):
            log(f"delta_t is {delta_t.dat.data[0]}, and is changing to")
            delta_t.assign(ndtime_now - time)
            log(f"{delta_t.dat.data[0]}")
        # Make sure we are not going past present day

        # Temperature system:
        energy_solver.solve()

        # Updating time
        time += float(delta_t)
        timestep_index += 1

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - Tobs) ** 2 * dx)

    # Assembling the objective
    objective = (
        t_misfit 
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()
    return Tic, ReducedFunctional(objective, control)


def assign_1d_profile(q, one_d_filename):
    """
    Assign a one-dimensional profile to a Function `q` from a file.

    The function reads a one-dimensional radial viscosity profile from a file, broadcasts
    the read array to all processes, and then interpolates this
    array onto the function space of `q`.

    Args:
        q (firedrake.Function): The function onto which the 1D profile will be assigned.
        one_d_filename (str): The path to the file containing the 1D radial viscosity profile.

    Returns:
        None: This function does not return a value. It directly modifies the input function `q`.

    Note:
        - This function is designed to be run in parallel with MPI.
        - The input file should contain an array of viscosity values.
        - It assumes that the function space of `q` is defined on a radial mesh.
        - `rmax` and `rmin` should be defined before this function is called, representing
          the maximum and minimum radial bounds for the profile.
    """
    from firedrake.ufl_expr import extract_unique_domain
    from scipy.interpolate import interp1d
    
    with stop_annotating():
        # find the mesh
        mesh = extract_unique_domain(q)

        visc = None
        rshl = None
        # read the input file
        if mesh.comm.rank == 0:
            # The root process reads the file
            rshl, visc = np.loadtxt(one_d_filename, unpack=True, delimiter=",")

        # Broadcast the entire 'visc' array to all processes
        visc = mesh.comm.bcast(visc, root=0)
        # Similarly, broadcast 'rshl' if needed (assuming all processes need it)
        rshl = mesh.comm.bcast(rshl, root=0)

        element_family = q.function_space().ufl_element()
        X = Function(VectorFunctionSpace(mesh=mesh, family=element_family)).interpolate(SpatialCoordinate(mesh))
        rad = Function(q.function_space()).interpolate(sqrt(X**2))
        averager = LayerAveraging(mesh, cartesian=False)
        averager.extrapolate_layer_average(q, interp1d(rshl, visc, fill_value="extrapolate")(averager.get_layer_average(rad)))
    q.create_block_variable() 


def get_plate_reconstruction_info():
    plate_rec_files = {}

    base_path = Path(__file__).resolve()

    # rotation filenames
    plate_rec_files["rotation_filenames"] = [
        str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/optimisation/1000_0_rotfile_MantleOptimised.rot") 
    ]

    # topology filenames
    plate_rec_files["topology_filenames"] = [
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/250-0_plate_boundaries.gpml"),
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/410-250_plate_boundaries.gpml"),
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Convergence.gpml"),
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Divergence.gpml"),
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Topologies.gpml"),
            str(base_path.parents[1] / "gplates_global/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.2/1000-410-Transforms.gpml"),
    ]

    return plate_rec_files


def generate_reference_fields(ref_level=7, nlayers=64):
    base_path = Path(__file__).resolve()

    # mesh file
    mesh_path = base_path / "spherical_mesh.h5"

    # sph file
    sph_file_path = base_path / "NEW_TEMP_lin_LLNLG3G_SLB_Q5_mt512_smooth_2.0_101.sph"

    # generate and write out mesh
    generate_spherical_mesh(str(base_path / "spherical_mesh.h5"))

    # load the mesh
    with CheckpointFile(str(mesh_path), mode="r") as fi:
        mesh = fi.load_mesh("firedrake_default_extruded")

    # Reference temperature field
    Tobs = load_tomography_model(
        mesh,
        fi_name=)

    # Average of temperature field
    Taverage = Function(Tobs.function_space(), name="Taverage")

    # Calculate the layer average of the initial state
    averager = LayerAveraging(mesh, cartesian=False, quad_degree=6)
    averager.extrapolate_layer_average(
        Taverage, averager.get_layer_average(Tobs))

    # Write out the file
    with CheckpointFile(str(sph_file_path.with_suffix(".h5")), mode="w") as fi:
        fi.save_mesh(mesh)
        fi.save_function(Tobs, name="Tobs")
        fi.save_function(Taverage, name="AverageTemperature")
        fi.save_function(mu, name="Viscosity")

    # Output for visualisation
    output = File(str(sph_file_pathj.with_suffix(".pvd")))
    output.write(Tobs, Taverage, mu)


def viscosity_function(mesh):
    # Set up function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 2)

    # tomography based temperature
    viscosity = Function(Q, name="Viscosity")

    # radius and coordinates
    r = Function(V).interpolate(SpatialCoordinate(mesh))
    rad = Function(Q).interpolate(sqrt(SpatialCoordinate(mesh) ** 2))

    # knowing how many extrusion layers we have
    vnodes = nlayers * 2 + 1
    rad_profile = np.array([np.average(rad.dat.data[i::vnodes])
                           for i in range(vnodes)])

    terra_mu = np.loadtxt("./ARCHIVE/mu_2_lith.visc_smoothened.rad")
    terra_rad = np.linspace(rmax, rmin, terra_mu.shape[0])
    dists, inds = KDTree(terra_rad).query(rad_profile, k=2)
    mu_1d = np.sum(1/dists * terra_mu[inds], axis=1)/np.sum(1/dists, axis=1)
    mu_1d[dists[:, 0] <= 1e-6] = terra_mu[inds[dists[:, 0] <= 1e-6, 0]]

    averager = LayerAveraging(mesh, r1d=rad_profile, cartesian=False, quad_degree=6)
    averager.extrapolate_layer_average(
        viscosity, mu_1d)

    return viscosity


def load_tomography_model(mesh, fi_name):
    # Loading temperature model
    LLNL_model = seismic_model(fi_name=fi_name)

    # Set up function spaces
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)

    # tomography based temperature
    Tobs = Function(Q, name="Tobs")

    # radius and coordinates
    r = Function(V).interpolate(SpatialCoordinate(mesh))
    rad = Function(Q).interpolate(sqrt(SpatialCoordinate(mesh) ** 2))

    # knowing how many extrusion layers we have
    vnodes = nlayers + 1
    rad_profile = np.array([np.average(rad.dat.data[i::vnodes])
                           for i in range(vnodes)])

    # load seismic tomogrpahy based temperature model
    LLNL_model.load_seismic_data(rads=rad_profile)
    LLNL_model.setup_mesh(r.dat.data[0::vnodes])

    # Assigning values to each layer
    for i in range(vnodes):
        Tobs.dat.data[i::vnodes] = LLNL_model.fill_layer(i)

    return Tobs


def generate_spherical_mesh(mesh_filename):
    # Set up geometry:

    resolution_func = np.ones((nlayers))

    # A gaussian shaped function
    def gaussian(center, c, a):
        return a * np.exp(
            -((np.linspace(rmin, rmax, nlayers) - center) ** 2) / (2 * c**2)
        )

    # building the resolution function
    for idx, r_0 in enumerate([rmin, rmax, rmax - 660 / 6370]):
        # gaussian radius
        c = 0.15
        # how different is the high res area from low res
        res_amplifier = 5.0
        resolution_func *= 1 / (1 + gaussian(center=r_0, c=c, a=res_amplifier))

    resolution_func *= 1.0 / np.sum(resolution_func)
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)

    mesh = ExtrudedMesh(
        mesh2d,
        layers=nlayers,
        layer_height=resolution_func,
        extrusion_type="radial",
    )

    with CheckpointFile(mesh_filename, "w") as fi:
        fi.save_mesh(mesh=mesh)

    return mesh_filename


def rcf2sphfile_array_pyshtools(fname, lmax_calc=None):
    """
    The same as rcf2sphfile_pyshtools, but converting the
    output already to an array
    """
    with open(fname, mode="r") as f_id:
        line = f_id.readline()
        lmax = int(line.split(",")[0].split()[-1])
        nr = int(line.split(",")[-1].split()[1]) + 1
        lmax_calc = lmax if lmax_calc is None else lmax_calc
        sph_all = np.zeros((nr, 2, lmax_calc + 1, lmax_calc + 1))

        # Read in the comment for the name of the array
        line = f_id.readline()
        # Read in the radius
        line = f_id.readline()
        rshl = np.zeros(nr)
        for ir in range(nr):
            rshl[ir] = float(f_id.readline())
        # Read in the averages
        line = f_id.readline()
        if line != "# Averages\n":
            raise Exception(f'Error! Expect "# Averages". Got: {line}')
        for ir in range(nr):
            f_id.readline()
        line = f_id.readline()
        if line != "# Spherical Harmonics\n":
            raise Exception(
                f'Error! Expect "# Spherical Harmonics". Got: {line}')
        for ir in range(nr):
            line = f_id.readline()
            if line[0] != str("#"):
                raise Exception(f'Error! Expect "# Comment". Got: {line}')
            clm = SHCoeffs.from_zeros(lmax=lmax, normalization="ortho")
            for l in range(lmax + 1):
                for m in range(l + 1):
                    line = f_id.readline()
                    if m == 0:
                        clm.set_coeffs(float(line.split()[0]), l, 0)
                    else:
                        clm.set_coeffs(
                            [float(line.split()[0]), float(line.split()[1])],
                            [l, l],
                            [m, -m],
                        )
            sph_all[ir, :, :, :] = clm.coeffs[:,
                                              : lmax_calc + 1, : lmax_calc + 1]
    return rshl, sph_all


def cartesian_to_lonlat(x, y, z):
    """
    Convert Cartesian coordinates to longitude and latitude.

    Parameters:
    x, y, z : array-like
        Cartesian coordinates

    Returns:
    lon, lat : array-like
        Longitude and latitude in radians
    """

    lon = np.arctan2(y, x) * 180.0 / np.pi
    lat = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180.0 / np.pi

    return lon, lat


def lonlatto_cartesian(lat, lon, radius=1.0):
    """
    Convert latitude and longitude to 3D Cartesian coordinates.

    Parameters:
    - lat: latitude in degrees
    - lon: longitude in degrees
    - radius: radius of the Earth (default is 6371.0 km)

    Returns:
    - x, y, z: Cartesian coordinates
    """

    # Convert latitude and longitude from degrees to radians
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0

    # Calculate Cartesian coordinates
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    return x, y, z


class seismic_model(object):
    def __init__(self, fi_name):
        self.fi_name = fi_name
        self.coords = self.fibonacci_sphere(360 * 180)
        self.lon, self.lat = cartesian_to_lonlat(
            self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        )

    def fill_layer(self, layer_index):
        data = SHCoeffs.from_array(self.sph[layer_index], normalization="4pi").expand(
            lon=self.lon, lat=self.lat
        )
        data = 1 / 3900.0 * data - 3.0 / 39.0
        return IDW(np.column_stack((self.coords, data)), mode="3d").execute(
            self.target_mesh
        )

    def fibonacci_sphere(self, samples):
        """
        Generating equidistancial points on a sphere
        Fibannoci_spheres
        """

        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

        y = 1 - (np.array(list(range(samples))) / (samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * np.array(list(range(samples)))
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        return np.column_stack((x, y, z))

    def load_seismic_data(self, rads, k=2):
        self.rads = rads
        rshl, sph = rcf2sphfile_array_pyshtools(
            fname=self.fi_name, lmax_calc=None)

        rshl = np.array(
            [1 / 2890e3 * r + (rmax - 1 / 2890e3 * 6370e3) for r in rshl])
        tree = KDTree(rshl[:, np.newaxis])

        epsilon = 1e-10  # A small value to prevent division by zero

        dists, inds = tree.query(np.asarray(self.rads)[:, np.newaxis], k=k)

        # Add epsilon to dists to avoid division by zero
        self.sph = np.einsum(
            "i, iklm->iklm",
            1 / np.sum(1 / (dists + epsilon), axis=1),  # Add epsilon here
            np.einsum("ij, ijklm->iklm", 1 / (dists + epsilon), sph[inds]),  # And here
        )

        # Handle the case for very small distances separately
        self.sph[dists[:, 0] < 1e-6] = sph[inds[dists[:, 0] < 1e-6, 0]]

        self.lmax = self.sph[0].shape[1] - 1

    def setup_mesh(self, coords):
        self.target_mesh = coords


if __name__ == "__main__":
    __main__()
