# Mantle convection base case using adaptive meshing
# ==================================================
# This tutorial demonstrates the adaptive meshing capability available in
# G-ADOPT, which dynamically modifies the mesh to focus resolution where needed.
# As demonstrated in [Davies et al.
# (2011)](https://doi.org/10.1029/2011GC003551) this may significantly reduce
# the computational requirements of mantle convection models, in particular
# when using anisotropic metric-based adaptivity which can very efficiently
# resolve the anisotropic features in geodynamical flows.
#
# Additional installation instructions
# ------------------------------------
# This functionality is available in G-ADOPT via the [mmg remeshing
# library](https://www.mmgtools.org/) and therefore requires a few extra flags
# in the configuration of the PETSc library used in the Firedrake installation
# as explained
# [here](https://github.com/mesh-adaptation/docs/wiki/Installation-Instructions).
# Additionally, for the assembly of the metric field, which allows us to
# specify exactly where mesh resolution is needed, we use the
# [animate](https://mesh-adaptation.github.io/docs/animate/index.html) Python
# package that can be pip installed from
# https://github.com/mesh-adaptation/animate

from gadopt import *
from animate import RiemannianMetric, adapt
from types import SimpleNamespace
from pop import generate_superparametric_spherical_shell, min_max_coordinates, correct_surface_radius
import numpy as np
import assess


nu = 1.0
l = 64
k = 64
m = l
solution = assess.SphericalStokesSolutionSmoothFreeSlip(l, m, k, nu=nu)

# In the other demos, we define a mesh and then, in turn, function spaces,
# functions, solvers etc. that depend on that mesh. In this demo, the
# anisotropic, metric-based mesh adaptivity will generate a completely new mesh
# after a number of time steps, to better resolve the solution as it evolves
# over time. The creation of function spaces, functions solvers etc.  will
# therefore have to be repeated after each mesh adaptivity step. We then run a
# few timesteps using these solvers, before we again adapt the mesh and repeat
# the whole process.
#
# For this reason we wrap most of the code of the base case demo into a python
# function that takes in a mesh, creates all mesh-dependent objects and runs a
# specified number of timesteps. We also need to provide the solution at the
# start of these timesteps. In the first call to this function this will simply
# be the initial conditions for the solution fields.  Each call will be
# followed by a mesh adaptivity step, but then we still have the solution that
# we have just solved at the end of the timesteps in that call on the mesh
# before the adapt. The python function below therefore takes whatever solution
# for velocity and pressure is provided and *interpolates* it onto
# the provided new mesh.
# +


def run_interval(mesh, time, timestep, Nt, u_init, p_init):
    """Run for Nt timesteps on the given mesh

    This sets up all the usual function spaces, equations, solvers, etc.
    on the given mesh, and runs the timeloop for the given n/o timesteps.
    Everything is exactly like in the base case.

    args:
    mesh - mesh to solve the equations on
    time - starting time (for logging purposes)
    timestep - starting timestep no. (for logging purposes)
    Nt - n/o timesteps to run
    u_init, p_init - initial conditions, or last solution from previous mesh
                             we interpolate these onto the current mesh

    returns:
    time, T, u, p - time, temperature, velocity and pressure at the end of the timesteps
    """

    mesh.cartesian = False
    boundary = SimpleNamespace(bottom=1, top=2)

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    z = Function(Z, name='Solution')  # A field over the mixed function space Z.

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
    log("Number of Temperature DOF:", Q.dim())

    elements = FunctionSpace(mesh, "DG", 0).dim()
    p1nodes = W.dim()
    p2nodes = Q.dim()

    # Set up temperature field and extract velocity and pressure from "mixed" function z
    T = Function(Q, name="Temperature")
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")

    # Interpolate velocity and pressure. In the first call this will
    # simply interpolate the provided UFL expresssions for the initial conditions.
    # In subsequent calls this will perform cross-mesh interpolation from the
    # solutions at the previous mesh, to the current mesh.
    u.interpolate(u_init)
    p.interpolate(p_init)

    gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top)

    xy = mesh.coordinates.dat.data
    assert xy.shape[0] == len(T.dat.data)
    T.dat.data[:] = [-solution.delta_rho_cartesian(xyi) for xyi in xy]

    # Stokes related constants (note that since these are included in UFL, they
    # are wrapped inside Constant):
    Ra = Constant(1)  # Rayleigh number
    approximation = BoussinesqApproximation(Ra)

    stokes_bcs = {
        boundary.bottom: {'un': 0},
        boundary.top: {'un': 0},
    }
    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

    # use tighter tolerances than default to ensure convergence:
    solver_params_extra = {
        "fieldsplit_0": {
            "ksp_rtol": 1e-5,
            "ksp_atol": 1e-7,
            "ksp_converged_reason": None,
        },
        "fieldsplit_1": {
            "ksp_rtol": 1e-4,
            "ksp_atol": 1e-7,
            "ksp_monitor": None,
        }
    }

    # Projection solver parameters for nullspaces:
    project_solver_parameters = {
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "pc_type": "sor",
        "mat_type": "aij",
        "ksp_rtol": 1e-12,
    }

    stokes_solver = StokesSolver(
        z,
        approximation,
        T,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
        near_nullspace=Z_near_nullspace,
        solver_parameters_extra=solver_params_extra,
    )

    min_max_r_str = " ".join([str(x) for x in min_max_coordinates(mesh)])

    xy = mesh.coordinates.dat.data[:]
    assert np.all(xy.shape == u.dat.data.shape)
    uana = Function(V, name='AnalyticalVelocity')
    uana.dat.data[:] = [solution.velocity_cartesian(xyi) for xyi in xy]

    X = SpatialCoordinate(mesh)
    p1_coords = Function(VectorFunctionSpace(mesh, "CG", 1), name="P1Coordinates")
    p1_coords.interpolate(X)
    p1xy = p1_coords.dat.data[:]
    assert p1xy.shape[0] == p.dat.data.shape[0]
    pana = Function(W, name='AnalyticalPressure')
    pana.dat.data[:] = [solution.pressure_cartesian(xyi) for xyi in p1xy]
    volume = assemble(Constant(1.0) * dx(domain=mesh))

    nsana = Function(W, name="AnalyticalSurfaceNormalStress")
    nsana.dat.data[:] = [-solution.radial_stress_cartesian(xyi) for xyi in p1xy]

    # Now perform the time loop:
    for ts in range(timestep, timestep+Nt):

        xy = mesh.coordinates.dat.data.shape

        # Write output:
        if ts % output_frequency == 0:
            output_file.write(u, p, T, uana, pana)

        # Solve Stokes sytem:
        stokes_solver.solve()

        # take out null modes through L2 projection from velocity and pressure
        # removing rotation from velocity:
        rot = as_vector((0, X[2], -X[1]))
        coef = assemble(dot(rot, u)*dx) / assemble(dot(rot, rot)*dx)
        u.project(u - rot*coef, solver_parameters=project_solver_parameters)
        rot = as_vector((-X[2], 0, X[0]))
        coef = assemble(dot(rot, u)*dx) / assemble(dot(rot, rot)*dx)
        u.project(u - rot*coef, solver_parameters=project_solver_parameters)
        rot = as_vector((-X[1], X[0], 0))
        coef = assemble(dot(rot, u)*dx) / assemble(dot(rot, rot)*dx)
        u.project(u - rot*coef, solver_parameters=project_solver_parameters)

        # removing constant nullspace from pressure
        coef = assemble(p * dx) / volume
        p.project(p - coef, solver_parameters=project_solver_parameters)

        # Compute diagnostics:
        energy_conservation = abs(abs(gd.Nu_top()) - abs(gd.Nu_bottom()))

        time += float(delta_t)

        ns = stokes_solver.force_on_boundary(boundary.top)
        uerr = np.sqrt(assemble(dot(u-uana, u-uana)*dx))
        perr = np.sqrt(assemble(dot(p-pana, p-pana)*dx))
        nserr = np.sqrt(assemble(dot(ns-nsana, ns-nsana)*ds(boundary.top)))

        # Log diagnostics:
        plog.log_str(f"{ts} {time} {float(delta_t)} "
                     + min_max_r_str +
                     f" {gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} {gd.Nu_top()} "
                     f"{gd.Nu_bottom()} {energy_conservation} {gd.T_avg()} "
                     f"{elements} {p1nodes} {p2nodes} "
                     f"{uerr} {perr} {nserr}")

    return time, T, u, p


# -

# Our initial mesh is very coarse. Note that for the type of mesh adaptivity
# we use here we need a triangular mesh.

ncells = 16
nlayers = 10
rmin = 1.22
rmax = 2.22
mesh = Mesh('spherical_shell.msh')
min_max_r = min_max_coordinates(mesh)
np.testing.assert_allclose(min_max_r, [rmin, rmin, rmax, rmax])
mesh_p2 = generate_superparametric_spherical_shell(mesh)
min_max_r = min_max_coordinates(mesh_p2)
np.testing.assert_allclose(min_max_r, [rmin, rmin, rmax, rmax])

# We set some options with regards to the timestepping. In particular
# we specify how many timesteps are performed between mesh adapts.

time = 0.0  # Initial time
timestep = 0  # Placeholder for initial timestep
timesteps_per_adapt = 2
delta_t = Constant(1)  # Initial time-step

# Create pvd-file to output solutions fields at the specified output
# frequency, which we can visualise using ParaView or pyvista. We need
# to pass `adaptive=True` to `VTKFile`, as the mesh will not be the
# same during the entire simulation.  We choose the same output
# frequency as the number of timesteps between mesh adapts, so that we
# get one output on each different mesh. Additionally, we open a log
# file to output diagnostic values.

# +
output_file = VTKFile("output.pvd", adaptive=True)
output_frequency = 1

plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt r_min_bot r_max_bot r_min_top r_max_top u_rms u_rms_surf ux_max nu_top nu_base energy avg_t "
             "elements p1nodes p2nodes "
             "uerr perr nserr"
             )
# -

# Initial conditions for the model, these will be interpolated onto
# the initial mesh later on.

X = SpatialCoordinate(mesh)
u = as_vector((0., 0., 0.))
p = 0.

# Metric based mesh adaptation
# ----------------------------
#
# The metric field is used to control where in the domain resolution is
# focussed in the adapted mesh. It is a rank 2 tensor field $M(x)$, providing a
# symmetric and positive definite dim x dim matrix at each location $x$, that
# encodes the local optimal edge length. Representing an edge by a vector $e$
# between two vertices, we define an optimal edge to satisfy the condition:
#
# $$e^T M(x) e \approx 1$$
#
# By our choice of the metric we can thus specify what edge lengths we desire
# in different directions.  For example, if we want an anisotropic mesh with
# edges of length 1 in the x-direction, and 5 in the y-direction, we choose the
# metric $M = \begin{pmatrix} 1 & 0 \\ 0 & 1/25\end{pmatrix}$.
#
# A common choice for the metric is to use the Hessian (second derivatives)
# $H(q)$ of a solution field $q$, typically scaled by some scalar $\epsilon$:
# $M(x) = \epsilon H(u)$.  In this way, we ask for smaller edges in the
# directions of high curvature in the solution (and vice versa). Moreover, this
# choice can be related to an estimate of the local interpolation error
# (through the second order term in a Taylor expansion) where our choice of
# $\epsilon$ determines the level of this interpolation error everywhere in the
# domain if the edges of the adapted mesh all satisfy the optimality condition.
# Mathematically, the interpolation error can be related to the numerical error
# in the solution for some PDEs (see [Céa's
# lemma](https://en.wikipedia.org/wiki/C%C3%A9a%27s_lemma)), but in practice it
# is hard to predict what level of estimated interpolation error is required
# for a desired level of accuracy. A more pragmatic choice therefore uses an
# estimate of the number of elements in the adapted mesh, which can be computed
# directly from the metric field, to choose the scale $\epsilon$ such that the
# estimated number of elements in the outputs corresponds to a user chosen
# number, referred to as target complexity.
#
# The $\epsilon$ that is computed in this way still corresponds to a
# level of estimated local interpolation error in the adapted mesh
# that is satisfied everywhere. Thus, we can think about the resulting
# mesh as the mesh that, for the specified desired number of elements,
# optimally distributes the resolution everywhere to achieve a certain
# uniform level of interpolation error.  In some simulations however,
# this choice may lead to too much focussing of resolution in areas of
# high curvature.  In particular, when there are discontinuities in
# the solution the curvature may become practically infinite,
# depending on resolution. Rather than aiming for the optimal mesh to
# have the same upper bound for interpolation error everywhere - in
# mathematical terms this means bounding the infinity norm of the
# local interpolation error - we can also ask for a local rescaling of
# the metric that minimizes the interpolation error in a different
# norm: the `animate` package allows us to specify any $L^p$ norm.
# The choice $p=\inf$ corresponds to the scaling with a constant
# $\epsilon$ as described here.
#
# Finally, we can select an overall minimum and maximum edge length, and a
# maximum aspect ratio to avoid excessively small, large, or flat cells
# respectively. The gradation factor limits the variation in edge lengths going
# from one cell to the next, where a factor of 1.5 mean that the edges in a
# neighbouring cell can only be 50% larger, and reversely, the edges in this
# cell can only be 50% larger than those in its neighbouring cells. The
# configuration choices for the metric that we have just described,
# are specified in the following dictionary:

metric_parameters = {
    # metric gets rescaled s.t. we always end up with ~ 1000 vertices:
    # 'dm_plex_metric_target_complexity': 10000,
    'dm_plex_metric_p': np.inf,  # Use infinity norm for estimated interpolation error
    'dm_plex_metric_gradation_factor': 1.5,  # Variation in edge length from one cell to another
    'dm_plex_metric_a_max': 10,  # maximum aspect ratio
    'dm_plex_metric_h_min': 1e-4,  # minimum edge length
    'dm_plex_metric_h_max': 0.5,  # maximum edge length
    'dm_plex_metric_hausdorff_number': 1,
    'dm_plex_metric_num_iterations': 10,
}

# Mesh adaptivity loop
# ----------------------------
#
# We perform `nadapts` iterations in which we perform `timesteps_per_adapt`
# timesteps, followed by the assembly of the metric based on the current
# solution and then we adapt the mesh based on that metric. The total number of
# timesteps, `nadapts*timesteps_per_adapt`, is chosen fairly low, so that you
# can run this relatively quickly and look at the results. To achieve steady
# state, at low Rayleigh number (Ra=1e4) you need `nadapts`>1000. For higher
# Rayleigh numbers the simulation never reaches steady state. For really high
# numbers (Ra>1e6) you will also need to increase the target complexity to
# ensure adaptivity provides sufficient resolution.
#
# To assemble the metric based on the Hessian of solution fields we use the
# `RiemannianMetric` class from `animate` which is a (subclass of a) Firedrake
# Function with additional functionality.  The `compute_hessian()` method
# provides a way to numerically reconstruct the Hessian of a scalar solution
# field. Since we have multiple solution fields available, we can combine the
# Hessians of these using either the `intersect()` or `average()` methods. The
# intersect method ensures that we impose the minimum edge length required to
# satisfy a certain interpolation error bound in all solution fields, whereas
# the average method simply uses the average of the required edge lengths. As
# the last step, we call the adapt() function with the current mesh and the
# metric, which will then use the Mmg library to return a newly adapted mesh
# according to our specifications.

# +
metric_pvd = VTKFile('metric.pvd', adaptive=True)
nadapts = 10
for _ in range(nadapts):
    time, T, u, p = run_interval(mesh_p2, time, timestep, timesteps_per_adapt, u, p)
    timestep += timesteps_per_adapt

    # we first assemble the metric as a P1 tensor field on the curved mesh_p2
    # (superparametric), to ensure the calculations such as the Hessian
    # reconstruction and normalisation are done with the most accurate geometric
    # domain representation
    TV = TensorFunctionSpace(mesh_p2, "CG", 1)
    # first generate a metric based on the Hessian for each velocity component
    metrics = []
    for i in range(3):
        # a RiemannianMetric is just a Firedrake Function on a tensor function space
        # with additional functionality
        H = RiemannianMetric(TV, name=f"{u.name()}{i}Metric")
        H.set_parameters(metric_parameters)
        H.compute_hessian(u.sub(i), method='L2')
        H.assign(H*Constant(1./4e-5))
        H.enforce_spd()
        metrics.append(H)

    # we use the first as *the* metric
    metric = RiemannianMetric(TV, name='CombinedMetric', metric_parameters=metric_parameters)
    metric.assign(metrics[0])
    # which we intersect with the others
    metric.intersect(*metrics[1:])
    metric_intersected = RiemannianMetric(TV, name='IntersectedMetric')
    metric_intersected.assign(metric)

    # this applies the rescaling to achieve the desired target complexity
    # (estimate of number of elements)
    metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
    # metric.normalise()

    metric_pvd.write(metric, metric_intersected, *metrics)

    # the adaptivity itself is applied on the linear mesh (i.e. straight
    # triangles) and so we need a version of the metric on that mesh
    TVP1 = TensorFunctionSpace(mesh, "CG", 1)  # function space for the metric
    metric_p1 = RiemannianMetric(TVP1)
    metric_p1.set_parameters(metric_parameters)
    metric_p1.interpolate(metric)

    mesh = adapt(mesh, metric_p1)

    # due to adaptivity at the surface, the surface nodes are not
    # necessarily exactly on the analytical surface (circle/sphere)
    # anymore. However mmg seems (in experiments so far) fairly good
    # at approximately maintaing the analytical surface, so it appears
    # we can fix by simply moving the surface nodes back to their
    # fixed radius (without tangling, so far)
    correct_surface_radius(mesh, 1, rmin)
    correct_surface_radius(mesh, 2, rmax)
    # the new, adapted mesh is linear, so we again generate a superparametric
    # version, where the additional edges nodes are located through linear
    # interpolation of the radius, which a.o. ensures that on a analytical
    # boundary surface of constant radius, all new nodes also have this radius
    mesh_p2 = generate_superparametric_spherical_shell(mesh)

plog.close()
# -

# To look at the results you can use the following code that reads back in the
# .vtu-files that have been produced, and writes a .gif-movie. As you can see
# in the movie, the adaptive mesh captures the main features of the flow,
# using anisotropic meshes to efficiently resolve the boundary
# layers.

# + tags=["active-ipynb"]
# import pyvista as pv
#
# plotter = pv.Plotter(notebook=True)
# plotter.open_gif('movie.gif')
# plotter.camera_position = "xy"
#
# for i in range(nadapts):
#     mesh_data = pv.read(f"output/output_{i}.vtu")
#     plotter.add_mesh(mesh_data, scalars='Temperature')
#     edges = mesh_data.extract_all_edges()
#     plotter.add_mesh(edges, color="black")
#     plotter.view_xy()
#     plotter.write_frame()
#     plotter.clear()
# plotter.close()
# -

# ![adaptive mesh base case](./movie.gif)

# To learn more about the anisotropic, metric based adaptivity and various
# choices that can be made to assemble the metric, see the [documentation of
# animate](https://mesh-adaptation.github.io/docs/animate/index.html) which
# also provides a number of tutorials.
