from RichardsSolver import *
from surface_mesh import *
import time

import numpy as np


"""
Lower Murrumbidgee River Basin

This script simulates groundwater flow using a 3D Discontinuous Galerkin (DG) 
discretization on an extruded mesh. It accounts for terrain-following coordinates
and depth-dependent soil properties.

"""


def setup_mesh_and_spaces():


    # --- Function spaces ---
    horiz_elt = FiniteElement("DG", triangle, 1)
    vert_elt  = FiniteElement("DG", interval, 1)
    factor = 1.00

    refinement_levels     = 0
    refinement_ratio      = 1
    horizontal_resolution = 2000*(2**refinement_levels)/factor  # Coarsest resolution if using mg
    number_layers         = round(factor*300/(1))
    layer_height          = 1.00/number_layers

    # --- build 2d heiracy mesh ---
    surface_mesh(horizontal_resolution)
    m2d = Mesh("Murrumbidgee_SurfaceMesh.msh")
    mh2d = MeshHierarchy(m2d, refinement_levels=refinement_levels)

    # 2. Extrude the hierarchy
    # Note: Keep it flat at first (height=1.0)
    mh3d = ExtrudedMeshHierarchy(mh2d, height=1.0, base_layer=number_layers, refinement_ratio=refinement_ratio)

    V0  = FunctionSpace(mh3d[0], "CG", 1)
    coords_coarse = mh3d[0].coordinates

    bed0 = Function(V0); bed0.dat.data[:] = data_2_function(coords_coarse.dat.data_ro, 'bedrock_data.csv')
    elv0 = Function(V0); elv0.dat.data[:] = data_2_function(coords_coarse.dat.data_ro, 'elevation_data.csv')

    beds = [bed0]
    elvs = [elv0]
    for lev in range(1, len(mh3d)):
        Vi  = FunctionSpace(mh3d[lev], "CG", 1)
        bed = Function(Vi); prolong(beds[-1], bed)   # safe & cheap on flat geometry
        elv = Function(Vi); prolong(elvs[-1], elv)
        beds.append(bed); elvs.append(elv)

    for lev, m in enumerate(mh3d):
        coords = m.coordinates
        z      = coords.dat.data[:, 2]
        bed    = beds[lev].dat.data[:]
        elv    = elvs[lev].dat.data[:]
        # Your mapping: z_new = bedrock * z + elevation - bedrock
        coords.dat.data[:, 2] = bed * z + elv - bed


    mh3d_moved = HierarchyBase(
        list(mh3d),                           # moved levels in order
        mh3d.coarse_to_fine_cells,
        mh3d.fine_to_coarse_cells,
        refinements_per_level=mh3d.refinements_per_level,
        nested=mh3d.nested)

    mesh = mh3d_moved[-1]

    W = VectorFunctionSpace(mesh, 'CG', 1)
    X = assemble(interpolate(mesh.coordinates, W))
    mesh_coords = X.dat.data_ro

    elt = TensorProductElement(horiz_elt, vert_elt)
    V   = FunctionSpace(mesh, elt)
    W   = VectorFunctionSpace(mesh, elt)

    PETSc.Sys.Print("Refinement levels:", refinement_levels)
    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())
    PETSc.Sys.Print("Horizontal resolution:", horizontal_resolution)
    PETSc.Sys.Print("Number of layers:", number_layers)
    PETSc.Sys.Print(horiz_elt, vert_elt)

    # --- Load spatial fields (elevation, layers, rainfall, etc.) ---
    Vcg = FunctionSpace(mesh, 'CG', 1)
    x = SpatialCoordinate(mesh)

    elevation    = load_spatial_field("Elevation",    "elevation_data.csv", V, Vcg, mesh_coords)
    shallowLayer = load_spatial_field("shallowLayer", "shallow_layer.csv",  V, Vcg, mesh_coords)
    lowerLayer   = load_spatial_field("lowerLayer",   "lower_layer.csv",    V, Vcg, mesh_coords)
    bedrock      = load_spatial_field("Bedrock",      "bedrock_data.csv",   V, Vcg, mesh_coords)
    watertable   = load_spatial_field("WaterTable",   "water_table.csv",    V, Vcg, mesh_coords)
    rainfall     = load_spatial_field("Rainfall",     "rainfall_data.csv",  V, Vcg, mesh_coords)
    depth        = Function(V, name='depth').interpolate(conditional(elevation - x[2] < 0, 0.0, elevation - x[2]))

    # Package spatial data
    spatial_data = {
        'depth'       : depth,
        'elevation'   : elevation,
        'bedrock'     : bedrock,
        'lowerLayer'  : lowerLayer,
        'shallowLayer': shallowLayer,
        'rainfall'    : rainfall,
        'watertable'  : watertable
    }

    return mesh, V, W, spatial_data


def define_time_parameters():

    """Sets simulation time and time-stepping constants."""

    t_final_years = 0.20
    t_final = t_final_years * 3.156e+7  # in seconds

    dt_days = 0.5
    dt = Constant(dt_days * 86400)  # in seconds

    time_integrator = "BackwardEuler"

    PETSc.Sys.Print(time_integrator)

    return t_final, dt, time_integrator


def define_soil_curves(mesh, V, spatial_data):

    """Specifies depth-dependent hydraulic properties using Haverkamp curves.
    
    Uses tanh-based indicator functions to transition between soil layers 
    (Shallow, Lower, Bedrock) smoothly to assist nonlinear solver convergence.
    """

    # Extract relavant spatial profiles
    shallowLayer = spatial_data['shallowLayer']
    lowerLayer  = spatial_data['lowerLayer']
    depth       = max_value(spatial_data['depth'], 0.0)

    # Indicator functions I1, I2 transition between shallow, lower, and bedrock layers.
    delta = 0.2
    I1 = 0.5*(1 + tanh(delta*(shallowLayer - depth)))
    I2 = 0.5*(1 + tanh(delta*(lowerLayer - depth)))

    # We employ a depth (in metres) dependent porosity and water saturation based on emperical formula
    S_depth = max_value(1/((1 + 0.000071*depth)**5.989), 0)     # Depth dependent porosity
    K_depth = max_value((1 - depth / (58 + 1.02*depth))**3, 0)  # Depth dependent conductivity
    Ks_layer1 = 2.5e-05
    Ks_layer2 = 1e-03
    Ks_layer3 = 5e-04
    Ks = Function(V, name='SaturatedConductivity').interpolate(K_depth*(Ks_layer1*I1 + Ks_layer2*(1 - I1)*I2 + Ks_layer3*(1-I2)))

    # Specify the hydrological parameters
    soil_curve = HaverkampCurve(
        theta_r = 0.399*S_depth,         # Residual water content [-]
        theta_s = 0.40*S_depth,  # Saturated water content [-]
        Ks      = Ks,            # Saturated hydraulic conductivity [m/s]
        alpha   = 0.44,          # Fitting parameter [m]
        beta    = 1.2924,        # Fitting parameter [-]
        A       = 0.0104,        # Fitting parameter [m]
        gamma   = 1.5722,        # Fitting parameter [-]
        Ss      = 0,             # Specific storage coefficient [1/m]
    )

    return soil_curve


def setup_boundary_conditions(mesh, time_var, spatial_data):

    """
    Produces a dictionary that describes the imposed boundary conditions:
        top - rainfall imposed from external data
        bottom - localised extraction points and bedrock (no-flux) elsewhere
        sides - water table is fixed
    """

    x = SpatialCoordinate(mesh)

    watertable = spatial_data['watertable']
    rainfall   = spatial_data['rainfall']
    depth       = max_value(spatial_data['depth'], 0.0)

    rain_scaling = 0.14 * 3.171e-11  # Percentage of rainfall that enters ground

    richards_bcs = {
        1        : {'h' : depth - watertable},               # Side boudaries
        'bottom' : {'flux': 0},
        'top'    : {'flux': rain_scaling*rainfall},
    }

    return richards_bcs


def source_term(mesh, spatial_data):

    """
    Models the extraction of water from wells/bore holes
    """

    x = SpatialCoordinate(mesh)

    depth       = max_value(spatial_data['depth'], 0.0)

    # Extraction/sink points
    spread_h, spread_v = 5e07, 10  # defines the radius of influence for each pumping well.
    extraction_indicator = 0*x[0]

    # Coordinates of extraction sites
    xPts = np.array([ 1.7e05, 2.2e05, 2.4e05, 2.0e05, 1.6e05, 1.7e05, 2.2e05, 2.5e05, 2.0e05, 1.5e05, 2.3e05, 1.0e05, 2.0e05, 1.9e05, 1.9e05, 1.9e05, 1.6e05, 2.6e05, 1.2e05, 2.5e05 ])
    yPts = np.array([ 8.0e04, 4.3e04, 4.2e04, 7.3e04, 3.5e04, 9.3e04, 6.0e04, 6.5e04, 6.0e04, 5.0e04, 9.0e04, 6.5e04, 2.0e04, 1.0e05, 8.5e04, 4.4e04, 6.0e04, 5.5e04, 6.5e04, 2.2e04 ])
    well_depth = 30

    for x0, y0 in zip(xPts, yPts):
        rh = (x[0] - x0)**2 + (x[1] - y0)**2
        rv = (x[2] - well_depth)**2
        extraction_indicator -= exp(-rh / spread_h - rv / spread_v)

    extraction = 5e-10 * extraction_indicator

    return extraction

def main():

    time_var = Constant(0.0)

    mesh, V, W, spatial_data = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curves = define_soil_curves(mesh, V, spatial_data)
    richards_bcs = setup_boundary_conditions(mesh, time_var, spatial_data)
    extraction = source_term(mesh, spatial_data)

    x = SpatialCoordinate(mesh)

    # Define initial condition
    depth       = spatial_data['depth']
    watertable = spatial_data['watertable']

    solver_parameters = {"ksp_type": "gmres", 
                        "pc_type": "bjacobi",
                        "sub_ksp_type": "preonly",
                        "sub_pc_type": "ilu",
                        "sub_pc_factor_levels": 0,
                        "snes_linesearch_type": "bt",
                        }

    PETSc.Sys.Print(solver_parameters)  

    # Solver Instantiation
    eq = RichardsSolver(V=V,
                        W=W,
                        mesh=mesh,
                        soil_curves=soil_curves,
                        bcs=richards_bcs,
                        solver_parameters=solver_parameters,
                        source_term=extraction,
                        time_integrator=time_integrator,
                        )
    
    initial_head = Function(V, name="InitialCondition").interpolate(depth - watertable)
    #initial_head = Function(V, name="InitialCondition").interpolate(-10)

    h     = Function(V, name="PressureHead").assign(initial_head)
    h_old = Function(V, name="OldSolution").assign(h)
    q     = Function(W, name='VolumetricFlux')
    
    moisture_content      = soil_curves.moisture_content
    relative_conductivity = soil_curves.relative_conductivity
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))
    K     = Function(V, name='RelativeConductivity').interpolate(relative_conductivity(h))

    # Solver Instantiation
    richards_solver = richardsSolver(h, h_old, time_var, dt, eq)

    current_time        = 0.0
    n_steps             = 0.0
    
    total_exterior_flux = 0.0
    total_extracted     = 0.0
    initial_mass        = assemble(theta*eq.dx)

    total_nl_it = 0.0
    total_l_it  = 0.0

    n_steps = 0
    t_next = 0
    t_interval = 3.154e+7

    while current_time < t_final:

        time_var.assign(current_time)

        # Solve model with retries
        h_old.assign(h)
        
        start_time = time.perf_counter()
        richards_solver.solve()
        end_time = time.perf_counter()

        PETSc.Sys.Print(f"The code block executed in {(end_time - start_time):.4f} seconds")

        snes = richards_solver.snes
        nl_it = snes.getIterationNumber()
        l_it  = snes.ksp.getIterationNumber()


        current_time += float(dt)

        theta.interpolate(moisture_content(h))
        water_content = assemble(theta*eq.dx)
        K.interpolate(relative_conductivity((h+h_old)/2))
        q.interpolate(-K*grad((h+h_old)/2 + x[eq.dim-1]))

        # Diagnostics and mass balance
        n_steps             += 1
        total_exterior_flux += assemble(dt*dot(q, -eq.n)*eq.ds)
        total_nl_it         += nl_it
        total_l_it          += l_it
        total_extracted += float(dt)*assemble(extraction*dx)
        exterior_flux = 0

        PETSc.Sys.Print( f"t = {current_time/(24*3600):.2f} d | "
                        f"dt = {float(dt)/(24*3600):.2f} d | "
                        f"NL iters = {nl_it} | "
                        f"L iters = {l_it} | "
                        f"Water content = {water_content:.5e} | "
                        f"Exterior flux = {exterior_flux:.5e} | "
                        f'Water Balabnce =  {(water_content-initial_mass)/(total_extracted+total_exterior_flux):.4f}'
                        )

    final_mass = assemble(theta*eq.dx)
    PETSc.Sys.Print(f"Number of timesteps: {n_steps}")
    PETSc.Sys.Print(f"Total number of nonlinear iterations: {total_nl_it}")
    PETSc.Sys.Print(f"Total number of linear iterations: {total_l_it}")
    PETSc.Sys.Print("")
    PETSc.Sys.Print(f"Initial mass: {initial_mass}")
    PETSc.Sys.Print(f"Final mass: {final_mass}")
    PETSc.Sys.Print(f"Exterior flux: {exterior_flux}")
    PETSc.Sys.Print(f"Total extracted: {total_extracted}")




if __name__ == "__main__":
    main()
