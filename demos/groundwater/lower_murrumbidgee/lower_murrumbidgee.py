from gadopt import *
from surface_mesh import *

"""
Lower Murrumbidgee River Basin

This script simulates groundwater flow using a 3D Discontinuous Galerkin (DG) 
discretization on an extruded mesh. It accounts for terrain-following coordinates
and depth-dependent soil properties.

"""

def setup_mesh_and_spaces():

    """
    Build the extruded 3D mesh, apply the terrain-following coordinate transform, and construct all required function spaces and spatial fields. 
    """

    # --- Build extruded mesh ---
    horizontal_resolution = 3500
    number_layers         = 300
    layer_height          = 1/number_layers

    surface_mesh(horizontal_resolution)
    mesh2D = Mesh('MurrumbidgeeMeshSurface.msh')

    mesh = ExtrudedMesh(
        mesh2D,
        number_layers,
        layer_height=layer_height,
        extrusion_type='uniform',
        name='mesh'
        )
    
    W = VectorFunctionSpace(mesh, 'CG', 1)
    X = assemble(interpolate(mesh.coordinates, W))
    mesh_coords = X.dat.data_ro

    z = mesh_coords[:, 2]  # uniform [0,1] vertical coordinate

    bedrock_raw   = data_2_function(mesh_coords, 'bedrock_data.csv')
    elevation_raw = data_2_function(mesh_coords, 'elevation_data.csv')

    mesh.coordinates.dat.data[:, 2] = bedrock_raw*z + elevation_raw - bedrock_raw

    # --- Function spaces ---
    poly_degree = 1
    V = FunctionSpace(mesh, "DQ", poly_degree)
    W = VectorFunctionSpace(mesh, "DQ", poly_degree)

    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

    # --- Load spatial fields (elevation, layers, rainfall, etc.) ---
    Vcg = FunctionSpace(mesh, 'CG', 1)
    x = SpatialCoordinate(mesh)

    elevation    = load_spatial_field("Elevation",    "elevation_data.csv", V, Vcg, mesh_coords)
    shallowLayer = load_spatial_field("shallowLayer", "shallow_layer.csv",  V, Vcg, mesh_coords)
    lowerLayer   = load_spatial_field("lowerLayer",   "lower_layer.csv",    V, Vcg, mesh_coords)
    bedrock      = load_spatial_field("Bedrock",      "bedrock_data.csv",   V, Vcg, mesh_coords)
    watertable   = load_spatial_field("WaterTable",   "water_table.csv",    V, Vcg, mesh_coords)
    rainfall     = load_spatial_field("Rainfall",     "rainfall_data.csv",  V, Vcg, mesh_coords)
    depth        = Function(V, name='depth').interpolate(elevation - x[2])

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

    t_final_years = 1
    t_final = t_final_years * 3.156e+7  # in seconds

    dt_days = 1
    dt = Constant(dt_days * 86400)  # in seconds

    time_integrator = BackwardEuler

    return t_final, dt, time_integrator


def define_soil_curves(mesh, V, spatial_data):

    """Specifies depth-dependent hydraulic properties using Haverkamp curves.
    
    Uses tanh-based indicator functions to transition between soil layers 
    (Shallow, Lower, Bedrock) smoothly to assist nonlinear solver convergence.
    """

    # Extract relavant spatial profiles
    shallowLayer = spatial_data['shallowLayer']
    lowerLayer  = spatial_data['lowerLayer']
    depth       = spatial_data['depth']

    # Indicator functions I1, I2 transition between shallow, lower, and bedrock layers.
    delta = 1
    I1 = 0.5*(1 + tanh(delta*(shallowLayer - depth)))
    I2 = 0.5*(1 + tanh(delta*(lowerLayer - depth)))

    # We employ a depth (in metres) dependent porosity and water saturation based on emperical formula
    S_depth = 1/((1 + 0.000071*depth)**5.989)     # Depth dependent porosity
    K_depth = (1 - depth / (58 + 1.02*depth))**3  # Depth dependent conductivity
    Ks = Function(V, name='SaturatedConductivity').interpolate(K_depth*(2.5e-05*I1 + 1e-03*(1 - I1)*I2 + 5e-04*(1-I2)))

    # Specify the hydrological parameters
    soil_curve = HaverkampCurve(
        theta_r = 0.025,         # Residual water content [-]
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
    depth      = spatial_data['depth']


    richards_bcs = {
        1        : {'h' : depth - watertable},           # Side boudaries
        'bottom' : {'flux': 0},
        'top'    : {'flux': 0},
    }

    return richards_bcs


def main():

    time_var = Constant(0.0)

    mesh, V, W, spatial_data = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curve = define_soil_curves(mesh, V, spatial_data)
    richards_bcs = setup_boundary_conditions(mesh, time_var, spatial_data)
    
    # Define initial condition
    depth      = spatial_data['depth']
    watertable = spatial_data['watertable']

    x = SpatialCoordinate(mesh)
    h = Function(V, name="PressureHead").interpolate(depth - watertable)

    solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": 'gmres',
                    "pc_type": 'bjacobi',
                    'snes_type': 'newtonls'
                    }

    richards_solver = RichardsSolver(
        h,
        soil_curve,
        delta_t=dt,
        timestepper=time_integrator,
        bcs=richards_bcs,
        solver_parameters=solver_parameters,
    )

    current_time = 0

    while current_time < t_final:

        time_var.assign(current_time)

        richards_solver.solve()
        current_time += float(dt)
        PETSc.Sys.Print(current_time)
        

if __name__ == "__main__":
    main()
