import os, sys
sys.path.insert(0, '../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from gadopt import *
from utilities import data_2_function
from richards_equation import *
from richards_solver import *
from soil_curves import *
import ufl
import time

"""
A case study of the Lower Murrumbidgee basin
=================================================================
"""
# Details of mesh extrusion
number_layers, mesh_depth = 800, 400
layer_height = mesh_depth/number_layers

mesh2D = Mesh('MurrumbidgeeMeshSurface.msh') # Load 2d mesh
mesh = ExtrudedMesh(mesh2D, number_layers, layer_height=layer_height, extrusion_type='uniform')

x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DG", 1)

m = V.mesh()
W = VectorFunctionSpace(m, V.ufl_element())
X = assemble(interpolate(m.coordinates, W))
mesh_coords = X.dat.data_ro

# Transform mesh such that top and bottom are the points given by elevation_data.csv and bedrock_data.csv
mesh.coordinates.dat.data[:, 2] = data_2_function(mesh_coords, 'bedrock_data.csv')*mesh.coordinates.dat.data[:, 2]/mesh_depth + data_2_function(mesh_coords, 'elevation_data.csv') - data_2_function(mesh_coords, 'bedrock_data.csv')
