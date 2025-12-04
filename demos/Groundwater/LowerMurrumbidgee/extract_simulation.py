from gadopt import *

with CheckpointFile("DG10_dx=3000_layers=300.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    h     = afile.load_function(mesh, "PressureHead")
    theta = afile.load_function(mesh, "MoistureContent")
    q     = afile.load_function(mesh, "VolumetricFlux")
    depth = afile.load_function(mesh, "depth")

outfile = VTKFile("lower_murrumbidgee.pvd")
outfile.write(h, theta, q, depth)