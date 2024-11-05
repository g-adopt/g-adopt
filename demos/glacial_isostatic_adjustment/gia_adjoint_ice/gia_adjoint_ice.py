# Adjoint GIA modelling and ice inversions in an annulus
# =======================================================
#
# In this tutorial, we will have a first go at applying adjoint modelling to our cylindrical loading problem. We will see how to setup the adjoint model using pyadjoint, visualise gradients and have a go at inverting for a simple ice history using surface displacements.

# This example
# -------------
# Let's get started! As always, the first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality. We also import pyvista and matplotlib, for plotting.

from gadopt import *
from gadopt.utility import step_func, vertical_component
from forward_model import forward_model
import pyvista as pv
import matplotlib.pyplot as plt


# Next we need to tell G-ADOPT that we want to solve an inverse problem by importing `gadopt.inverse`. Under the hood this tells Firedrake and pyadjoint that we need to start *taping* the forward problem, so that an adjoint model is automatically generated.

#from gadopt.inverse import *

# No rol currently on my wsl installation!
from firedrake.adjoint import *
# emulate the previous behaviour of firedrake_adjoint by automatically
# starting the tape
continue_annotation()
tape = get_working_tape()
tape.clear_tape()


checkpoint_file = "../gia_2d_cylindrical/viscoelastic_loading-chk.h5"
with CheckpointFile(checkpoint_file, 'r') as afile:
    mesh = afile.load_mesh() # FIXME: name = ....
    target_displacement = afile.load_function(mesh, name="Displacement")


# Let's use the same triangular mesh from the *gia_2d_cylindrical* demo with a target resolution of 200km near the surface of the Earth coarsening to 500 km in the interior.

# Now let's setup the ice load. For this tutorial we will have two synthetic ice sheets. Let's put one a larger one over the South Pole, with a total horizontal extent of 40 $^\circ$ and a maximum thickness of 2 km, and a smaller one offset from the North Pole with a width of 20 $^\circ$ and a maximum thickness of 1 km. To simplify things let's keep the ice load fixed in time.

# +
W = FunctionSpace(mesh, "CG", 1)
Hice1 = 1000
Hice2 = 2000
# Disc ice load but with a smooth transition given by a tanh profile
disc_halfwidth1 = (2*pi/360) * 10  # Disk half width in radians
disc_halfwidth2 = (2*pi/360) * 20  # Disk half width in radians
surface_dx = 200*1e3
Re = 6371e3
ncells = 2*pi*Re/ surface_dx
surface_resolution_radians = 2*pi / ncells
X = SpatialCoordinate(mesh)
colatitude = atan2(X[0], X[1])
disc1_centre = (2*pi/360) * 25  # centre of disc1
disc2_centre = pi  # centre of disc2
disc1 = 0.5*(1-tanh((abs(colatitude-disc1_centre) - disc_halfwidth1) / (2*surface_resolution_radians)))
disc2 = 0.5*(1-tanh((abs(abs(colatitude)-disc2_centre) - disc_halfwidth2) / (2*surface_resolution_radians)))
target_normalised_ice_thickness = Function(W, name="target normalised ice thickness")
target_normalised_ice_thickness.interpolate(disc1 + (Hice2/Hice1)*disc2)

normalised_ice_thickness = Function(W).interpolate(target_normalised_ice_thickness*0.5)

control = Control(normalised_ice_thickness)
adj_ice_file = File(f"adj_ice.pvd")
top_id = 2
#converter = RieszL2BoundaryRepresentation(W, top_id)  # convert to surface L2 representation

final_inc_disp, final_disp = forward_model(mesh, normalised_ice_thickness)

circumference = 2 * pi * Re

# Define the component terms of the overall objective functional
displacement_error = final_disp - target_displacement
displacement_scale = 50
displacement_misfit = assemble(dot(displacement_error, displacement_error) / (circumference * displacement_scale**2) * ds(top_id))
damping = assemble((normalised_ice_thickness) ** 2 /circumference * ds)
smoothing = assemble(dot(grad(normalised_ice_thickness), grad(normalised_ice_thickness)) / circumference * ds)

alpha_smoothing = 0.1
alpha_damping = 0.1
J = displacement_misfit + alpha_damping * damping + alpha_smoothing * smoothing
log("J = ", J)
log("J type = ", type(J))


        # All done with the forward run, stop annotating anything else to the tape
pause_annotation()

updated_ice_thickness = Function(normalised_ice_thickness)
updated_ice_thickness_file = File(f"update_ice_thickness.pvd")
updated_displacement = Function(final_disp, name="updated displacement")
updated_incremental_displacement = Function(final_inc_disp, name="updated incremental displacement")
updated_out_file = File(f"updated_out.pvd")
def eval_cb(J, m):
    log("Control", m.dat.data[:])
    log("minimum Control", m.dat.data[:].min())
    log("J", J)
    circumference = 2 * pi * Re
    # Define the component terms of the overall objective functional
    damping = assemble((normalised_ice_thickness.block_variable.checkpoint) ** 2 /circumference * ds)
    smoothing = assemble(dot(grad(normalised_ice_thickness.block_variable.checkpoint), grad(normalised_ice_thickness.block_variable.checkpoint)) / circumference * ds)
    log("damping", damping)
    log("smoothing", smoothing)
    
    updated_ice_thickness.assign(m)
    updated_ice_thickness_file.write(updated_ice_thickness, target_normalised_ice_thickness)
    updated_displacement.interpolate(final_disp.block_variable.checkpoint) 
    updated_incremental_displacement.interpolate(final_inc_disp.block_variable.checkpoint)
    updated_out_file.write(updated_displacement, updated_incremental_displacement, target_displacement)
    #ice_thickness_checkpoint_file = f"updated-ice-thickness-iteration{c}.h5"
    #with CheckpointFile(ice_thickness_checkpoint_file, "w") as checkpoint:
    #    checkpoint.save_function(updated_ice_thickness, name="Updated ice thickness")
    #c += 1
reduced_functional = ReducedFunctional(J, control, eval_cb_post=eval_cb)

log("J", J)
log("replay tape RF", reduced_functional(normalised_ice_thickness))

grad = reduced_functional.derivative()
h = Function(normalised_ice_thickness)
h.dat.data[:] = np.random.random(h.dat.data_ro.shape)

#taylor_test(reduced_functional, normalised_ice_thickness, h)


# Perform a bounded nonlinear optimisation for the viscosity
# is only permitted to lie in the range [1e19, 1e40...]
ice_thickness_lb = Function(normalised_ice_thickness.function_space(), name="Lower bound ice thickness")
ice_thickness_ub = Function(normalised_ice_thickness.function_space(), name="Upper bound ice thickness")
ice_thickness_lb.assign(0.0)
ice_thickness_ub.assign(5)

bounds = [ice_thickness_lb, ice_thickness_ub]
#minimize(reduced_functional, bounds=bounds, options={"disp": True})

minimisation_problem = MinimizationProblem(reduced_functional, bounds=bounds)

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir=f"optimisation_checkpoint",
)
optimiser.run()

# If we're performing mulitple successive optimisations, we want
# to ensure the annotations are switched back on for the next code
# to use them
continue_annotation()

# -


# Let's visualise the ice thickness using pyvista, by plotting a ring outside our synthetic Earth.

# +
# Read the PVD file
ice_thickness = Function(W, name="Ice thickness").interpolate(Hice1 * disc1 + Hice2 * disc2)
zero_ice_thickness = Function(W, name="zero").assign(0)  # Used for plotting later
ice_thickness_file = VTKFile('ice.pvd').write(ice_thickness, zero_ice_thickness)
reader = pv.get_reader("ice.pvd")
data = reader.read()[0]  # MultiBlock mesh with only 1 block
# Make two points at the bounds of the mesh and one at the center to
# construct a circular arc.
normal = [0, 0, 1]
polar = [radius_values[0]-surface_dx/2, 0, 0]
center = [0, 0, 0]
angle = 360.0
arc = pv.CircularArcFromNormal(center, 500, normal, polar, angle)
arc_data = arc.sample(data)

# Stretch line by 20%
transform_matrix = np.array(
    [
        [1.2, 0, 0, 0],
        [0, 1.2, 0, 0],
        [0, 0, 1.2, 0],
        [0, 0, 0, 1],
    ]
)

transformed_arc_data = arc_data.transform(transform_matrix)
ice_cmap = plt.get_cmap("Blues", 25)

reader = pv.get_reader("viscosity.pvd")
data = reader.read()[0]  # MultiBlock mesh with only 1 block

# Create a plotter object
plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)

# Make a colour map
boring_cmap = plt.get_cmap("inferno_r", 25)
# Add the warped displacement field to the frame
plotter.add_mesh(
    data,
    component=None,
    lighting=False,
    show_edges=True,
    edge_color='grey',
    cmap=boring_cmap,
    scalar_bar_args={
        "title": 'Normalised viscosity',
        "position_x": 0.8,
        "position_y": 0.3,
        "vertical": True,
        "title_font_size": 20,
        "label_font_size": 16,
        "fmt": "%.0f",
        "font_family": "arial",
    }
)
plotter.add_mesh(
    transformed_arc_data,
    line_width=10,
    cmap=ice_cmap,
    scalar_bar_args={
        "title": 'Ice thickness (m)',
        "position_x": 0.1,
        "position_y": 0.3,
        "vertical": True,
        "title_font_size": 20,
        "label_font_size": 16,
        "fmt": "%.0f",
        "font_family": "arial",
    }
)
plotter.camera_position = 'xy'
plotter.show()
# Closes and finalizes movie
plotter.close()

