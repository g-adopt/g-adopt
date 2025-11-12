# Level-set initialisaion
# =

# Rationale
# -

# G-ADOPT handles multi-material simulations via the conservative level-set approach,
# which is an interface-capturing method. At the core of this method lies the need to
# calculate a signed-distance function, which expresses the distance from the material
# interface location to any point in the numerical domain. As calculating such a field
# can be challenging, G-ADOPT exposes the `assign_level_set_values` function as part of
# its API to ease computing the signed-distance function. Under the hood, this function
# leverages `Shapely` to calculate planar distances.

# This example
# -

# Here, we demonstrate how to initialise the level-set function in multiple scenarios.
# We restrict our demonstration to 2-D geometries but will note when extension to 3-D is
# possible.

# As with all examples, the first step is to import the `gadopt` package, which
# provides access to Firedrake and associated functionality. We also import `matplotlib`
# for plotting purposes and the `linspace` function from `numpy`.

from gadopt import *

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# from numpy import linspace
# -

# We start by writing a function that will use repeatedly to visualise the level-set
# field throughout this tutorial. To this end, we use Firedrake's built-in plotting
# functionality.


# + tags=["active-ipynb"]
# def plot_level_set(psi: Function):
#     fig, axes = plt.subplots(figsize=(12, 8))
#     axes.set_aspect("equal")

#     contourf = tricontourf(
#         psi, levels=linspace(0.0, 1.0, 11), cmap="PiYG", extend="both", axes=axes
#     )
#     tricontour(psi, levels=[0.5], axes=axes)
#     fig.colorbar(contourf, label="Conservative level set")


# -

# Let's now set up a rectangular mesh; we will investigate an annulus mesh later on. We
# define the function space where the level-set field lives and its corresponding
# function.

# +
mesh_elements = (100, 100)  # Number of cells in x and y directions
domain_dims = (1.0, 1.0)  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(*mesh_elements, *domain_dims, quadrilateral=True)
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects

# Level-set function space (scalar, discontinuous, attains maximal values at nodes)
K = FunctionSpace(mesh, "DQ", 2, variant="equispaced")

psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now define objects that will be useful for level-set initialisation: spatial
# coordinates and the thickness of the hyperbolic tangent profile used in the
# conservative level-set formulation.
x, y = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates
epsilon = interface_thickness(K, min_cell_edge_length=True)

# Let's start exploring some possible initialisation strategies. A first scenario is the
# one for which the signed-distance function can be easily deduced from the domain's
# spatial coordinates. Here, we take the example of an interface located at $x = 0.2$.

# +
interface_coord_x = 0.2
signed_distance = interface_coord_x - x

assign_level_set_values(psi, epsilon, signed_distance)
# -

# As you can see, initialisation is straightforward in this case, and we will note that
# it is compatible with 3-D domains. Let us visualise the location of the material
# interface that we have just initialised to verify its correctness.

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# We successfully obtain a scaled hyperbolic tangent profile ranging from 0 to 1, where
# the material interface corresponds to the 0.5 isocontour.

# What would have happened if we defined the signed-distance function the opposite way?
# The positive and negative sides would have been swapped, and so would have the 0 and 1
# sides of the conservative level-set field. Let's confirm this thought experiment.

assign_level_set_values(psi, epsilon, -signed_distance)

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# As expected, sides have been swapped.

# Using the same approach, we can specify a circular material interface. Let's write the
# signed-distance function corresponding to a circle of radius 0.3, centred on
# $(x, y) = (0.6, 0.4)$.

# +
circle_centre = (0.6, 0.4)
circle_radius = 0.3
signed_distance = (
    (circle_centre[0] - x) ** 2 + (circle_centre[1] - y) ** 2 - circle_radius**2
)

assign_level_set_values(psi, epsilon, signed_distance)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# The material interface, as delineated by the 0.5 isocontour, does represent a circle.

# It is not always possible to easily deduce the mathematical expression of the
# signed-distance function. In such cases, G-ADOPT allows a user to mathematically
# describe the interface geometry as a sufficient step to calculate the signed-distance
# function. This endeavour is delegated to `Shapely`. Let's start with the case where
# the material interface is a curve. We will explore two possibilities: either G-ADOPT
# exposes an implementation of that curve or it does not, in which case a user will have
# to provide the implementation. We will start with the first possibility and examine a
# material interface represented by a cosine function.

# +
callable_args = (
    curve_parameter := linspace(0.0, domain_dims[0], 1000),
    interface_deflection := 0.4,
    perturbation_wavelength := domain_dims[0] / 3.0,
    interface_coord_y := 0.6,
)
boundary_coordinates = [domain_dims, (0.0, domain_dims[1]), (0.0, interface_coord_y)]

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=callable_args,
    boundary_coordinates=boundary_coordinates,
)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -
