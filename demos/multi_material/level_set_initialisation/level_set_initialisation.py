# Level-set initialisaion
# =

# Rationale
# -

# G-ADOPT handles multi-material simulations via the conservative level-set approach
# ([Olsson and Kreiss, 2005](https://doi.org/10.1016/j.jcp.2005.04.007)), which is an
# interface-capturing method. At the core of this method lies the need to calculate a
# signed-distance function, which expresses the distance from the material interface
# location to any point in the numerical domain. As calculating such a field can be
# challenging, G-ADOPT exposes the `assign_level_set_values` function as part of its API
# to ease computing the signed-distance function. Under the hood, this function
# leverages [`Shapely`](https://shapely.readthedocs.io/en/stable/) to calculate planar distances.

# This example
# -

# Here, we demonstrate how to initialise the level-set function in multiple scenarios.
# We restrict our demonstration to 2-D geometries but will note when extension to 3-D is
# possible.

# As with all examples, the first step is to import the `gadopt` package, which
# provides access to Firedrake and associated functionality. We also import `shapely`
# for handling geometric shapes, `matplotlib` for plotting purposes, and `numpy` for
# generic mathematical manipulations.

import shapely as sl

from gadopt import *

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import numpy as np
# -

# We start by writing a function that will be used repeatedly throughout this tutorial
# to visualise the level-set field. To this end, we use Firedrake's built-in plotting
# functionality.


# + tags=["active-ipynb"]
# def plot_level_set(psi: Function) -> None:
#     fig, axes = plt.subplots(figsize=(14, 6))
#     axes.set_aspect("equal")

#     contourf = tricontourf(
#         psi, levels=np.linspace(0.0, 1.0, 11), cmap="PiYG", extend="both", axes=axes
#     )
#     tricontour(psi, levels=[0.5], axes=axes)
#     fig.colorbar(contourf, label="Conservative level set")


# -

# Let's now set up a mesh, define the function space where the level-set field lives,
# and its corresponding function.

# +
mesh_elements = (100, 200)  # Number of cells in x and y directions
domain_dims = (2.0, 1.0)  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(*mesh_elements, *domain_dims, quadrilateral=True)
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects

# Level-set function space (scalar, discontinuous, attains maximal values at nodes)
K = FunctionSpace(mesh, "DQ", 2, variant="equispaced")

psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now set up objects that will be useful for level-set initialisation: spatial
# coordinates and the thickness of the hyperbolic tangent profile defined in the
# conservative level-set formulation.

x, y = SpatialCoordinate(mesh)  # Extract UFL representation of spatial coordinates
epsilon = interface_thickness(K, min_cell_edge_length=True)

#### Providing the signed-distance function

# Let's start exploring some possible initialisation strategies. A first scenario is the
# one for which the signed-distance function can be easily deduced from the domain's
# spatial coordinates. Here, we take the example of an interface located at $x = 0.2$.

# +
interface_coord_x = 0.2
signed_distance = interface_coord_x - x

assign_level_set_values(psi, epsilon, signed_distance)
# -

# As you can see, initialisation is straightforward in this case, and it does not
# require any external package under the hood, making it compatible with 3-D domains.
# Let's visualise the location of the material interface that we have just initialised
# to verify its correctness.

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
# $(x, y) = (1.4, 0.4)$.

# +
circle_centre = (1.4, 0.4)
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

#### Providing a mathematical description of the interface

# It is not always possible to easily deduce the mathematical expression of the
# signed-distance function. In such cases, G-ADOPT allows a user to mathematically
# describe the interface geometry as a sufficient step to calculate the signed-distance
# function. Under the hood, this mathematical description is provided to `Shapely`,
# which generates a geometric object representing the interface. `Shapely` then computes
# the distance of any mesh node to this object (i.e. the material interface), which
# G-ADOPT then uses to determine the conservative level-set profile.

##### Curve interface

# Let's start with the case where the material interface is a curve. We will explore two
# possibilities: either G-ADOPT exposes an implementation of that curve or it does not,
# in which case a user will have to provide the implementation. We will start with the
# first possibility and examine a material interface represented by a cosine function.
# G-ADOPT's exposed implementations use a parametric curve representation, which can
# also describe algebraic curves by identifying the parameter with a domain coordinate.
# Thus, these implemented functions require a parameter to be given; here, it will be
# the x-coordinate. To complete the description of the cosine function, its amplitude,
# wavelength, and shift are also supplied. Finally, we need to close the curve by
# providing coordinates along domain boundaries. The material enclosed in such a way
# will be attributed the 1-side of the conservative level-set profile.

# +
callable_args = (
    curve_parameter := np.linspace(0.0, domain_dims[0], 1000),
    interface_deflection := 0.4,
    perturbation_wavelength := domain_dims[0] / 3.0,
    interface_coord_y := 0.6,
)
boundary_coordinates = [(0.0, domain_dims[1])]

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

# The material interface does correspond to the cosine function we defined.

# As we did earlier, we can swap the location of the 0 and 1 sides of the profile. To do
# so, we need to provide the complementary choice of boundary coordinates.

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=callable_args,
    boundary_coordinates=[(domain_dims[0], 0.0), (0.0, 0.0), (0.0, domain_dims[1])],
)

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# As expected, sides have been swapped.

# Let us now examine the scenario where the material interface is still a curve, but the
# user has to provide its implementation. We choose a
# [Lissajous curve](https://www.wikiwand.com/en/articles/Lissajous_curve), which admits
# a parametric representation. We set $a = 1$, $b = 2$, $\delta = \frac{\pi}{2}$, and
# scale the curve to make it fit within our rectangular domain. This curve is closed; we
# do not need to provide `boundary_coordinates` anymore. However, given the underlying
# `Shapely`` engine, we need to ensure that the first and final points describing the
# curve match. In other words, we need to know the curve's arc length. Here, it is
# $2\pi$, but sometimes it can be non-trivial to determine. And even when known,
# floating-point arithmetic can make the comparison inexact. We will first assume that
# we know this length and then demonstrate how to modify the curve's implementation if
# it is unknown.


# +
def lissajous_curve(t: np.ndarray, a: float, b: float, delta: float) -> np.ndarray:
    """Lissajous curve."""
    curve = np.column_stack((np.sin(a * t + delta) + 1.0, (np.sin(b * t) + 1.0) / 2.0))

    return np.vstack((curve, curve[0]))


callable_args = (
    curve_parameter := np.linspace(0.0, 2 * np.pi, 1000),
    a := 1.0,
    b := 2.0,
    delta := np.pi / 2.0,
)

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable=lissajous_curve,
    interface_args=callable_args,
)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# The interface matches the desired Lissajous curve.

# As promised, we now demonstrate how to modify the curve's implementation when the arc
# length is unknown. We compress the curve horizontally to depict such a situation.


# +
def lissajous_curve(t: np.ndarray, a: float, b: float, delta: float) -> np.ndarray:
    """Lissajous curve."""
    curve = np.column_stack(
        (0.9 * np.sin(a * t + delta) + 1.0, (np.sin(b * t) + 1.0) / 2.0)
    )
    index_stop = np.nonzero((curve[2:, 0] > 1.8999) & (curve[2:, 1] < 0.51))[0][0]

    return np.vstack((curve[: index_stop + 2], curve[0]))


callable_args = (curve_parameter := np.linspace(0.0, 10.0, 1000), a, b, delta)

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="curve",
    interface_callable=lissajous_curve,
    interface_args=callable_args,
)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# The level-set profile is again successfully initialised. Feel free to attempt using
# more complex closed curves to initialise the level-set field.

##### Polygon interface

# So far, we have described the material interface as a planar curve. Another option is
# for the interface to represent a polygon. Here again, G-ADOPT exposes a `Callable` in
# the simple case of a rectangle aligned with coordinate axes. The function requires the
# coordinates of the reference vertex, which is taken as the lower-left vertex, and the
# length of each length (horizontal first, then vertical).

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="polygon",
    interface_callable="rectangle",
    interface_args=(reference_vertex := (0.38, 0.12), edge_sizes := (0.21, 0.88)),
)

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# That interface does follow the shape of a rectangle.

# As was the case for curves, polygons can also be generated via user-defined functions.
# Let's define a random polygon here and see if the material interface can be
# successfully generated. The `Callable` must return a closed loop of polygonal
# vertices.


# +
def random_polygon() -> list[list[float, float]]:
    return [
        [0.6, 0.1],
        [1.3, 0.3],
        [1.3, 0.7],
        [1.1, 0.4],
        [0.3, 0.8],
        [0.1, 0.3],
        [0.6, 0.1],
    ]


assign_level_set_values(
    psi, epsilon, interface_geometry="polygon", interface_callable=random_polygon
)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# Another successful initialisation. Feel free to experiment with your own random
# polygon.

##### Circle interface

# We have already initialised the level-set interface as a circle when we directly
# provided the signed-distance function. As it is simple to generate a circle with
# Shapely, G-ADOPT also exposes a way to describe this geometry via
# `assign_level_set_values`. This time, `interface_coordinates` must be a tuple holding
# the coordinates of the circle's centre and the circle's radius. Let's reproduce the
# earlier circle.

assign_level_set_values(
    psi,
    epsilon,
    interface_geometry="circle",
    interface_coordinates=(circle_centre, circle_radius),
)

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# As expected, we recover the earlier circular geometry, noting that level-set sides are
# swapped because, when using `interface_geometry`, the 1-side is attributed to the
# enclosed material.

##### Shapely interface

# While G-ADOPT provides ways to easily generate some simple interface shapes, it is not
# reasonable to expect that it covers all possible scenarios. For more complex
# scenarios, one is encouraged to create the material interface geometry directly via
# `Shapely` and to provide the resulting object to G-ADOPT via the `interface` argument.
# We demonstrate such an approach here by defining a material interface as an annulus
# indented by a triangle.

# +
triangle = sl.Polygon([[0.61, 0.3], [1.07, 0.3], [1.4, 0.6], [0.61, 0.3]])

ann_centre = (1.1, 0.5)
ann_inner_radius = 0.2
ann_outer_radius = 0.4
ann_inner_circle = sl.Point(ann_centre).buffer(ann_inner_radius)
ann_outer_circle = sl.Point(ann_centre).buffer(ann_outer_radius)
annulus = ann_outer_circle.difference(ann_inner_circle)

interface = annulus.difference(triangle)

assign_level_set_values(psi, epsilon, interface_geometry="shapely", interface=interface)
# -

# + tags=["active-ipynb"]
# plot_level_set(psi)
# -

# What a fancy-looking material interface!