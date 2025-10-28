"""This module provides helper functions for plotting, associated with the
glacial isostatic adjustment demos, via Pyvista. This includes plotting
'ice rings' showing ice thickness offset from the surface of
the domain. This module also default options for plotting viscosity
fields and animations with artificially warped meshes based on the
displacement field.

"""
from firedrake import atan2, conditional, pi, tanh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista.core.pointset import PolyData
from pyvista.core.utilities.reader import PVDReader
from pyvista.plotting.plotter import Plotter
import ufl

# camera settings
radius = 2.2
zoom = 4.25
lw = 5


def ice_sheet_disc(
        X: ufl.geometry.SpatialCoordinate,
        disc_centre: float,
        disc_halfwidth: float,
        radius: float = 6371e3,
        surface_dx_smooth: float = 200e3
) -> ufl.core.expr.Expr:
    '''Initialises ice sheet disc with a smooth transition given
    by a tanh profile

    Args:
      X:
        Spatial coordinate associated with the mesh
      disc_centre:
        centre of disc in radians (assumed to be between -pi to pi)
      disc_halfwidth:
        Half width of disc in radians
      radius:
        Radius of domain in m
      surface_dx_smooth:
        characteristic length scale for tanh smoothing in m

    Returns:
      Expression for ice sheet disc with tanh smoothing
    '''

    # Setup lengthscales for tanh smoothing
    ncells_smooth = 2*pi*radius / surface_dx_smooth
    surface_resolution_radians_smooth = 2 * pi / ncells_smooth

    # Colatitude defined between -pi -> pi radians with zero at 'north pole'
    # and -pi / pi transition at 'south pole' (x,y) = (0, -R)
    colatitude = atan2(X[0], X[1])

    # Position opposite disc centre in radians
    opp = disc_centre - pi if disc_centre >= 0 else disc_centre + pi

    # Angular distance accounting for discontinuity at 'south pole'
    angular_distance = conditional(abs(colatitude - disc_centre) < pi,
                                   abs(colatitude-disc_centre),
                                   pi - abs(colatitude-opp)
                                   )

    arg = angular_distance - disc_halfwidth
    disc = 0.5*(1-tanh(arg / (2*surface_resolution_radians_smooth)))
    return disc


def make_ice_ring(
        reader: PVDReader,
        domain_depth: float = 2891e3,
) -> PolyData:
    """Create a ring of ice thickness outside an annulus domain

    Args:
      reader:
        Pyvista reader that contains ice thickness data
      domain_depth:
        Depth of domain in m used to re-dimensionalise ice thickness

    Returns:
      Ice thickness transformed to surface mesh
    """

    data = reader.read()[0]
    data['Ice thickness'] *= domain_depth  # Convert ice thickness to m

    # Isolate data on surface of sphere
    surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                      feature_edges=False, manifold_edges=False)
    sphere = pv.Sphere(radius=0.8*radius)
    clipped_surf = surf.clip_surface(sphere, invert=False)

    # Stretch line by 15%
    stretch = 1.15
    transform_matrix = np.array(
        [
            [stretch, 0, 0, 0],
            [0, stretch, 0, 0],
            [0, 0, stretch, 0],
            [0, 0, 0, 1],
        ])
    transformed_surf = clipped_surf.transform(transform_matrix, inplace=True)
    return transformed_surf


def plot_ice_ring(
        plotter: Plotter,
        fname: str = 'ice.pvd',
        scalar: str = "Ice thickness"):
    """Add an ice ring to Pyvista plot

    Args:
      plotter:
        Pyvista plotter
      fname:
        Name of VTK pvd file to read input data from
      scalar:
        Name of scalar field to plot
    """
    ice_reader = pv.get_reader(fname)
    ice_ring = make_ice_ring(ice_reader)
    ice_cmap = plt.get_cmap("Blues", 25)
    ice_lw = 20

    # add outline of ice ring
    plotter.add_mesh(ice_ring, color='black', line_width=ice_lw+2, lighting=False,
                     show_scalar_bar=False)

    # plot ice ring
    plotter.add_mesh(
        ice_ring,
        scalars=scalar,
        line_width=ice_lw,
        cmap=ice_cmap,
        clim=[0, 2000],
        scalar_bar_args={
            "title": 'Ice thickness (m)',
            "position_x": 0.05,
            "position_y": 0.3,
            "vertical": True,
            "title_font_size": 20,
            "label_font_size": 16,
            "fmt": "%.0f",
            "font_family": "arial",
        }
    )


def plot_viscosity(
        plotter: Plotter,
        fname: str = 'viscosity.pvd',
        viscosity_scale: float = 1e21):
    """Add viscosity field to Pyvista plot

    Args:
      plotter:
        Pyvista plotter
      fname:
        Name of VTK pvd file to read input data from
      viscosity_scale:
        Characteristic viscosity scale used to re-dimensionalise the viscosity field
    """
    reader = pv.get_reader(fname)
    data = reader.read()[0]
    data['viscosity'] *= viscosity_scale  # Convert viscosity to Pa s

    # add outline of domain
    surf = data.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                      feature_edges=False, manifold_edges=False)
    plotter.add_mesh(surf, color='black', line_width=lw,
                     lighting=False, show_scalar_bar=False)

    inferno_cmap = plt.get_cmap("inferno_r", 25)

    plotter.add_mesh(
        data,
        component=None,
        lighting=False,
        show_edges=False,
        cmap=inferno_cmap,
        clim=[1e20, 1e25],
        log_scale=True,
        scalar_bar_args={
            "title": 'Viscosity (Pa s)',
            "position_x": 0.8,
            "position_y": 0.3,
            "vertical": True,
            "title_font_size": 20,
            "label_font_size": 16,
            "fmt": "%.0e",
            "font_family": "arial",
            "n_labels": 6,
        },
        show_scalar_bar=True
    )


def plot_animation(
        plotter: Plotter,
        fname: str = 'output.pvd',
        domain_depth: float = 2891e3,
        dt_out_years: float = 1e3):
    """Artifically warp mesh based on displacement and create a gif

    Args:
      plotter:
        Pyvista plotter
      fname:
        Name of VTK pvd file to read input data from
      domain_depth:
        Depth of domain in m used to re-dimensionalise displacement
      dt_out_years:
        Time step duration in years of simulation outputs
    """

    reader = pv.get_reader(fname)
    data = reader.read()[0]

    plotter.open_gif("displacement_warp.gif")

    inferno_cmap = plt.get_cmap("inferno_r", 25)

    # Make a list of output times (non-uniform because also
    # outputing first (quasi-elastic) solve
    times = [0]
    for i in range(len(reader.time_values)):
        times.append((i+1)*dt_out_years)

    for i in range(len(reader.time_values)):
        print("Step: ", i)
        reader.set_active_time_point(i)
        data = reader.read()[0]

        # Artificially warp the output data by the displacement
        # Note the mesh is not really moving!
        warped = data.warp_by_vector(vectors="displacement", factor=1500)
        arrows = warped.glyph(orient="velocity", scale="velocity", factor=5e4, tolerance=0.01)
        plotter.add_mesh(arrows, color="grey", lighting=False)

        data['displacement'] *= domain_depth  # Convert displacement to m
        # Add the warped displacement field to the frame
        plotter.add_mesh(
            warped,
            scalars="displacement",
            component=None,
            lighting=False,
            clim=[0, 600],
            cmap=inferno_cmap,
            scalar_bar_args={
                "title": 'Displacement (m)',
                "position_x": 0.85,
                "position_y": 0.3,
                "vertical": True,
                "title_font_size": 20,
                "label_font_size": 16,
                "fmt": "%.0f",
                "font_family": "arial",
            }
        )

        plotter.camera_position = [(0, 0, radius*5),
                                   (0.0, 0.0, 0.0),
                                   (0.0, 1.0, 0.0)]

        plotter.add_text(f"Time: {times[i]:6} years", name='time-label')

        if i == 0:
            plot_ice_ring(plotter, scalar="zero")
            for j in range(10):
                plotter.write_frame()

        plot_ice_ring(plotter)

        # Write end frame multiple times to give a pause before gif starts again!
        for j in range(10):
            plotter.write_frame()

        if i == len(reader.time_values)-1:
            # Write end frame multiple times to give a pause before gif starts again!
            for j in range(20):
                plotter.write_frame()

        plotter.clear()
