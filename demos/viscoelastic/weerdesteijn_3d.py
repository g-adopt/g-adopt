# 2d box model based on weerdesteijn et al 2023

from gadopt import *
from weerdesteijn_2d import Weerdesteijn2d


class Weerdesteijn3d(Weerdesteijn2d):
    name = "weerdesteijn_3d"
    vertical_component = 2

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def setup_surface_mesh(self):
        return Mesh("./aspect_box_refined_surface.msh", name="surface_mesh")

    def initialise_r(self):
        return pow(pow(X[0], 2) + pow(X[1], 2), 0.5)

    def setup_bcs(self):
        super().setup_bcs()
        self.stokes_bcs[self.bottom_id] = {'uz': 0},
        self.stokes_bcs[3] = {'uy': 0},
        self.stokes_bcs[4] = {'uy': 0},


if __name__ == "__main__":
    simulation = Weerdesteijn3d()
    simulation.run_simulation()
