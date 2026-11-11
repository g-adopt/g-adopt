"""Plate-reconstruction fields for G-ADOPT.

Two things live here. ``GplatesVelocityFunction`` reconstructs surface plate
velocities for a Stokes boundary condition, and ``GplatesScalarFunction``
carries a time-dependent scalar field, such as a lithosphere indicator or a
geotherm, that follows the same reconstruction.

The scalar side is built from three pieces that stay separate on purpose. A
Source says where the reconstructed points are at a geological age and what
they carry (``sources``); an OutputStrategy turns those interpolated values
into the field the model wants (``outputs``); and a ScalarFieldConnector pairs
one of each and manages caching across MPI ranks (``connectors``). The kNN
machinery between them is in ``interpolation``, and ``factories`` assembles the
common combinations, which is where most users should start.
"""

from .connectors import *
from .interpolation import *
from .gplates import *
from .outputs import *
from .sources import *
from .factories import *
