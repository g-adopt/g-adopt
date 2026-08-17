"""Shared configuration for the connector regression fixture.

Imported by BOTH ``tests/unit/test_connectors.py`` (which loads the pickled
reference) and ``generate_expected_connectors.py`` (which produces it). The
reference values are computed from these numbers, so the two sides must agree
exactly — a disagreement would compare the test against a fixture built for a
different configuration. One definition, read by both, removes that hazard.
"""

# Plate-model oldest age, and the cloud point counts for the two producers.
OLDEST_AGE = 120
LITH_TRACKER_POINT_COUNT = 2000
POLYGON_BACKGROUND_POINT_COUNT = 3000
POLYGON_SCALAR_INPUT_POINT_COUNT = 3000

# Ages (Ma) at which the reference quantities are recorded and compared.
TEST_AGES = (100, 50, 0)

# Radial layer heights, bottom (CMB) to top (surface), summing to 1.0.
#
# Graded rather than uniform. The indicator outputs are one-sided steps whose
# base tracks a lithospheric thickness of order 100 km, i.e. 0.035 in mesh
# units. With four uniform 0.25 layers no node ever falls inside the
# lithosphere: the field reads exactly 1 at the surface node and exactly 0 at
# the next node down at every age, so the reduced integrals do not move with
# the reconstruction and the regression asserts nothing. These heights put
# nodes at roughly 0, 29, 87 and 202 km depth, which straddle the base and
# make the integrals genuinely age-sensitive.
REGRESSION_LAYER_HEIGHTS = (0.60, 0.25, 0.08, 0.04, 0.02, 0.01)
