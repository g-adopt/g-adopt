"""`gadopt.sea_level_masks.surface_slope`: the frozen surface slope.

`mask_steepness` divides the steepness by a frozen slope field, so that the
mask transition has a fixed width in arc length instead of a fixed width in sea
level. `surface_slope` builds that field from the initial sea level. The design
is W3 of `NOTES/DESIGN-MARTINEC-3D.md`.

Four things have to hold, and there is one test for each.

1. The value is the tangential gradient. On the 2-D annulus with
   `SL_init = A cos(phi)` the answer is known in closed form,
   `A |sin(phi)| / r`, because `grad(A cos phi) = -A sin(phi) phi_hat / r` is
   already tangential. What is left in the measured error is the interpolation
   error of `cos(phi)` in the finite-element space, and raising the degree of
   `SL_init` from 2 to 3 is what shows that.

2. The radial part of the gradient is removed. A field that depends on an
   angle only, like the one in test 1 and like the B2 bed in test 3, has no
   radial gradient, so for those two fields the projection is the identity and
   neither test can see it. `SL_init = A X[0] = A r cos(phi)` separates the
   two parts: its gradient is the constant vector `A e_x`, whose radial part
   `A cos(phi)` and tangential part `A |sin(phi)|` are both O(1).

3. The value is right in 3-D at the number the benchmark cares about. The B2
   basin of Martinec et al. (2018) has a bed slope of 1.26e-3 at its initial
   coastline, `psi = 24.85` degrees from the basin centre. That slope, together
   with 2.51e-4 for the B1 basin, is the reason the frozen slope exists at all.

4. The field is not on the tape. `mask_steepness` divides by it, so a taped
   slope would put a `1 / |grad SL|^2` term in the adjoint that is not physics
   (`NOTES/DESIGN-SEA-LEVEL.md` section 6).

The 3-D mesh is a very coarse four-region sphere, written by
`generate_selfgrav_sphere.generate` into a session-scoped fixture. The coarse
benchmark meshes (for example `b2_coarse_ar7.msh`, 99 059 cells) are far too
slow for a unit test. Both mesh fixtures need gmsh and skip without it.
"""

import sys
import tempfile
from pathlib import Path

import gadopt  # noqa: F401  (the project rule: import gadopt before firedrake)
import firedrake as fd
import numpy as np
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
)
from pyadjoint.tape import annotate_tape

from gadopt.sea_level_masks import surface_slope
from test_gia_gravity import (  # noqa: F401  (module-level fixture)
    CURVE_RE,
    meshes,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Amplitude of the analytic 2-D sea level `SL_init = A cos(phi)`, in the
#: length unit of the model (the mantle thickness D). The value is arbitrary:
#: `surface_slope` is homogeneous of degree one in `SL_init`, so A cancels out
#: of every relative error below.
A_SEA_LEVEL = 0.3

#: Relative L2 error of the slope on the surface Re that the degree-2 sea level
#: is allowed on this annulus (dr 0.2, 32 azimuthal cells). Measured 2.08e-3.
#: The threshold is the measured value with room for a different quadrature or
#: a regenerated mesh, and the degree-3 check below is what shows that this
#: number is interpolation error and not a wrong formula.
TOL_2D_DEGREE_2 = 5e-3

#: The same for a degree-3 sea level. Measured 4.31e-5, which is 48 times
#: smaller.
TOL_2D_DEGREE_3 = 1e-4

#: Relative L2 error on Re allowed for the radial-projection case,
#: `SL_init = A X[0]`, at degrees 2 and 3 of the sea level. Measured 1.44e-3
#: and 1.25e-6. These thresholds are not a convergence statement: they exist to
#: separate the projected slope from the unprojected one. With the projection
#: removed the same measurement is 6.73e-1 at both degrees, which is 470 times
#: the degree-2 threshold and 67 000 times the degree-3 one.
TOL_RADIAL_DEGREE_2 = 5e-3
TOL_RADIAL_DEGREE_3 = 1e-5

#: The lateral cell spacing of the unit-test sphere, in km, and the lower bound
#: on the number of cells around a great circle. 1000 km gives 19 314 cells in
#: the whole sphere and 13 649 in the mantle, which takes about 10 s to read
#: and to curve. One lithosphere layer, because the test reads a surface
#: gradient and has no use for radial resolution in the lithosphere.
SPHERE_H_KM = 1000.0
SPHERE_MIN_CELLS = 16
SPHERE_LITHO_LAYERS = 1

#: The B2 basin of Martinec et al. (2018), from
#: `~/Workplace/gia-mip/cases/benchmarks/martinec2018/{C,D}.json` through
#: section 1 of `NOTES/DESIGN-MARTINEC-3D.md`. The bed elevation is
#: `zeta0(psi) = BMAX - B0 exp(-psi^2 / (2 sigma_b^2))` in metres, with `psi`
#: the angular distance from the basin centre, and the initial sea level is
#: `SL_init = -zeta0 / D`: positive where the bed is below the geoid, which is
#: where there is water.
BASIN_BMAX_M = 3800.0
BASIN_B0_M = 6000.0
BASIN_SIGMA_DEG = 26.0
#: Colatitude and longitude of the B2 basin centre, in degrees.
BASIN_CENTRE_DEG = (35.0, 25.0)
#: The initial coastline, where `zeta0` changes sign. 24.85 degrees.
BASIN_COAST_DEG = 24.85

#: Depth of the evaluation points below Re, as a fraction of Re. The slope is a
#: discontinuous field and Re is the outer boundary of the mantle mesh, so a
#: point exactly on Re is not reliably located inside a cell: on this very
#: coarse mesh the P2-curved surface leaves the sphere by more than the
#: point-location tolerance. 1e-3 of Re is 6.4 km, which is well inside the
#: 70 km lithosphere layer and therefore inside the cells that touch Re.
EVALUATION_DEPTH = 1e-3

#: Number of longitudes sampled around the B2 coastline.
N_COAST_POINTS = 12

#: The spread over those longitudes that the degree-2 sea level is allowed,
#: relative to the analytic slope. Measured 1.71e-2 peak to peak, with the mean
#: 1.0e-3 below the analytic value. The spread is the anisotropy of an
#: unstructured tetrahedral mesh whose cells are 9 degrees wide against a
#: 26-degree basin; the degree-3 check divides it by 19.
TOL_3D_MEAN_DEGREE_2 = 5e-3
TOL_3D_SPREAD_DEGREE_2 = 3e-2
TOL_3D_SPREAD_DEGREE_3 = 5e-3


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_tape():
    """A fresh tape for every test, and annotation off afterwards.

    The tape test annotates. A test that runs after it must not find a tape
    that still records, because every assemble would then be taped.
    """
    tape = get_working_tape()
    tape.clear_tape()
    yield tape
    if annotate_tape():
        pause_annotation()
    tape.clear_tape()


def sphere_generator():
    """The mesh generator module of the 3-D benchmark, imported by path.

    It lives in a demo directory and not in the package, so it is reached
    through `sys.path` the way the benchmark drivers reach it.
    """
    root = Path(__file__).resolve().parents[2]
    demo = root / "demos" / "glacial_isostatic_adjustment" / "3d_spada_selfgrav"
    if str(demo) not in sys.path:
        sys.path.insert(0, str(demo))
    import generate_selfgrav_sphere as gen

    return gen


@pytest.fixture(scope="session")
def sphere():
    """The mantle submesh of a very coarse four-region sphere, P2-curved.

    A fixed path in the temporary directory and not `tmp_path_factory`: under
    MPI every rank runs its own pytest and would otherwise be handed a
    different directory. The mesh is curved after the `Submesh`, because
    `Submesh` does not inherit the parent's P2 coordinates, and an uncurved
    sphere would put its own O(h^2) geometric error into a test whose whole
    subject is a surface gradient.

    Returns:
      `(mesh, gen)`: the curved mantle submesh and the generator module, whose
      module constants `RE` and `D_KM` the test needs.
    """
    pytest.importorskip("gmsh")
    gen = sphere_generator()

    path = Path(tempfile.gettempdir()) / (
        f"gadopt_slope_sphere_{SPHERE_H_KM:.0f}_{SPHERE_MIN_CELLS}.msh")
    if fd.COMM_WORLD.rank == 0 and not path.exists():
        gen.generate(str(path), h=SPHERE_H_KM / gen.D_KM,
                     litho_layers=SPHERE_LITHO_LAYERS,
                     min_cells_per_great_circle=SPHERE_MIN_CELLS)
    fd.COMM_WORLD.barrier()

    parent = gen.curve_mesh(fd.Mesh(str(path)), name="slope_parent")
    parent.cartesian = False
    # Dimension 3: the mantle is a cell group of a 3-D mesh.
    sub = gen.curve_mesh(fd.Submesh(parent, 3, gen.CELL_MANTLE),
                         name="slope_mantle")
    sub.cartesian = False
    return sub, gen


def basin_sea_level(mesh, gen):
    """`SL_init = -zeta0 / D` of the B2 basin, as a UFL expression on `mesh`.

    The bed elevation `zeta0` depends only on the angular distance `psi` from
    the basin centre, so the expression is valid at every radius of the mesh
    and its gradient is tangential everywhere.

    Args:
      mesh: the mesh whose coordinates the expression reads.
      gen: the generator module, for the length scale `D_KM`.

    Returns:
      A UFL expression, in the non-dimensional length unit D.
    """
    colatitude, longitude = np.radians(BASIN_CENTRE_DEG)
    centre = fd.as_vector((np.sin(colatitude) * np.cos(longitude),
                           np.sin(colatitude) * np.sin(longitude),
                           np.cos(colatitude)))
    X = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(X, X))
    # Clamp the cosine before `acos`: a coordinate of the curved mesh gives a
    # value marginally outside [-1, 1] and `acos` is then NaN.
    cos_psi = fd.max_value(fd.min_value(fd.dot(X, centre) / r, 1.0), -1.0)
    psi = fd.acos(cos_psi)
    sigma = np.radians(BASIN_SIGMA_DEG)
    zeta0 = BASIN_BMAX_M - BASIN_B0_M * fd.exp(-psi**2 / (2.0 * sigma**2))
    return -zeta0 / (gen.D_KM * 1.0e3)


def basin_slope_analytic(radius, gen):
    """The exact surface slope of the B2 bed at the coastline, dimensionless.

    `SL_init = -zeta0 / D` and the arc length at radius `radius` is
    `radius * psi`, so the slope is `(d zeta0 / d psi) / (D * radius)` with

        d zeta0 / d psi = B0 (psi / sigma^2) exp(-psi^2 / (2 sigma^2)).

    Args:
      radius: the non-dimensional radius at which the slope is read. The test
        evaluates a little below Re, and the slope grows as `1 / radius`.
      gen: the generator module, for the length scale `D_KM`.

    Returns:
      The slope at the coastline, about 1.26e-3 at Re.
    """
    psi = np.radians(BASIN_COAST_DEG)
    sigma = np.radians(BASIN_SIGMA_DEG)
    d_zeta_d_psi = BASIN_B0_M * (psi / sigma**2) * np.exp(
        -psi**2 / (2.0 * sigma**2))
    return d_zeta_d_psi / (gen.D_KM * 1.0e3) / radius


def coastline_points(radius):
    """`N_COAST_POINTS` points at the B2 coastline, one per longitude.

    They lie on the small circle of angular radius `BASIN_COAST_DEG` around the
    basin centre, at the given radius. Sampling several of them is what shows
    the anisotropy of the unstructured mesh, which a single point would hide.

    Args:
      radius: the non-dimensional radius of the points.

    Returns:
      An `(N_COAST_POINTS, 3)` array of Cartesian coordinates.
    """
    colatitude, longitude = np.radians(BASIN_CENTRE_DEG)
    centre = np.array([np.sin(colatitude) * np.cos(longitude),
                       np.sin(colatitude) * np.sin(longitude),
                       np.cos(colatitude)])
    # Two unit vectors that span the plane perpendicular to the centre.
    e1 = np.cross(centre, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    psi = np.radians(BASIN_COAST_DEG)
    angles = np.linspace(0.0, 2.0 * np.pi, N_COAST_POINTS, endpoint=False)
    return np.array([
        radius * (np.cos(psi) * centre
                  + np.sin(psi) * (np.cos(a) * e1 + np.sin(a) * e2))
        for a in angles])


def sample(field, points):
    """Read a field at a list of points, through a `VertexOnlyMesh`.

    This is the route the benchmark driver uses for its profiles, and it is
    also the route that still works in parallel, where every rank holds a part
    of the mesh. Only the maximum, the minimum and the mean of the values are
    taken here, so the order of the returned array does not matter.

    Args:
      field: a `Function` on the mesh that contains the points.
      points: an array of coordinates, one row per point.

    Returns:
      A NumPy array of the values, in the order the point cloud stores them.
    """
    cloud = fd.VertexOnlyMesh(field.function_space().mesh(), points,
                              missing_points_behaviour="error")
    at_points = fd.Function(fd.FunctionSpace(cloud, "DG", 0))
    at_points.interpolate(field)
    return np.asarray(at_points.dat.data_ro)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def test_the_annulus_slope_is_the_analytic_tangential_gradient(meshes):  # noqa: F811
    """2-D: `SL_init = A cos(phi)` gives `A |sin(phi)| / r`.

    `grad(A cos phi) = -A sin(phi) phi_hat / r` has no radial part, so the
    analytic answer is the magnitude of the full gradient and the projection
    that `surface_slope` performs must leave it unchanged. The error is
    measured in the relative L2 norm on the surface Re, where the sea-level
    measure of the solver reads the field.

    Two degrees are used, and the second is the point of the test: the
    remaining error must fall when `SL_init` resolves `cos(phi)` better, which
    identifies it as interpolation error.

    Because the gradient here is tangential already, this test says nothing
    about the projection: it passes unchanged when the projection is deleted.
    `test_the_radial_part_of_the_gradient_is_removed` is the test that covers
    it. What this test does cover is the magnitude and the `1 / r` metric
    factor, and a slope built from the radial part alone fails it with a
    relative error near 1.
    """
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)
    phi = fd.atan2(X[1], X[0])
    radius = fd.sqrt(fd.dot(X, X))
    exact = A_SEA_LEVEL * abs(fd.sin(phi)) / radius
    surface = fd.ds(CURVE_RE, domain=sub)
    norm_exact = np.sqrt(float(fd.assemble(exact**2 * surface)))

    errors = {}
    for degree in (2, 3):
        space = fd.FunctionSpace(sub, "CG", degree)
        SL_init = fd.Function(space).interpolate(
            A_SEA_LEVEL * fd.cos(phi))
        slope = surface_slope(SL_init)
        # The default target space is one degree below `SL_init`, which is the
        # degree the gradient has.
        assert slope.function_space().ufl_element().degree() == degree - 1
        errors[degree] = np.sqrt(float(
            fd.assemble((slope - exact)**2 * surface))) / norm_exact

    assert errors[2] < TOL_2D_DEGREE_2, errors
    assert errors[3] < TOL_2D_DEGREE_3, errors


def test_the_radial_part_of_the_gradient_is_removed(meshes):  # noqa: F811
    """2-D: `SL_init = A X[0]` gives `A |sin(phi)|`, not `A`.

    This is the test of the projection. The two analytic sea levels of the
    other two physics tests depend on an angle only, so their gradient is
    tangential already and `surface_slope` returns the same field whether it
    projects or not. `SL_init = A X[0] = A r cos(phi)` is the simplest field
    whose gradient has both parts at O(1):

        grad(A X[0]) = A e_x,   e_x = cos(phi) rhat - sin(phi) phihat,

    so the radial part is `A cos(phi)`, the tangential part is `A |sin(phi)|`
    and the full magnitude is the constant `A`. The exact slope is therefore
    `A |sin(phi)|`, with no `1 / r` factor: the gradient of a linear function
    is constant, and the arc-length metric is already inside it.

    A linear function is in the CG2 space exactly, even on the P2-curved
    cells, so `SL_init` itself carries no interpolation error and the whole
    measured error comes from the discontinuous space that holds the result.
    Two degrees are used, as in the other physics tests, so that the remaining
    error is identified as interpolation error.

    The thresholds are what makes this a test of the projection. With the
    projection the measured errors are 1.44e-3 and 1.25e-6. Without it the
    result is the constant `A` against `A |sin(phi)|`, a relative error of
    6.73e-1 at both degrees, which is more than two orders of magnitude above
    either threshold.
    """
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)
    phi = fd.atan2(X[1], X[0])
    exact = A_SEA_LEVEL * abs(fd.sin(phi))
    surface = fd.ds(CURVE_RE, domain=sub)
    norm_exact = np.sqrt(float(fd.assemble(exact**2 * surface)))

    errors = {}
    for degree in (2, 3):
        space = fd.FunctionSpace(sub, "CG", degree)
        SL_init = fd.Function(space).interpolate(A_SEA_LEVEL * X[0])
        slope = surface_slope(SL_init)
        errors[degree] = np.sqrt(float(
            fd.assemble((slope - exact)**2 * surface))) / norm_exact

    assert errors[2] < TOL_RADIAL_DEGREE_2, errors
    assert errors[3] < TOL_RADIAL_DEGREE_3, errors


def test_the_b2_basin_slope_at_its_coastline_is_the_published_value(sphere):
    """3-D: the B2 bed gives 1.26e-3 at `psi = 24.85` degrees.

    This is the number that section 1 of `NOTES/DESIGN-MARTINEC-3D.md` records
    for the B2 basin, and it is what sets the mask width of cases C and D. The
    slope is read at `N_COAST_POINTS` longitudes around the coastline, a little
    below Re, and compared with the closed-form derivative of the bed profile.

    The mean over the longitudes is the value; the spread over them is the
    anisotropy of an unstructured mesh whose cells are about 9 degrees wide
    against a 26-degree basin. Raising the degree of `SL_init` from 2 to 3 must
    shrink that spread, which identifies it as interpolation error.
    """
    mesh, gen = sphere
    radius = gen.RE * (1.0 - EVALUATION_DEPTH)
    expected = basin_slope_analytic(radius, gen)
    # The published value is quoted at Re. The evaluation points sit 6.4 km
    # below it, where the slope is larger by one part in a thousand.
    assert abs(basin_slope_analytic(gen.RE, gen) - 1.26e-3) < 0.005e-3
    points = coastline_points(radius)

    spreads = {}
    for degree in (2, 3):
        space = fd.FunctionSpace(mesh, "CG", degree)
        SL_init = fd.Function(space).interpolate(basin_sea_level(mesh, gen))
        slope = surface_slope(SL_init)
        got = sample(slope, points)
        spreads[degree] = (got.max() - got.min()) / expected
        if degree == 2:
            mean_error = abs(got.mean() - expected) / expected
            assert mean_error < TOL_3D_MEAN_DEGREE_2, (got.mean(), expected)

    assert spreads[2] < TOL_3D_SPREAD_DEGREE_2, spreads
    assert spreads[3] < TOL_3D_SPREAD_DEGREE_3, spreads


def test_the_slope_is_not_on_the_tape(meshes):  # noqa: F811
    """Nothing is recorded, even when the caller annotates.

    `mask_steepness` divides the steepness by the slope, so a taped slope
    would put the derivative of `1 / |grad SL|` into the adjoint. That term is
    an artefact of the mask width and not physics, so the frozen slope is built
    inside `stop_annotating()`.

    The check is in two parts: the call adds no block to the tape, and the
    tape that a `ReducedFunctional` then holds contains no block that produced
    the slope field.
    """
    _, sub = meshes
    X = fd.SpatialCoordinate(sub)
    phi = fd.atan2(X[1], X[0])
    space = fd.FunctionSpace(sub, "CG", 2)
    tape = get_working_tape()

    continue_annotation()
    try:
        # `SL_init` itself is taped, so the tape is not empty and a block that
        # the slope added would be visible as a change in the count.
        SL_init = fd.Function(space).interpolate(A_SEA_LEVEL * fd.cos(phi))
        blocks_before = len(tape.get_blocks())
        slope = surface_slope(SL_init)
        blocks_after = len(tape.get_blocks())
        functional = fd.assemble(SL_init**2 * fd.dx(domain=sub))
        reduced = ReducedFunctional(functional, Control(SL_init))
    finally:
        pause_annotation()

    assert blocks_before > 0
    assert blocks_after == blocks_before

    outputs = [variable.output
               for block in reduced.tape.get_blocks()
               for variable in block.get_outputs()]
    assert all(output is not slope for output in outputs)
