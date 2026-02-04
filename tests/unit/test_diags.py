import firedrake as fd
import gadopt
import gmsh
import tempfile
import pytest

import numpy as np
from typing import Literal
from numbers import Number

takes_boundary_id = ["min", "max", "integral", "l1norm", "l2norm"]
takes_dim = ["min", "max"]
# (diag, boundary, expected_value)
square_2by2_mesh_diag_params = [
    ("min", None, 0),
    ("max", None, 4),
    ("integral", None, 8),
    ("l1norm", None, 8),
    ("l2norm", None, np.sqrt(56 / 3)),
    ("rms", None, np.sqrt(14 / 3)),
    ("min", 1, 0),
    ("min", 2, 2),
    ("min", 3, 0),
    ("min", 4, 2),
    ("max", 1, 2),
    ("max", 2, 4),
    ("max", 3, 2),
    ("max", 4, 4),
    ("integral", 1, 2),
    ("integral", 2, 6),
    ("integral", 3, 2),
    ("integral", 4, 6),
    ("l1norm", 1, 2),
    ("l1norm", 2, 6),
    ("l1norm", 3, 2),
    ("l1norm", 4, 6),
    ("l2norm", 1, np.sqrt(8 / 3)),
    ("l2norm", 2, np.sqrt(56 / 3)),
    ("l2norm", 3, np.sqrt(8 / 3)),
    ("l2norm", 4, np.sqrt(56 / 3)),
]

# Translate the Firedrake square mesh boundary id to the tuples set by the
# "multiple_boundary" gmsh mesh.
equivalent_boundaries = {1: (11, 21), 2: (31, 41), 3: (71, 81), 4: (51, 61), None: None}

annulus_mesh_diag_params = [
    ("min", None, -np.sqrt(8)),
    ("max", None, np.sqrt(8)),
    ("integral", None, 0),
    ("l1norm", None, 28 * np.sqrt(2) / 3),
    ("l2norm", None, np.sqrt(15 * np.pi / 2)),
    ("rms", None, np.sqrt(5 / 2)),
    ("min", "top", -np.sqrt(8)),
    ("min", "bottom", -np.sqrt(2)),
    ("max", "top", np.sqrt(8)),
    ("max", "bottom", np.sqrt(2)),
    ("integral", "top", 0),
    ("integral", "bottom", 0),
    ("l1norm", "top", 4 * np.sqrt(32)),
    ("l1norm", "bottom", np.sqrt(32)),
    ("l2norm", "top", np.sqrt(16 * np.pi)),
    ("l2norm", "bottom", np.sqrt(2 * np.pi)),
]

scalar_diags = []
vector_diags = []
for diag in square_2by2_mesh_diag_params:
    scalar_diags.append(("square",) + diag)
    scalar_diags.append(
        ("multiple_boundary", diag[0], equivalent_boundaries[diag[1]], diag[2])
    )
    if diag[0] in takes_dim:
        vector_diags.append(("square",) + diag)
        vector_diags.append(
            ("multiple_boundary", diag[0], equivalent_boundaries[diag[1]], diag[2])
        )
for diag in annulus_mesh_diag_params:
    scalar_diags.append(("annulus",) + diag)
    if diag[0] in takes_dim:
        vector_diags.append(("annulus",) + diag)


def get_mesh(
    mtype: Literal["square", "annulus", "multiple_boundary"],
    nx: int = 10,
    ny: int = 10,
    L: int = 1,
    rmin: int = 1,
    ncells: int = 2048,
    nlayers: int = 256,
) -> fd.MeshGeometry:
    """Return a square or annulus mesh

    Return a square mesh described by the `nx`, `ny` and `L` args, or an annulus mesh
    described by the `L`, `ncells` and `nlayers` args. The `multiple_boundary` option
    creates an L x L square mesh with two boundary ids on each side.

    In each case, the arguments
    corresponding to the mesh type not selected is ignored.


    Args:
        mtype: Mesh type, "square", "annulus" or "multiple_boundary"
        nx: The number of cells in the x direction. Defaults to 10.
        ny: The number of cells in the y direction. Defaults to 10.
        L: The extent in the x and y directions. Defaults to 1.
        rmin: Radius of inner ring of annulus. Defaults to 1.
        ncells: Number of cells on the initial 1D circle surface mesh. Defaults to 2048.
        nlayers: Number of layers to extruded the 1D circle surface mesh. Defaults to 256.

    Raises:
        ValueError: If mesh is neither "square" nor "annulus"

    Returns:
        Either a square or annulus Firedrake mesh
    """
    match mtype:
        case "square":
            mesh = fd.SquareMesh(nx, ny, L)
            mesh.cartesian = True
        case "annulus":
            mesh1d = fd.CircleManifoldMesh(ncells, radius=rmin, degree=2)
            mesh = fd.ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")
            mesh.cartesian = False
        case "multiple_boundary":
            gmsh.initialize()
            lc = L / nx
            gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
            gmsh.model.geo.addPoint(L / 2, 0, 0, lc, 2)
            gmsh.model.geo.addPoint(L, 0, 0, lc, 3)
            gmsh.model.geo.addPoint(L, L / 2, 0, lc, 4)
            gmsh.model.geo.addPoint(L, L, 0, lc, 5)
            gmsh.model.geo.addPoint(L / 2, L, 0, lc, 6)
            gmsh.model.geo.addPoint(0, L, 0, lc, 7)
            gmsh.model.geo.addPoint(0, L / 2, 0, lc, 8)
            for i in range(1, 9):
                gmsh.model.geo.addLine(i, (i) % 8 + 1, i)
            gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 9)
            gmsh.model.geo.addPlaneSurface([9], 10)
            for i in range(1, 9):
                gmsh.model.geo.addPhysicalGroup(1, [i], 10 * i + 1)
            gmsh.model.geo.addPhysicalGroup(2, [10], 91)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)
            with tempfile.NamedTemporaryFile(suffix=".msh") as f:
                gmsh.write(f.name)
                gmsh.finalize()
                mesh = fd.Mesh(f.name)
            mesh.cartesian = True
        case _:
            raise ValueError(f"Don't know {mtype} mesh")
    return mesh


def test_unregistered_function():
    """
    Ensure that a function that has not been registered with a diagnostic object
    raises a Key Error if diagnostics are requested on that function.
    """
    mesh = get_mesh("square")
    F = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(F)
    g = fd.Function(F)
    diags = gadopt.BaseDiagnostics(quad_degree=4, f=f)
    with pytest.raises(KeyError):
        diags.integral(g)


def test_mixed_function():
    """
    Ensure that, if a mixed function is passed to a diagnostic object, the
    subfunctions are stored with the correct <name>_<index> attribute name and that
    attempting to access the mixed function raises an AttributeError
    """
    mesh = get_mesh("square")
    F = fd.FunctionSpace(mesh, "CG", 1)
    G = fd.FunctionSpace(mesh, "CG", 1)
    FG = fd.MixedFunctionSpace((F, G))
    fg = fd.Function(FG)
    diags = gadopt.BaseDiagnostics(quad_degree=4, fg=fg)
    assert hasattr(diags, "fg_0")
    assert hasattr(diags, "fg_1")
    with pytest.raises(AttributeError):
        diags.fg


def test_component_of_scalar():
    """
    Ensure that a attempting to retrieve a component of a scalar function raises a
    TypeError
    """
    mesh = get_mesh("square")
    F = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(F)
    diags = gadopt.BaseDiagnostics(quad_degree=4, f=f)
    with pytest.raises(TypeError):
        diags._check_dim_valid(f)


@pytest.mark.parametrize(
    "mesh_type,boundary_id",
    [("square", "top"), ("annulus", 1), ("multiple_boundary", 1)],
)
def test_unknown_boundary(
    mesh_type: Literal["square", "annulus", "multiple_boundary"], boundary_id: int | str
):
    """
    Ensure that a KeyError is raised when diagnostics are requested on a non-existent
    boundary ID.

    Args:
        mesh_type: The mesh type to use (`square`, `annulus` or `multiple_boundary`)
        boundary_id: The boundary ID to test - note, should not exist on the mesh
    """
    mesh = get_mesh(mesh_type)
    F = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(F)
    diags = gadopt.BaseDiagnostics(quad_degree=4, f=f)
    with pytest.raises(KeyError):
        diags._check_boundary_id(f, boundary_id)


def test_function_extraction():
    """
    Test the ability of _extract_functions to extract the base Firedrake function
    from an arbitrary UFL Form.
    """
    mesh = get_mesh("annulus")
    F = fd.VectorFunctionSpace(mesh, "CG", 1)
    f = fd.Function(F)
    vc = gadopt.utility.vertical_component(f)
    assert gadopt.diagnostics.extract_functions(vc) == {f}


@pytest.mark.parametrize(
    "mesh_type,diag_name,boundary_id,analytical_soln",
    scalar_diags,
    ids=["_".join([str(j) for j in i[:3]]) for i in scalar_diags],
)
def test_analytical_scalar(
    mesh_type: Literal["square", "annulus", "multiple_boundary"],
    diag_name: str,
    boundary_id: tuple[int, int] | int | str,
    analytical_soln: Number,
):
    """
    Test the diagnostic function `diag_name` on the domain described by `mesh_name`
    on the boundary `boundary_id`. The output of the diagnostic function is compared
    to the analytical solution for that diagnostic on the function:

    $$
        f(x,y) = x + y
    $$

    Args:
        mesh_type: The mesh type to use (`square`, `annulus` or `multiple_boundary`)
        diag_name: The diagnostic to check.
        boundary_id: The boundary ID to test, where `None` indicates the entire domain
        analytical_soln: The expected solution for the diagnostic tested
    """
    mesh = get_mesh(mesh_type, L=2)
    F = fd.FunctionSpace(mesh, "CG", 1)
    f = fd.Function(F)
    diags = gadopt.BaseDiagnostics(quad_degree=4, f=f)
    X, Y = fd.SpatialCoordinate(mesh)
    f.interpolate(X + Y)
    diag = getattr(diags, diag_name)
    # Tolerances need to be a bit slacker for the annulus mesh
    match mesh_type:
        case "square":
            rtol = 1e-8
            atol = 1e-11
        case "multiple_boundary":
            rtol = 1e-8
            atol = 1e-11
        case "annulus":
            rtol = 1e-6
            atol = 1e-5
    if diag_name in takes_boundary_id:
        np.testing.assert_allclose(
            diag(f, boundary_id), analytical_soln, rtol=rtol, atol=atol
        )
    else:
        np.testing.assert_allclose(diag(f), analytical_soln, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "mesh_type,diag_name,boundary_id,analytical_soln",
    vector_diags,
    ids=["_".join([str(j) for j in i[:3]]) for i in vector_diags],
)
def test_analytical_vector(
    mesh_type: Literal["square", "annulus", "multiple_boundary"],
    diag_name: str,
    boundary_id: tuple[int, int] | int | str,
    analytical_soln: Number,
):
    """
    Test the diagnostic function `diag_name` on the domain described by `mesh_name`
    on the boundary `boundary_id` for the y component of a vector function. The vector
    function is defined as

    $$
        f(x,y) = [ xy, x + y ]
    $$

    Args:
        mesh_type: The mesh type to use (`square`, `annulus` or `multiple_boundary`)
        diag_name: The diagnostic to check.
        boundary_id: The boundary ID to test, where `None` indicates the entire domain
        analytical_soln: The expected solution for the diagnostic tested
    """
    mesh = get_mesh(mesh_type, L=2)
    F = fd.VectorFunctionSpace(mesh, "CG", 2)
    f = fd.Function(F)
    diags = gadopt.BaseDiagnostics(quad_degree=4, f=f)
    X, Y = fd.SpatialCoordinate(mesh)
    f.interpolate(fd.as_vector([X * Y, X + Y]))
    diag = getattr(diags, diag_name)
    # Tolerances need to be a bit slacker for the annulus mesh
    match mesh_type:
        case "square":
            rtol = 1e-8
            atol = 1e-11
        case "multiple_boundary":
            rtol = 1e-8
            atol = 1e-11
        case "annulus":
            rtol = 1e-6
            atol = 1e-5
    if diag_name in takes_boundary_id:
        np.testing.assert_allclose(
            diag(f, boundary_id, dim=1), analytical_soln, rtol=rtol, atol=atol
        )
    else:
        np.testing.assert_allclose(diag(f, dim=1), analytical_soln, rtol=rtol, atol=atol)
