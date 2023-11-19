import firedrake as fd
import ufl
from gadopt.utility import is_continuous, normal_is_continuous, get_functionspace


def assert_continuity_test(V, continuity_test, expected):
    # assert function in functionspace V tests continuous
    assert continuity_test(fd.Function(V)) == expected
    # same check for sub space of mixed space
    W = fd.FunctionSpace(V.mesh(), "CG", 1)
    Z = V * W
    assert continuity_test(fd.Function(Z).sub(0)) == expected


def assert_continuity(V, expected=True):
    assert_continuity_test(V, is_continuous, expected)


def assert_normal_continuity(V, expected=True):
    assert_continuity_test(V, normal_is_continuous, expected)


def test_continuity():
    """Test is_continuous and normal_is_continuous for functionspaces that we use

    Should add tests here if we end up using more exotic functionspaces."""
    mesh1d = fd.UnitIntervalMesh(1)
    meshes = [
        fd.UnitSquareMesh(1, 1, quadrilateral=True),
        fd.UnitSquareMesh(1, 1, quadrilateral=False),
        fd.ExtrudedMesh(mesh1d, 1)
    ]

    for mesh in meshes:
        P1 = fd.FunctionSpace(mesh, "CG", 1)
        P1DG = fd.FunctionSpace(mesh, "DG", 1)
        assert_continuity(P1)
        assert_continuity(P1DG, expected=False)
        if mesh.ufl_cell() == ufl.triangle:
            RT1 = fd.FunctionSpace(mesh, "RT", 1)
            assert_continuity(RT1, expected=False)
            assert_normal_continuity(RT1)
        else:
            DPC1 = fd.FunctionSpace(mesh, "DPC", 1)
            assert_continuity(DPC1, expected=False)
        VP1 = fd.VectorFunctionSpace(mesh, "CG", 1)
        assert_continuity(VP1)
        assert_normal_continuity(VP1)
        VDG1 = fd.VectorFunctionSpace(mesh, "DG", 1)
        assert_continuity(VDG1, expected=False)
        assert_normal_continuity(VDG1, expected=False)

        if not isinstance(mesh.ufl_cell(), ufl.cell.TensorProductCell):
            mesh3d = fd.ExtrudedMesh(mesh, 1)
            V = get_functionspace(mesh3d, "CG", 1, "CG", 1)
            assert_continuity(V)
            assert_normal_continuity(V)
            V = get_functionspace(mesh3d, "CG", 1, "DG", 1)
            assert_continuity(V, expected=False)
            assert_normal_continuity(V, expected=False)
            if mesh.ufl_cell() == ufl.triangle:
                N2_1 = fd.FiniteElement("N2curl", fd.triangle, 1, variant="integral")
                CG_2 = fd.FiniteElement("CG", fd.interval, 2)
                N2CG = fd.TensorProductElement(N2_1, CG_2)
                Ned_horiz = fd.HCurlElement(N2CG)
                P2tr = fd.FiniteElement("CG", fd.triangle, 2)
                P1dg = fd.FiniteElement("DG", fd.interval, 1)
                P2P1 = fd.TensorProductElement(P2tr, P1dg)
                Ned_vert = fd.HCurlElement(P2P1)
                Ned_wedge = Ned_horiz + Ned_vert
                V = fd.FunctionSpace(mesh3d, Ned_wedge)
                assert_continuity(V, expected=False)
                assert_normal_continuity(V, expected=False)
