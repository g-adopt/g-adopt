"""Divergence-free near-nullspace enrichment for nearly-incompressible elasticity.

`solenoidal_modes`, `near_incompressible_modes` and the
`NearlyIncompressibleAssembledPC` that hands them to GAMG. Three things are
checked: the generated fields really are divergence-free (and the filter drops
those that are not, on a curved mesh), the combined basis is orthonormal and a
strict superset of the rigid-body modes, and the preconditioner actually cuts
CG iterations against rigid modes alone on a nearly-incompressible operator --
which is the whole reason the class exists -- without tripping GAMG at its
default mode count.

The mechanics operator mirrored here is the u-tangent of the compressible
internal-variable stress:

    a(u, v) = 2 mu (dev eps u : dev eps v) + lam (div u)(div v),  lam = K_bs * K

with `dev` subtracting tr/3 in every dimension exactly as
`InternalVariableApproximation.deviatoric_strain` does.
"""

import firedrake as fd
import numpy as np
import pytest

from gadopt.nullspaces import (
    near_incompressible_modes,
    rigid_body_modes,
    solenoidal_modes,
)
from gadopt.solver_options_manager import (
    gamg_parameters,
    nearly_incompressible_mg_parameters,
)


def _dev(e, dim):
    return e - (1.0 / 3.0) * fd.tr(e) * fd.Identity(dim)


def _rel_div(m):
    m_norm = fd.sqrt(fd.assemble(fd.inner(m, m) * fd.dx))
    div_norm = fd.sqrt(fd.assemble(fd.inner(fd.div(m), fd.div(m)) * fd.dx))
    return float(div_norm / m_norm)


def _box(dim, n=8):
    return fd.UnitSquareMesh(n, n) if dim == 2 else fd.UnitCubeMesh(n, n, n)


def _gram(basis):
    vecs = basis._petsc_vecs
    n = len(vecs)
    G = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = vecs[i].dot(vecs[j])
    return G


@pytest.mark.parametrize("dim", [2, 3])
def test_solenoidal_modes_are_divergence_free_on_affine_mesh(dim):
    # On an affine simplex mesh a low-degree polynomial is reproduced exactly in
    # CG2, so every generated field is divergence-free to machine precision.
    V = fd.VectorFunctionSpace(_box(dim), "CG", 2)
    modes = solenoidal_modes(V, max_degree=2)
    assert modes, "expected some solenoidal modes"
    assert all(_rel_div(m) < 1e-11 for m in modes)


def test_divfree_filter_drops_nonsolenoidal_modes_on_curved_mesh():
    # A curved cubed sphere reproduces only degree-1 fields exactly; the degree-2
    # ones pick up a few percent of divergence and must be filtered out.
    base = fd.CubedSphereMesh(radius=1.22, refinement_level=1, degree=2)
    shell = fd.ExtrudedMesh(base, layers=3, layer_height=1.0 / 3,
                            extrusion_type="radial")
    V = fd.VectorFunctionSpace(shell, "CG", 2)

    unfiltered = solenoidal_modes(V, max_degree=2)
    filtered = solenoidal_modes(V, max_degree=2, divfree_tol=1e-8)

    assert max(_rel_div(m) for m in unfiltered) > 1e-2   # degree-2 leak is real
    assert len(filtered) < len(unfiltered)               # something was dropped
    assert all(_rel_div(m) < 1e-8 for m in filtered)     # survivors are clean


@pytest.mark.parametrize("dim", [2, 3])
def test_near_incompressible_modes_orthonormal_and_superset(dim):
    V = fd.VectorFunctionSpace(_box(dim), "CG", 2)
    n_rigid = len(rigid_body_modes(
        V, rotational=True, translations=list(range(dim)))._petsc_vecs)

    basis = near_incompressible_modes(V, max_degree=1)
    G = _gram(basis)
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-9)   # orthonormal

    # Degree-1 solenoidal fields span the complete linear divergence-free space:
    # 5 modes in 2-D, 11 in 3-D, strictly more than the rigid modes alone.
    n_modes = G.shape[0]
    assert n_modes == (5 if dim == 2 else 11)
    assert n_modes > n_rigid


def _elasticity_iterations(mesh, ratio, pc_python, max_degree=None):
    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    dim = mesh.geometric_dimension
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    mu = fd.Constant(1.0)
    lam = fd.Constant(ratio)
    a = (2 * mu * fd.inner(_dev(fd.sym(fd.grad(u)), dim),
                           _dev(fd.sym(fd.grad(v)), dim))
         + lam * fd.div(u) * fd.div(v)) * fd.dx
    load = [0.0] * dim
    load[-1] = -1.0
    L = fd.inner(fd.as_vector(load), v) * fd.dx
    bc = fd.DirichletBC(V, fd.as_vector([0.0] * dim), 1)   # clamp one face

    params = {"ksp_type": "cg", "ksp_rtol": 1e-8, "ksp_max_it": 5000,
              "ksp_norm_type": "unpreconditioned",
              "pc_type": "python", "pc_python_type": pc_python,
              **gamg_parameters("assembled_")}
    if max_degree is not None:
        params["solenoidal_max_degree"] = max_degree

    uh = fd.Function(V)
    problem = fd.LinearVariationalProblem(a, L, uh, bcs=bc)
    solver = fd.LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    return solver.snes.ksp.getIterationNumber()


def test_pc_reduces_iterations_versus_rigid_modes():
    # The value proposition: on a nearly-incompressible operator the enriched
    # near-nullspace converges in strictly fewer CG iterations than rigid modes
    # alone, and both far fewer than no near-nullspace. Numbers are not asserted
    # exactly (they are GAMG/version dependent); the ordering is the invariant.
    mesh = fd.UnitSquareMesh(32, 32)
    ratio = 1000.0
    none = _elasticity_iterations(mesh, ratio, "firedrake.AssembledPC")
    rigid = _elasticity_iterations(mesh, ratio, "gadopt.RigidBodyAssembledPC")
    incomp = _elasticity_iterations(
        mesh, ratio, "gadopt.NearlyIncompressibleAssembledPC")

    assert incomp < rigid < none
    # A meaningful win, not a rounding difference.
    assert incomp <= 0.8 * rigid


def test_pc_default_does_not_saturate_gamg():
    # The default mode count must leave GAMG a full-rank tentative prolongator;
    # a regression that raised the default degree showed up as DIVERGED_NANORINF
    # at iteration 0. A finite iteration count is the whole assertion.
    its = _elasticity_iterations(
        fd.UnitSquareMesh(48, 48), 100.0,
        "gadopt.NearlyIncompressibleAssembledPC")
    assert 0 < its < 5000


def _elasticity_iterations_mg(hier_mesh, ratio, mg_params):
    V = fd.VectorFunctionSpace(hier_mesh, "CG", 2)
    dim = hier_mesh.geometric_dimension
    u, v = fd.TrialFunction(V), fd.TestFunction(V)
    mu = fd.Constant(1.0)
    lam = fd.Constant(ratio)
    a = (2 * mu * fd.inner(_dev(fd.sym(fd.grad(u)), dim),
                           _dev(fd.sym(fd.grad(v)), dim))
         + lam * fd.div(u) * fd.div(v)) * fd.dx
    load = [0.0] * dim
    load[-1] = -1.0
    L = fd.inner(fd.as_vector(load), v) * fd.dx
    bc = fd.DirichletBC(V, fd.as_vector([0.0] * dim), 1)
    params = {"ksp_type": "fgmres", "ksp_rtol": 1e-8, "ksp_max_it": 2000,
              "ksp_norm_type": "unpreconditioned", **mg_params}
    uh = fd.Function(V)
    problem = fd.LinearVariationalProblem(a, L, uh, bcs=bc)
    solver = fd.LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    return solver.snes.ksp.getIterationNumber()


@pytest.mark.parametrize("coarse", ["gamg", "lu"])
def test_mg_patch_parameters_converge(coarse):
    # The A2 recipe: geometric MG with vertex-star patch smoothers over a
    # (GAMG-with-divergence-free-near-nullspace or LU) coarse solve. On a
    # 2-level hierarchy at a large ratio it must converge -- the flexible
    # coarse="gamg" arm especially, since a naive plain-GAMG coarse diverged
    # there. Convergence, not a specific count, is the invariant.
    mesh = fd.MeshHierarchy(fd.UnitSquareMesh(6, 6), 2)[-1]
    its = _elasticity_iterations_mg(
        mesh, 1000.0, nearly_incompressible_mg_parameters(coarse=coarse))
    assert 0 < its < 2000
