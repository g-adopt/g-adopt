# Gravity Poisson Equation with Submesh Coupling

## What this directory contains

A minimal working example that solves the gravitational Poisson equation on an extended disc mesh, with density sources restricted to a mantle submesh. The purpose is to validate the Firedrake submesh coupling machinery and the `intersect_measures` mechanism for the gravity equation that will eventually be coupled to viscoelastic GIA simulations.

There are two scripts that form the test:

- `generate_gravity_disc.py` — gmsh mesh generator for a full disc (r=0 to R_grav) with mesh-conforming density shell boundaries.
- `gravity_poisson_test.py` — solves -nabla^2 psi = 4 pi gamma rho and compares against the analytical solution from the `passess` package.

The original scripts by Dale (`submesh_2way_coupling.py` and `unstructured_annulus.py`) are the fully coupled viscoelastic GIA problem that this test is working toward.

## The equation

We solve the 2D gravitational Poisson equation in polar coordinates:

    nabla^2 psi = -4 pi gamma rho

where rho is a density anomaly localised to a thin radial shell inside the mantle, with azimuthal structure cos(m phi). The analytical solution comes from `passess.polar.PoissonPolar2D`, which provides closed-form Green's function convolutions for each azimuthal Fourier mode.

## Mesh geometry

The gmsh mesh is a full disc with five concentric regions separated by mesh-conforming circles:

    r = 0  -->  rmin  -->  r1_shell  -->  r2_shell  -->  rmax  -->  R_grav

- Inner disc (0 to rmin): no sources, no physics of interest.
- Mantle (rmin to rmax): where the Poisson source lives and where we assess the error. Extracted as a Firedrake `Submesh`.
- Exterior (rmax to R_grav): extended domain for the infinite-boundary approximation. R_grav = 10 * rmax.

The density shell [r1_shell, r2_shell] is a 50 km thick layer at 500 km depth (non-dimensionalised by D = rmax - rmin = 1.0, corresponding to 2891 km mantle depth). Its boundaries are explicit circles in the gmsh geometry so that element edges align with them exactly.

## Submesh workflow

The Firedrake submesh construction follows the same pattern as the fully coupled script:

1. Load the full mesh from gmsh.
2. Mark mantle cells with a DG0 indicator function (conditional on rmin <= r <= rmax).
3. `RelabeledMesh(full_mesh, [F_mantle, F_all], [98, 99])` to attach cell labels.
4. `Submesh(mesh, 2, 98)` to extract the mantle as a standalone mesh.

## Cross-mesh variational form

The potential psi lives on the full mesh (CG1), while the density rho lives on the submesh (DG0). To couple them, we use a `MixedFunctionSpace` with a dummy field on the submesh:

    W = V_fullmesh * V_dummy_submesh

with intersect_measures:

    dx_full = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", domain=subm),))
    dx_sub  = Measure("dx", domain=subm, intersect_measures=(Measure("dx", domain=mesh),))

The Laplacian integrates over the full mesh (via dx_full), the source term integrates over the submesh (via dx_sub). The MixedFunctionSpace is necessary because Firedrake's cross-mesh assembly requires it to set up the entity maps between parent and child mesh DOFs.

Firedrake does support cross-mesh coefficient forms without MixedFunctionSpace (there is a test `test_assemble_parent_coefficient` in the submesh test suite), but for solving a coupled variational problem the MixedFunctionSpace route is the tested and reliable path.

## Critical finding: mesh-conforming shell boundaries

When the density shell boundaries do NOT align with element edges, DG0 introduces spurious low-order azimuthal content. For a cos(m phi) density with m=10, even a small m=0 leakage dominates the global error because the m=0 potential decays as ln(r) while the m=10 potential decays as r^{-10}. In early testing this produced 230% relative error despite the solver converging correctly.

The fix is to include the shell radii r1_shell and r2_shell as explicit circles in the gmsh geometry. With mesh-conforming boundaries, every DG0 cell is entirely inside or entirely outside the shell, so the integral of rho over the domain is machine-zero (no m=0 leakage) and the relative L2 error drops to 0.3%.

## Error assessment

We measure the L2 error only within the mantle submesh:

    error = sqrt(integral over mantle of (psi_h - psi_exact)^2 dx)

This is the physically meaningful region. The exterior is just a buffer for the infinite-domain approximation and the potential there is negligible for high azimuthal modes.

Current results with lc_mantle = 0.04, m = 10:

- Relative L2 error over mantle: 0.33%
- Point-wise numerical/analytical ratio: 0.95 to 1.00 throughout the mantle
- integral(rho) = O(1e-18) (machine zero, no m=0 leakage)

## Analytical solutions (passess)

The `passess` package lives at `~/Workplace/passess` and provides `PoissonPolar2D` for 2D polar, `PoissonCartesian2D` for 2D Cartesian, and spherical harmonic solutions for 3D. It is not installed in the Firedrake venv; the test script adds it to sys.path.

The class interface:

    from passess.polar import PoissonPolar2D
    solver = PoissonPolar2D(m=10, rho_m=1.0, r1=r1, r2=r2, gamma=1.0)
    psi_at_r = solver.psi_m(r)          # radial mode coefficient
    psi_spatial = solver.to_spatial(r, phi)  # full spatial field (complex)

For real rho_m the spatial potential is Re[psi_m(r) exp(i m phi)] = psi_m(r) cos(m phi).

For m=0 (axisymmetric) the potential has a gauge freedom; pass r_ref to fix psi(r_ref)=0.

## Non-dimensionalisation

Matches the G-ADOPT 2D cylindrical demo: rmin=1.22, rmax=2.22, D=1.0. Physical mantle depth is 2891 km, so 1 non-dimensional unit = 2891 km. The density shell at 500 km depth, 50 km thick corresponds to:

    r_center = rmax - 500/2891 = 2.047
    r1_shell = r_center - 50/(2*2891) = 2.038
    r2_shell = r_center + 50/(2*2891) = 2.056

## Dependencies

- Firedrake (with gmsh Python bindings: `pip install gmsh` in the venv)
- passess (at ~/Workplace/passess, added to sys.path)
- MUMPS (for the direct LU solver; could switch to iterative)

## What comes next

This standalone Poisson test validates the submesh coupling for the gravity equation in isolation. The next step is coupling it to the momentum and viscoelastic internal variable equations as in Dale's `submesh_2way_coupling.py`, where the density perturbation rho1 = -u dot grad(rho) - rho div(u) comes from the displacement field and the gravitational potential feeds back as a body force -rho grad(psi) in the momentum equation.
