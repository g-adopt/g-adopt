# Spada compatibility with Firedrake 2026.10.0

The supported study environment is Gadi's stable release module,
`module load firedrake/2026.10.0`. New study jobs require that package version before
starting a solve. Existing plans must be regenerated after these changes;
see [SPADA_STUDY.md](SPADA_STUDY.md).

## Container environment

The official image tested is:

```text
firedrakeproject/firedrake:2026.10.0
sha256:241c3e90b7ed160d8d38647bdc7faea14749ad22889536e9de469c2081f36985
```

The amd64 environment contains Python 3.12.3, Firedrake 2026.10.0,
PETSc/petsc4py 3.26.0, UFL 2026.2.0, NumPy 2.5.3, h5py 3.16.0 and VTK
9.7.1. Added dependencies are pinned in `Dockerfile.spada`, including gmsh
4.15.2 and Irksome revision `584f097cdd4a5cbe260d12c9e01ac21a9400b3d4`.
Firedrake and PETSc are used unchanged, with no compatibility shims.
The external HDF5 converter is used unchanged at revision
`ba77f27fcf2d8cc4531e5547489924187c4308af`.

From the repository root, build and run:

```bash
docker build -f tests/gia_selfgrav_benchmarks/Dockerfile.spada \
    -t gadopt-spada:2026.10.0 .
docker run --rm -v "$PWD:/workspace:ro" gadopt-spada:2026.10.0 \
    python3 -m pytest -q -p no:cacheprovider \
    test_spada_loads.py test_spada_configuration.py \
    test_spada_profile_geometry.py test_spada_study.py test_spada_surface_output.py
docker run --rm --shm-size=1g -v "$PWD:/workspace:ro" gadopt-spada:2026.10.0 \
    mpiexec -n 4 python3 -m pytest -q -p no:cacheprovider \
    test_spada_configuration.py test_spada_surface_output.py
```

The Dockerfile reproduces the additional dependencies installed for these
checks; the test environment was assembled from the pinned base image with
those installation commands rather than from a separate Docker build.

## Compatibility fixes

- PETSc 3.26 removes `DM.getDefaultSection()`. Surface restriction now uses
  `DM.getLocalSection()` for both scalar CG node maps. Curved P2 geometry,
  CG3 displacement, edge orientations and empty surface ranks are covered.
- Redundant `VertexOnlyMesh` input ordering lives on rank 0. The shared point
  sampler now broadcasts ordered values so each rank sees the full profile,
  including its found mask. This fixes a non-root shape error in profile
  export. MPI tests also share their output directory and synchronize reads.
- Gadi's `qsub` rejects `-d`. Submission now uses its existing `cwd` and
  `#PBS -l wd`, and reports captured scheduler diagnostics on failure.
- New plans load `firedrake/2026.10.0` and check for package version `2026.10.0`.
  Selecting a different tested environment requires explicit `--module` and
  `--firedrake-version` values when preparing a fresh plan.
- A fluid-core transpose assertion previously required bitwise equality.
  Independent assembly kernels differ by 5.55e-17 on this build. The test
  now permits eight machine epsilons at the block scale; solver equations
  and physical acceptance bounds are unchanged.

## Validation

- All 97 Spada tests pass in serial.
- All 69 configuration and surface tests pass on four MPI ranks.
- The generated version preflight accepts 2026.10.0 and rejects a plan
  requesting 2026.4.2 before any solver or output-directory claim.
- The broader self-gravity, coupled GIA and rotation-moment suite passes
  171 serial tests, with one existing expected failure. Its one parallel
  test passes with an explicit two-rank launch (see below).
- Four complete short solver runs pass on four MPI ranks: cap and disc,
  each with rotation disabled and enabled. Each includes an elastic solve
  and one 100 yr time step, rather than the full default smoke schedule.
- All four runs save two profile epochs with zero missing points. Surface
  archives written on four ranks read back successfully on one rank, retaining
  P2 surface geometry and CG3 displacement. Saved rates agree with field
  differences over the 100 yr interval, total geoid equals gravitational plus
  rotational geoid, and nonrotating runs have zero rotational geoid.
- Surface HDF5-to-VTK conversion succeeds for all four runs, using the
  unchanged external converter. Serial conversion preserves both timestamps
  (0 and 0.1 kyr); a two-rank selected-field conversion of the rotating disc
  preserves its requested final timestamp (0.1 kyr). Displacement and rate
  profile PDFs also render for nonrotating disc and rotating cap outputs.
- A terminal-checkpoint restart of the nonrotating disc run succeeds and
  preserves its archive. No time steps remain in that checkpoint; continued
  integration and restart-frame handling are covered by the unit tests.
- All 31 generated PBS scripts pass `bash -n`.

The broader serial suite can be repeated from the repository root with:

```bash
docker run --rm -v "$PWD:/workspace:ro" -w /workspace gadopt-spada:2026.10.0 \
    python3 -m pytest -q -p no:cacheprovider -m 'not parallel' \
    tests/unit/test_self_gravity_terms.py tests/unit/test_gia_rotation_moments.py \
    tests/unit/test_gia_gravity.py
```

The four short driver checks use the default 1000 km smoke mesh, K/mu=1000,
DtN degree 5, CG3 displacement, and the baseline solver tolerances. They
set `--timestep_scale 10`: an elastic solve and one marched 100 yr step to
0.1 kyr, sufficient to exercise nonzero rates and both saved output epochs.
These are execution checks, not time-accuracy tests. Repeat them with:

```bash
mkdir -p outputs/spada-compatibility
for shape in cap disc; do
    for case in cap polar-motion; do
        docker run --rm --shm-size=1g -v "$PWD:/workspace:ro" \
            -v "$PWD/outputs/spada-compatibility:/checks" gadopt-spada:2026.10.0 \
            mpiexec -n 4 python3 spada.py --case "$case" --load_shape "$shape" \
            --smoke --timestep_scale 10 --write_profiles --write_surface_output \
            --checkpoint_every 1 --output_path "/checks/$shape-$case"
    done
done
```

The local short runs converge at the outer solver level, but inner block-0
solves frequently reach their 200-iteration limit (`DIVERGED_ITS`) on this
four-rank, K/mu=1000 smoke mesh. Thus these checks do not demonstrate that
every configured inner tolerance is attained. The coarse DtN quadrature
also emits its expected warning. Inspect iteration counts and convergence
on Gadi before starting the production matrix.

The Spada tests cover both load representations/shapes, independent rotation
and load location, numerical controls, profile components/rates, LOD export,
surface HDF5/VTK output, restarts, incomplete/superseded frames and conversion
through the external script. The surface archive is read back with fewer
ranks as part of the tests.

The bundled `mpi-pytest` automatic subprocess launcher fails in this Docker
environment after parent-process MPI initialization. The rotation-radius
test passes with an explicit two-rank launch:

```bash
docker run --rm -v "$PWD:/workspace:ro" \
    -e _PYTEST_MPI_CHILD_PROCESS=1 -w /workspace gadopt-spada:2026.10.0 \
    mpiexec -n 2 python3 -m pytest -q -p no:cacheprovider \
    tests/unit/test_gia_rotation_moments.py::test_the_radius_check_agrees_across_ranks
```

These checks establish local API and execution compatibility for the Spada
workflow. They do not establish production-resolution convergence, Gadi
104/1560-rank performance, or the final study's SU cost. The Gadi smoke and
baseline stages remain necessary.
