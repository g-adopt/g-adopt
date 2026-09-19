# The Spada et al. (2011) benchmark in 3-D

This directory contains the self-gravitating glacial isostatic adjustment
(GIA) benchmark of Spada et al. (2011) for G-ADOPT.

Spada, G., et al. (2011), A benchmark study for glacial isostatic adjustment
codes, Geophys. J. Int. 185(1), 106-132.

## The benchmark

The benchmark compares GIA codes with the normal-mode code TABOO. The Earth
model is M3-L70-V01. It has a 70 km elastic lithosphere, two upper-mantle
layers, a lower mantle and an inviscid core. The load is a parabolic ice cap
with a thickness of 1500 m and a half width of 10 degrees. The load starts at
t = 0 as a step and stays constant.

The driver `spada_benchmark.py` solves the mechanics, the gravitational
potential and, in one case, the rotation in one coupled system. It has two
cases:

- `--case cap`: The cap is at the north pole and rotation is off. The driver
  compares the surface radial displacement U, the horizontal displacement V
  and the geoid N with TABOO, degree by degree.
- `--case polar-motion`: The cap is at colatitude 25 degrees and longitude
  75 degrees, and rotation is on. The driver compares the polar motion
  (m_x, m_y), its magnitude |m| and its phase with the TABOO series of test
  T02-03, without the Chandler wobble.

The two cases need separate runs. Rotational feedback adds a degree-2, order-1
signal to U, V and N. Thus a run with rotation does not agree with the cap
reference.

The module docstring of `spada_benchmark.py` describes the Earth model, the
scales, the discretisation and the exterior gravity condition.

## Files

| File | Content |
|---|---|
| `spada_benchmark.py` | The driver for both cases. |
| `generate_selfgrav_sphere.py` | The gmsh mesh generator and the P2 mesh curving. |
| `reference_state.py` | The density layers, the reference gravity and the constants of M3-L70-V01. |
| `taboo_synthesis.py` | The synthesis of U, V and N from the TABOO Love-number spectra. |
| `reference.npz` | The TABOO spectra and the polar-motion series. |
| `run_spada_benchmark.pbs` | The PBS job script for the NCI Gadi cluster. |

`reference.npz` contains data from the supplementary material of Spada et al.
(2011). A script extracted the data. That script is in the git history at
commit `a8df4939`, as `extract_reference.py`.

## Make the mesh

The mesh generator needs the `gmsh` Python module. The benchmark driver does
not need it. Run this command in this directory:

```bash
python generate_selfgrav_sphere.py --configuration coarse \
    --litho-layers 1 --min-cells 32 --output b2_coarse_ar7.msh
```

The validated run used a mesh with 99 059 tetrahedra. The command gave
98 709 tetrahedra, with 64 751 in the mantle, on a different gmsh installation.
The lateral cell size is 500 km and the lithosphere has one 70 km layer. The
command takes a few seconds.

gmsh does not always give the same tetrahedra on different installations. A
new mesh can have a small difference in cell count from the mesh of the
validated run.

## Run the benchmark

If G-ADOPT is not installed, put the repository root on `PYTHONPATH`.

To check an installation, run the dry run. The dry run builds the meshes, the
load, the solver and the reference. It assembles the residual one time and
stops before the solve:

```bash
python spada_benchmark.py --case cap --dry-run
python spada_benchmark.py --case polar-motion --dry-run
```

To run a case locally with MPI, use these commands:

```bash
mpiexec -np 96 python spada_benchmark.py --case cap
mpiexec -np 96 python spada_benchmark.py --case polar-motion
```

The default values are the values of the validated run. The displacement is
CG3 and the internal variables are DG2. The load has degrees up to 10, and the
DtN condition is `SphericalDtN(5)`. The bulk/shear ratio is 100. The epochs
are 0, 0.1, 1, 2, 5, 10 and 20 kyr. The option `--help` shows all options.

A full run needs a cluster. The validated cap run to 20 kyr used 96 ranks,
about 6 h of wall time and 106 GB of memory. A laptop is sufficient for the
dry run only.

To run a case on a PBS cluster, submit the job script from this directory:

```bash
qsub run_spada_benchmark.pbs
qsub -v CASE=polar-motion run_spada_benchmark.pbs
```

The job script is for NCI Gadi. For a different cluster or project, edit the
project, the storage, the queue and the module lines.

## Output

- `spada-<label>.h5`: A Firedrake `CheckpointFile` with the two meshes. It
  has the displacement, the potential and the internal variables at each
  epoch. The label is the case name by default.
- `spada-<label>-mechanics.pvd` and `spada-<label>-potential.pvd`: VTK files
  for Paraview. The driver writes them only with `--vtk`.
- `params-<label>.log`: one line for each epoch, with a header line of column names.
  Each line has the checkpoint index, the epoch in kyr and the summary
  quantities of the case with their TABOO values. The cap case gives U(0),
  N(0), the maximum of V in metres and the colatitude of the peak in degrees.
  The polar-motion case gives m_x, m_y, |m| and the phase in degrees. The
  values are in full precision. A dry run writes no log file.
- Standard output: a comparison table at each epoch and a summary table at
  the end.

## Expected results

A run of the cap case to 20 kyr gave these ratios of model to TABOO. That run
used an earlier solver configuration with the same physics and the same
discretisation. The internal variables were eliminated pointwise (the
condensed layout). The DtN condition used one multiplier for each harmonic
mode. No run from t = 0 with the current defaults exists yet.

| t (kyr) | U(0) | N(0) | max V |
|---:|---:|---:|---:|
| 0 | 1.0251 | 0.9971 | 0.9641 |
| 0.1 | 1.0192 | 0.9972 | 0.9690 |
| 1 | 1.0032 | 0.9997 | 0.9803 |
| 2 | 0.9992 | 1.0027 | 0.9820 |
| 5 | 0.9992 | 1.0054 | 0.9862 |
| 10 | 1.0011 | 1.0025 | 0.9900 |
| 20 | 1.0017 | 0.9989 | 0.9916 |

At 20 kyr, the ratios for degrees 2 to 10 are 1.0004 to 1.0028 for U, 0.9888
to 0.9926 for V and 0.9967 to 1.0030 for N. The degree-10 geoid ratio is
1.0030. The model is compressible with a bulk/shear ratio of 100, and the
reference is incompressible. This difference causes a part of the remaining
error.

The evidence for the current defaults is from two 500 yr restart steps from
the 20 kyr state of that run. The full layout differed from the condensed
layout by 5e-7 relative in the displacement norm. The low-rank DtN
representation agreed with the multiplier representation to 4e-11. The restart
steps started from a history field that was transferred between layouts, so
they do not replace a run from t = 0.

For the first run with the defaults, run the cap case with `--epochs 0 0.1`.
Compare the result with the 0 kyr and 0.1 kyr rows of the table.

For the polar-motion case, the reference phase is exactly -105 degrees at all
epochs. The next section gives the results for |m|.

## Polar motion and the moment difference C - A

The reference of test 3/2 is not consistent with itself. It uses the moment
difference C - A = 2.63e35 kg m^2 in the load excitation of its eq. (31). It
uses the secular Love number k_s = 0.96672389 in the transfer function of its
eq. (7). A hydrostatic figure with that k_s has C - A = 2.6952e35 kg m^2. The
two values differ by 2.4 percent.

The solver has one C - A, and its rotational feedback is the physical
Q k_T(t), with Q = a^5 Omega^2 / (3 G). Therefore one option cannot match both
halves of the reference at the physical rotation rate. The option
`--c-minus-a` selects the half that the run matches:

| `--c-minus-a` | C - A (kg m^2) | Omega^2 | the run matches |
|---|---|---|---|
| `ks` (default) | 2.6952e35 | physical | the transfer function of the reference |
| `prescribed` | 2.63e35 | physical | the excitation of the reference |
| `taboo` | 2.63e35 | 0.97580637 of physical | both, and thus the numbers of TABOO |

Q is proportional to Omega^2. The factor 0.97580637 decreases Q from
2.78798e35 to 2.72056e35 kg m^2, which is 2.63e35 / k_s. The pair (C - A, k_s)
is then consistent, and the solver closes the polar motion with the same
equation as the reference.

If you compare the code with the published numbers of TABOO, use
`--c-minus-a taboo`. If you model a rotating Earth, use the default `ks`.
The option `taboo` reproduces a published inconsistency and is not a model of
the Earth.

The factor on Omega^2 also decreases the centrifugal potential that the
mechanics feels. The polar motion and its phase are correct for the reference,
because the factor is part of the closure.

The mechanics feels the product of Omega^2 and m. The factor decreases Omega^2
and increases m by the same factor, so the product does not change. The
displacement, the potential and the geoid in a `taboo` checkpoint are equal to
the values of a default `ks` run to about 4e-6. They are 2.4 percent below the
values that the physical Omega^2 gives with the m that this mode reports. Do
not multiply these fields by 1 / 0.97580637.

Three runs at t = 0 gave |m| against TABOO: 0.974382 with `ks`, 1.010012 with
`prescribed` and 0.998537 with `taboo`. All three used 96 ranks, the mesh
`b2_coarse_ar7.msh`, an outer tolerance of 1e-9 and a block-0 tolerance of
1e-7. A run to 20 kyr with `ks` and the default tolerances gave 0.9744 at
t = 0 and 0.9682 at 20 kyr.

The remaining difference of 0.15 percent at t = 0 is the model's own error:
its elastic tidal Love number is 0.107 percent above the value of TABOO.
