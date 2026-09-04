# Candidate-aware GAMG aggregate repair

## Motivation

PETSc's tentative GAMG prolongator restricts every supplied near-nullspace
candidate to each aggregate and factorises that local candidate block. A
three-component velocity aggregate containing `n` mesh nodes has `3n` rows.
It cannot represent ten independent conformal candidates unless it has at
least four nodes, and a larger aggregate can still be rank deficient after
boundary restriction or because of its geometry. Zero-padding an undersized
block before QR does not recover the missing rank.

An exact probe of PETSc 3.25's aggregates found the following deficiencies for
ten raw conformal candidates:

| Mesh nodes | MPI ranks | Rank-deficient aggregates |
| ---: | ---: | ---: |
| `9^3` | 1 | 8 / 90 |
| `9^3` | 4 | 17 / 87 |
| `17^3` | 1 | 58 / 532 |
| `17^3` | 4 | 106 / 544 |

This explains why direct ten-mode GAMG can reduce iterations on one case yet
fail during setup or the first velocity solve after refinement, repartitioning,
or an increase in viscosity contrast.

## Proposed PETSc algorithm

Keep the existing MIS or MIS-k graph, filters, smoother, and all other GAMG
defaults. Apply the following repair only at the finest aggregation level and
only when more candidates are supplied than an aggregate can represent:

1. Form each aggregate's boundary-treated candidate block.
2. Compute a rank-revealing TSQR/SVD factorisation and the scale-independent
   ratio `sigma_min / sigma_max`.
3. Mark only aggregates below a configurable conditioning threshold.
4. Merge a marked aggregate with its strongest adjacent aggregate in the
   already filtered graph. Use deterministic integer/topological priorities
   to prevent merge cycles.
5. Exchange only aggregate-neighbour ownership and small candidate factors;
   do not all-gather the graph or candidate vectors.
6. Repeat for a bounded number of rounds, while bounding aggregate diameter.
7. Report a clear setup failure for an isolated connected component that
   cannot span the supplied candidates.
8. Construct the normal ten-candidate tentative prolongator on the repaired
   aggregates, then continue with standard GAMG smoothing and Galerkin levels.

The rank test must be based on the actual candidate values after any strong
boundary treatment. An unpivoted-QR diagonal test is insufficient; formal
rank alone is also insufficient when the smallest singular value is close to
roundoff.

This design adds setup communication but no communication or vector work to
each preconditioner application. Merging deficient roots should partly offset
the wider ten-column prolongator by reducing the first coarse-grid size.

## Prototype evidence

An MPI ownership/merge prototype conserved every node and removed all detected
deficiencies with partition-independent final root counts:

| Case | Initial roots | Repaired roots | Merge rounds |
| --- | ---: | ---: | ---: |
| `17^3`, raw candidates | 729 | 512 | 7 |
| `17^3`, constrained candidates | 729 | 511 | 7 |

Replacing MIS wholesale with heavy-edge matching was not viable: a small case
diverged and the `n=8` setup exceeded 49 seconds and about 1.3 GB before coarse
matrix construction completed. The repair should therefore remain local to
deficient aggregates.

## Required validation

Before this becomes a PETSc or G-ADOPT production option, test:

- TALA and ALA on Cartesian and curved-shell meshes;
- raw and homogeneously constrained candidates;
- viscosity contrasts `1`, `1e4`, `1e6`, and `1e8`;
- frozen, Picard, and modified-Newton velocity operators;
- perturbed and translated meshes;
- serial and 2/4-rank regressions, followed by Gadi weak scaling;
- candidate reproduction error `||V - P V_c|| / ||V||`;
- aggregate singular-value histograms and merge-diameter bounds;
- velocity and pressure iterations, exact residuals, setup/apply time,
  hierarchy complexity, and peak memory.

The repaired-ten method is the preferred production direction. The guarded
six-vector `RitzConformalPC` in this branch is the immediately deployable
comparison arm because it retains the original six-column hierarchy and has
no extra per-application reductions.
