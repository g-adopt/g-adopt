# PETSc Submesh and FieldSplit Support

## Overview

This document describes how PETSc supports the cross-mesh assembly and solving that Firedrake relies on for the gravity coupling problem. The key PETSc components are: DMPlex submesh creation, MatNest for block-structured matrices, Section-based DOF management, and the FieldSplit preconditioner.

---

## 1. DMPlex Submesh Infrastructure

**Location:** `petsc/src/dm/impls/plex/plexsubmesh.c` (~198KB)

PETSc provides comprehensive submesh support through DMPlex:

### Core Functions

- **`DMPlexCreateSubmesh()`**: Extracts a hypersurface from a mesh using vertices marked by a DMLabel. Produces a submesh with one dimension lower than the parent.
- **`DMPlexGetSubpointMap()`**: Returns a DMLabel mapping original points in the submesh to their depth, enabling bidirectional tracking between submesh and parent mesh points.
- **`DMPlexCreateCohesiveSubmesh()`**: Specialised for extracting submeshes from meshes with cohesive elements.

### Implementation Details

- Submesh points are filtered from the parent mesh using point selection labels.
- The subpoint map tracks both cell indices and vertex indices with coordinate transformations.
- `PetscSF` (Scatter-Gather Forests) handle inter-partition communication for distributed submeshes.
- Submeshes maintain separate DMLabel hierarchies tracking depth information.

---

## 2. Section-Based DOF Management

**Location:** `petsc/src/dm/impls/plex/plexsection.c`

PETSc Sections define how DOFs are distributed over mesh entities:

- `PetscSectionSetNumFields()` and `PetscSectionSetFieldDof()` handle multi-field DOF assignment.
- Per-point DOF specification at different mesh depths (vertices, edges, faces, cells).
- `PetscSectionSetFieldComponents()` for field-specific DOF separation.

### Cross-Mesh Relevance

- Sections can be created independently for each mesh's DOF space.
- `DMPlexCreateSectionFields()` sets up multi-field layouts.
- `DMPlexCreateSectionDof()` handles depth-stratified DOF assignment.
- Each submesh has its own Section, and the global system combines them via index sets.

---

## 3. MatNest Matrix Type

**Location:** `petsc/src/mat/impls/nest/matnest.c`

MatNest is the primary PETSc mechanism for block-structured matrices with heterogeneous blocks.

### Design

```
MatCreateNest(comm, nr, is_row, nc, is_col, sub_matrices, &A)
```

- Row index sets `is_row[i]` define DOFs for each field/block.
- Column index sets `is_col[j]` define DOF ownership.
- Sub-matrices `m[i][j]` can have completely different row/column sizes.

### Heterogeneous Index Set Support

- Each row block i uses `isglobal.row[i]` which can have **any size**.
- Each column block j uses `isglobal.col[j]` which can have **any size**.
- No requirement for index sets to be contiguous or related.
- This is what enables assembly of blocks living on different meshes with different DOF counts.

### Key Operations

- `MatNestGetSize()`: Returns number of blocks and their individual dimensions.
- `MatNestGetISs()`: Retrieves the index sets that define block boundaries.
- `MatNestSetSubMat()`: Updates individual block matrices.
- `MatMult()` and `MatMultAdd()`: Block-wise matrix-vector products respecting heterogeneous index sets.

---

## 4. FieldSplit Preconditioner

**Location:** `petsc/src/ksp/pc/impls/fieldsplit/fieldsplit.c`

PCFieldSplit implements additive/multiplicative field splitting and integrates directly with MatNest.

### Architecture

The implementation uses a linked list of `PC_FieldSplitLink` structures, one per field:
- Each link has an IS (index set) defining which DOFs belong to this field.
- An independent KSP solver for the block.
- VecScatter contexts for gathering/scattering vectors between global and block-local representations.

### MatNest Integration (~lines 1982–1984)

```c
PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat, MATNEST, &matnest));
if (matnest) PetscCall(MatNestGetSize(pc->pmat, &mis, &nis));
PetscCall(MatNestGetISs(pc->pmat, rowis, colis));
```

When using MatNest, field i corresponds to the i-th IS pair from `MatNestGetISs()`. No need for explicit IS definition — the structure is automatically extracted from the matrix.

### Splitting Types Available

The gravity script uses `symmetric_multiplicative`:
```python
"pc_fieldsplit_type": "symmetric_multiplicative"
```

This means the fields are solved in sequence (0→1→2), then back (2→1→0), with each solve using the latest updates from preceding solves. This is effective for the coupled gravity-momentum system because:
- Displacement update (field 0) informs the internal variable solve (field 1).
- Both inform the gravity solve (field 2).
- The backward sweep refines the coupling.

---

## 5. DMComposite for Multiple DMs

**Location:** `petsc/src/dm/impls/composite/pack.c`

While less commonly used for submeshes, DMComposite allows packing multiple independent DMs:
- `DMCompositeAddDM()`: Adds a separate DM to the composite.
- `DMCompositeGetAccess()`: Extracts component vectors.
- `DMCompositeSetCoupling()`: Specifies coupling between components.

This is an alternative to MatNest for managing multi-physics problems, though Firedrake primarily uses the MatNest + FieldSplit approach.

---

## 6. The MatFree Path

The gravity script uses `"mat_type": "matfree"`, meaning the Jacobian matrix is never explicitly assembled. Instead:

1. The nonlinear solver computes the action of the Jacobian on a vector (Jv) by evaluating the form's derivative.
2. The FieldSplit preconditioner still assembles individual block preconditioners (via `AssembledPC` for each field).
3. This saves memory and allows coupling terms to be evaluated exactly without sparse matrix storage.

For each fieldsplit block:
- Field 0 (displacement): Assembled as GAMG (algebraic multigrid).
- Field 1 (internal variable): Assembled as SOR.
- Field 2 (gravity): Assembled as LU (direct solve).

---

## 7. Cross-Mesh Assembly Pattern

Putting it all together, the assembly-to-solve workflow is:

1. **Create submeshes** using `DMPlexCreateSubmesh()` with appropriate DMLabel.
2. **Build separate Sections** for each mesh's DOF layout.
3. **Assemble blocks independently** on their respective meshes.
4. **Create MatNest** (or use matfree) with row/column index sets from each mesh's DOF distribution.
5. **Apply FieldSplit preconditioner** using MatNest structure to decompose into per-field solves.
6. **Use VecScatter** to move data between global coupled vector and block-local vectors.

The key advantage of MatNest is that blocks A_ij can have completely different dimensions:
- A_00 has DOFs from submesh (displacement rows and columns).
- A_11 has DOFs from submesh (internal variable rows and columns).
- A_22 has DOFs from full mesh (gravity rows and columns).
- A_02, A_20 have heterogeneous indices mapping between submesh and full mesh.

All blocks can have completely different sparsity patterns, and PETSc handles the index translation transparently.

---

## 8. Distributed Mesh Handling

For parallel execution:
- `PetscSF` maintains partition boundary information.
- Submesh SF is created from parent mesh SF (lines 4087–4157 in `plexsubmesh.c`).
- Cross-mesh coupling requires proper ownership transfer.
- Each process owns a portion of both submesh and full mesh DOFs, and the global index sets ensure consistency.
