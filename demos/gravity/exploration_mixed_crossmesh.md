# Mixed Cross-Mesh Function Spaces in Firedrake

## Overview

The gravity coupling script creates a `MixedFunctionSpace` where components live on **different meshes**:

```python
V      = VectorFunctionSpace(subm, "CG", 2)   # on submesh (mantle only)
S      = TensorFunctionSpace(subm, "DG", 1)   # on submesh
V_grav = FunctionSpace(mesh, "CG", 2)          # on FULL mesh (mantle + exterior)
Z      = MixedFunctionSpace([V, S, V_grav])    # cross-mesh mixed space!
```

This is a remarkable capability. Here is how Firedrake makes it work.

---

## 1. MixedFunctionSpace Creation and Mesh Handling

**Location:** `firedrake/functionspace.py` (lines 266–321)

When `MixedFunctionSpace([V, S, V_grav])` is called:

1. Each component space can have a **different mesh**.
2. Line 298 collects all meshes: `meshes = list(itertools.chain(*[space.mesh() for space in spaces]))` — producing `[subm, subm, mesh]`.
3. This creates a `MeshSequenceGeometry` (line 319) wrapping the ordered tuple of mesh objects.
4. The topological representation uses `MeshSequenceTopology`, which wraps the individual mesh topologies.

The resulting `Z` has an internal `_mesh` that is a `MeshSequenceTopology` containing 3 meshes (with `subm` appearing twice for components 0 and 1).

---

## 2. Data Structure Architecture

**Location:** `firedrake/mesh.py` (lines 4999–5175)

### MeshSequenceGeometry (~line 4999)

- Wraps multiple `MeshGeometry` objects as a sequence.
- Provides `.unique()` method to extract a single mesh if all components are identical.
- Has `.topology` property returning `MeshSequenceTopology`.
- Maintains hierarchy information for multigrid.

### MeshSequenceTopology (~line 5099)

- Wraps multiple mesh topologies.
- Implements `__len__`, `__iter__`, `__getitem__` to act like a tuple.
- `ufl_mesh()` returns `ufl.MeshSequence` with one UFL mesh per component.
- Component spaces reference their respective mesh indices through this sequence.

---

## 3. Function Space Composition

**Location:** `firedrake/functionspaceimpl.py` (lines 1030–1260)

### MixedFunctionSpace Implementation

- Uses `IndexedFunctionSpace` wrapper (line 1051) for each component, which maintains a `.index` (component number) and `.parent` reference.
- The `entity_node_map()` method (lines 1180–1201) creates a `MixedMap` that maps entity DOFs based on which mesh the component lives on:
  ```python
  def entity_node_map(self, source_mesh, source_integral_type, ...):
      return op2.MixedMap(s.entity_node_map(source_mesh, ...)
                          for s in self._spaces)
  ```

### DOF Layout

- DOF dataset: `op2.MixedDataSet` (lines 1172–1178) concatenates DOF sets from all components.
- Total DOFs = sum of all component DOFs, arranged sequentially:
  - DOFs 0 to N₁: Displacement (V) on submesh
  - DOFs N₁ to N₁+N₂: Internal variable (S) on submesh
  - DOFs N₁+N₂ to N₁+N₂+N₃: Gravitational potential (V_grav) on full mesh

---

## 4. Assembly with Cross-Mesh Forms

**Location:** `firedrake/assemble.py`

### Key Architectural Features

- `extract_domains(form)` extracts all mesh objects referenced in the variational form.
- The `_GlobalKernelBuilder` class (line 1644+) generates assembly kernels.
- Lines 1727–1728 map mesh index to actual mesh object:
  ```python
  all_meshes = extract_domains(self._form)
  return all_meshes[self._kinfo.domain_number]
  ```

### How Cross-Mesh Assembly Works

For the coupled form `F` in the gravity script:
1. Assembly encounters terms on both `subm` and `mesh`.
2. The `intersect_measures` directives specify which integral applies to which mesh.
3. TSFC (the form compiler) generates kernels that know about multiple meshes.
4. The global kernel uses mixed maps to access DOFs on the correct mesh during element-wise assembly.

The form `F` (lines 500–504) combines:
- Momentum terms using `dx_m` (submesh with full-mesh intersection)
- Internal variable terms using `dx_m`
- Gravity Laplacian using `dx_all` (full mesh with submesh intersection)
- Gravity source using `dx_m`

All are assembled into a single monolithic residual vector and Jacobian matrix.

---

## 5. Solver Integration with PETSc FieldSplit

**Location:** `firedrake/variational_solver.py`, `firedrake/solving_utils.py`

### How FieldSplit Works with Cross-Mesh Mixed Spaces

- `MixedFunctionSpace` has a `.dm` property (line 1240 in `functionspaceimpl.py`) returning a PETSc DM.
- `dmhooks.attach_hooks()` (line 1247) enables field decomposition.
- The DM structure informs PETSc's `-pc_type fieldsplit` decomposition.

In the gravity script (lines 226–267):
```python
"pc_type": "fieldsplit",
"pc_fieldsplit_type": "symmetric_multiplicative",
"fieldsplit_0_...": {...},  # displacement V on submesh
"fieldsplit_1_...": {...},  # internal variable S on submesh
"fieldsplit_2_...": {...},  # gravity potential V_grav on full mesh
```

Each fieldsplit block solves its sub-problem independently, with its own KSP and PC. The symmetric multiplicative type means each block solve uses updated information from the previous blocks, improving convergence for the coupled system.

---

## 6. DOF Management Across Meshes

**Location:** `firedrake/dmhooks.py` (lines 50–103)

### How DOFs Are Tracked with MeshSequences

```python
def set_function_space(dm, V):
    mesh = V.mesh()
    # Line 102: stores tuple of mesh references (weakrefs)
    info = (tuple(weakref.ref(m) for m in mesh), element, ...)
    dm.setAttr("__fs_info__", info)
```

Lines 58–61 reconstruct the mesh from the DM:
```python
if len(meshref_tuple) == 1:
    mesh = meshref_tuple[0]()
else:
    mesh = MeshSequenceGeometry([meshref() for meshref in meshref_tuple])
```

This enables PETSc to understand the multi-mesh structure when decomposing problems.

---

## 7. The Key Insight

Firedrake doesn't literally "solve on different meshes simultaneously" in separate linear algebra systems. Instead, it creates a **unified global DOF vector** where:

- DOFs 0 to N₁: displacement (V) — mapped to submesh entities
- DOFs N₁ to N₁+N₂: internal variable (S) — mapped to submesh entities
- DOFs N₁+N₂ to N₁+N₂+N₃: gravitational potential (V_grav) — mapped to full mesh entities

The matrix assembly kernels know which DOFs correspond to which mesh through the `entity_node_map` mechanism. Coupling terms (like the gravity source ∫ρ₁·v dx_m, where ρ₁ depends on displacement u from the submesh) are assembled by evaluating functions on the appropriate mesh and writing into the correct DOF positions in the global system.

The PETSc fieldsplit preconditioner then decomposes this global system back into per-field blocks for efficient iterative solving, with each block's KSP operating on the DOFs of its respective mesh.

---

## 8. Design Constraints and Limitations

- `MeshSequenceTopology` requires exactly one `MeshTopology` per component — nested mixed spaces with the same mesh still produce separate sequence entries.
- All component meshes must have **compatible communicators** (line 1115 in `functionspaceimpl.py`).
- Assembly ordering: meshes appear in sequence order; DOFs are numbered sequentially by component.
- Field decomposition order in the solver matches component order in `MixedFunctionSpace([V, S, V_grav])`.

---

## 9. Test Coverage

Found in `firedrake/tests/firedrake/submesh/test_submesh_assemble.py`:

Tests demonstrate MixedFunctionSpace assembly across submeshes:
```python
V0 = FunctionSpace(mesh, "CG", 1)
V1 = FunctionSpace(subm, "CG", 1)
V = V0 * V1  # Mixed space: one component on each mesh
# Assembly uses intersect_measures for proper coupling
```

This confirms that the cross-mesh MixedFunctionSpace is a supported and tested Firedrake feature.
