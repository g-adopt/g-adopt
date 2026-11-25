# Firedrake Submesh and RelabeledMesh: Comprehensive Exploration

## Overview

This document provides a detailed analysis of how Firedrake's `Submesh` and `RelabeledMesh` work, including their implementation, parent-submesh relationships, DOF mapping, and how to couple integrals across different meshes using the `intersect_measures` parameter.

---

## 1. High-Level Architecture

### Key Classes and Functions

- **`Submesh(mesh, ...)`** (lines 4863-4964 in mesh.py): Factory function that creates a submesh from a parent mesh
- **`RelabeledMesh(mesh, ...)`** (lines 4732-4831 in mesh.py): Factory function that creates a mesh with new subdomain labels
- **`MeshGeometry`** (lines 2339+): Represents mesh topology and geometry combined
- **`MeshTopology`** (line 1070+): The topological part of a mesh
- **`AbstractMeshTopology`** (line 500+): Base class for all topologies

### Key Relationship: Parent-Submesh

Every `MeshGeometry` and `MeshTopology` can have an optional `submesh_parent` attribute:
- `submesh_parent` is set when creating a `Submesh` (line 4956, passed to the `Mesh()` factory)
- Both geometry and topology maintain this parent reference (lines 544, 2380, 4941)
- This enables a hierarchy: submesh → parent mesh → grandparent, etc.

---

## 2. RelabeledMesh Implementation

### What It Does

`RelabeledMesh` creates a NEW mesh that shares the SAME topology as the parent but with DIFFERENT subdomain labels. It's used when you want to mark cells/facets with custom IDs based on indicator functions.

### Implementation Details (lines 4732-4831)

```python
def RelabeledMesh(mesh, indicator_functions, subdomain_ids, **kwargs):
    """
    - mesh: input MeshGeometry
    - indicator_functions: list of Functions (DQ/DP degree 0 for cells, P/HDiv Trace degree 0 for facets)
    - subdomain_ids: integer IDs to assign to marked entities
    """
```

**Key Steps:**

1. **Clone the PETSc DMPlex topology** (line 4776):
   ```python
   plex1 = plex.clone()
   ```
   This creates a NEW DMPlex that shares the SAME underlying topology but can have different labels.

2. **Remove old distribution labels** (lines 4779-4781):
   - Clear `pyop2_core`, `pyop2_owned`, `pyop2_ghost` labels
   - Keep `exterior_facets` and `interior_facets` labels (for distributed meshes)

3. **Create new Cell/Face set labels** (lines 4784-4786):
   - Create `CELL_SETS_LABEL` (= "Cell Sets") or `FACE_SETS_LABEL` (= "Face Sets")

4. **Mark entities using indicator functions** (lines 4787-4807):
   ```python
   for f, subid in zip(indicator_functions, subdomain_ids):
       elem = f.topological.function_space().ufl_element()
       if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
           height = 0  # Mark cells
           dmlabel_name = CELL_SETS_LABEL
       elif elem.family() == "HDiv Trace" and elem.degree() == 0:
           height = 1  # Mark facets
           dmlabel_name = FACE_SETS_LABEL
       # Use mark_points_with_function_array to mark DMPlex points
       dmcommon.mark_points_with_function_array(plex, section, height, 
                                                 f.dat.data_ro_with_halos, 
                                                 label, subid)
   ```

5. **Create new MeshTopology and MeshGeometry** (lines 4809-4824):
   - Create `MeshTopology` with the new plex but SAME coordinates
   - Create `MeshGeometry` with the SAME coordinate values but on the new topology
   - **Important**: `submesh_parent` is NOT set (it's a relabeling, not a subset)

6. **Preserve distribution parameters** (lines 4828-4829):
   ```python
   rmesh._distribution_parameters = mesh._distribution_parameters
   ```

### Key Insight

**RelabeledMesh does NOT create a subset of the mesh** — it creates a topologically identical mesh with different entity labels. The new mesh covers the ENTIRE parent domain but with custom integer markers on selected entities.

### Constraints

- Cannot work with `ExtrudedMesh` or `VertexOnlyMesh` (would need to relabel base mesh first)
- All entities must be marked consistently
- Entity marking rules:
  - **Cells**: DQ/DP degree 0 functions
  - **Facets (2D/3D)**: HDiv Trace degree 0, or P degree 1 (1D), or Q degree 2 (hex only)

---

## 3. Submesh Implementation

### What It Does

`Submesh` creates a PROPER SUBSET of a parent mesh, containing only selected entities (cells or facets). The submesh has its OWN topology that is structurally smaller than the parent.

### Implementation Details (lines 4863-4964)

```python
def Submesh(mesh, subdim=None, subdomain_id=None, label_name=None, 
            name=None, ignore_halo=False, reorder=None, comm=None):
    """
    - mesh: parent MeshGeometry
    - subdim: topological dimension of submesh (defaults to parent's dimension)
    - subdomain_id: integer ID to select entities (if None, select ALL)
    - label_name: name of label containing subdomain_id (e.g., "Cell Sets")
    """
```

**Key Steps:**

1. **Validation** (lines 4920-4931):
   - Cannot create submesh of ExtrudedMesh or VertexOnlyMesh
   - Can only create co-dimension 0 or 1 submeshes (same dimension or one less)

2. **Select label and subdomain_id** (lines 4932-4942):
   ```python
   if subdomain_id is None:
       label_name = "depth"
       subdomain_id = subdim  # Select all entities at that depth
   elif label_name is None:
       if subdim == dim:
           label_name = CELL_SETS_LABEL       # "Cell Sets"
       elif subdim == dim - 1:
           label_name = FACE_SETS_LABEL       # "Face Sets"
   ```

3. **Create submesh DMPlex** (line 4943):
   ```python
   subplex = dmcommon.submesh_create(plex, subdim, label_name, subdomain_id, 
                                     ignore_halo, comm=comm)
   ```
   This uses PETSc's DMPlex submesh extraction algorithm.

4. **Create submesh Mesh object** (lines 4954-4961):
   ```python
   submesh = Mesh(
       subplex,
       submesh_parent=mesh,        # Key: set parent reference!
       name=name,
       comm=comm,
       reorder=reorder,
       distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
   )
   ```

### Key Insight

**The `submesh_parent` attribute is the CRITICAL LINK** that enables:
- DOF mapping between parent and submesh function spaces
- Cross-mesh assembly
- Interpolation between parent and submesh

---

## 4. Parent-Submesh Relationships via Topology

### Entity Maps in MeshTopology

Once a submesh is created, its topology maintains several MAPS that relate entities:

#### Child → Parent Maps (lines 1543-1558)

```python
submesh_child_cell_parent_cell_map
    # Maps submesh cells to parent cells

submesh_child_exterior_facet_parent_exterior_facet_map
    # Maps submesh exterior facets to parent exterior facets

submesh_child_exterior_facet_parent_interior_facet_map
    # Maps submesh exterior facets to parent interior facets
    # (new boundary facets become interior in submesh)

submesh_child_interior_facet_parent_interior_facet_map
    # Maps submesh interior facets to parent interior facets
```

#### Parent → Child Maps (lines 1580-1613)

```python
submesh_parent_cell_child_cell_map
submesh_parent_exterior_facet_child_exterior_facet_map
submesh_parent_exterior_facet_child_interior_facet_map
submesh_parent_interior_facet_child_exterior_facet_map
submesh_parent_interior_facet_child_interior_facet_map
submesh_parent_exterior_facet_child_cell_map
submesh_parent_interior_facet_child_cell_map
```

#### How Maps Are Created (lines 1531-1541)

```python
def _submesh_make_entity_entity_map(self, from_set, to_set, 
                                    from_points, to_points, 
                                    child_parent_map):
    """
    Maps entities between two meshes using PETSc subpoint information.
    
    from_set, to_set: PyOP2 Sets (cell_set, facet_set, etc.)
    from_points, to_points: DMPlex point IDs for the entities
    child_parent_map: bool indicating direction
    
    Returns: PyOP2 Map object for assembly/interpolation
    """
    with self.topology_dm.getSubpointIS() as subpoints:
        if child_parent_map:
            _, from_indices, to_indices = np.intersect1d(
                subpoints[from_points], to_points, 
                return_indices=True
            )
        else:
            _, from_indices, to_indices = np.intersect1d(
                from_points, subpoints[to_points], 
                return_indices=True
            )
    values = np.full(from_set.total_size, -1, dtype=IntType)
    values[from_indices] = to_indices
    return op2.Map(from_set, to_set, 1, values.reshape((-1, 1)), ...)
```

**Key Point**: The maps use PETSc's `getSubpointIS()` to map between local point numberings. A value of -1 means "not in the corresponding set" (e.g., a new boundary facet in the submesh that wasn't in the parent).

### Submesh Ancestors (lines 954-960)

```python
@cached_property
def submesh_ancestors(self):
    """Tuple of submesh ancestors."""
    if self.submesh_parent:
        return (self, ) + self.submesh_parent.submesh_ancestors
    else:
        return (self, )
```

This creates a chain: submesh → parent → grandparent, etc.

---

## 5. DOF Mapping and Function Spaces

### How FunctionSpace Works on Submesh

When you create a `FunctionSpace` on a submesh:

```python
V_parent = FunctionSpace(parent_mesh, "CG", 1)
V_submesh = FunctionSpace(submesh, "CG", 1)
```

The submesh FunctionSpace has FEWER DOFs than the parent (because the submesh has fewer cells/entities).

### Interpolation Between Meshes (test_submesh_interpolate.py)

The submesh parent relationship enables interpolation:

```python
f_parent = Function(V_parent).interpolate(expr)
f_submesh = Function(V_submesh).interpolate(f_parent)
```

This works because:
1. The submesh topology knows its parent topology
2. The entity maps allow DOFs to be matched between parent and submesh
3. Interpolation uses these maps to transfer values

### Example from test_submesh_interpolate.py (lines 38-50):

```python
def _test_submesh_interpolate_cell_cell(mesh, subdomain_cond, fe_fesub):
    label_value = 999
    subm = make_submesh(mesh, subdomain_cond, label_value)
    
    V = FunctionSpace(mesh, family, degree)
    Vsub = FunctionSpace(subm, family_sub, degree_sub)
    
    f = Function(V).interpolate(_get_expr(V))
    fsub = Function(Vsub).interpolate(f)  # Interpolate from parent to submesh
    
    # f and fsub should agree on the submesh domain
    assert np.allclose(fsub.dat.data_ro_with_halos, 
                      gsub.dat.data_ro_with_halos)
```

---

## 6. Assembly and Coupling via `intersect_measures`

### The Problem

When assembling forms on multiple meshes (parent and submesh), you need to tell Firedrake:
- Which mesh is the "primary" domain
- What integral type to use on OTHER meshes

Example: A form on parent and submesh:
```python
V_parent = FunctionSpace(parent_mesh, "CG", 1)
V_submesh = FunctionSpace(submesh, "CG", 1)
V = V_parent * V_submesh

u_parent, u_submesh = TrialFunctions(V)
v_parent, v_submesh = TestFunctions(V)

# How to define bilinear form that couples them?
a = ...
```

### The Solution: intersect_measures

Define measures with `intersect_measures` parameter (from ufl/measure.py):

```python
# Define measure on parent with reference to submesh
dx_parent = Measure("dx", domain=parent_mesh, 
                    intersect_measures=(Measure("dx", submesh),))

# Define measure on submesh with reference to parent
dx_submesh = Measure("dx", domain=submesh, 
                     intersect_measures=(Measure("dx", parent_mesh),))

# Now you can assemble forms that reference both
a = inner(grad(u_parent), grad(v_parent)) * dx_parent + \
    inner(u_submesh, v_submesh) * dx_submesh
```

### How It Works (from ufl/measure.py, lines 114-198)

```python
class Measure:
    def __init__(self, integral_type, domain=None, subdomain_id="everywhere",
                 metadata=None, subdomain_data=None, 
                 intersect_measures=None):  # <-- NEW PARAMETER
        self._integral_type = integral_type
        self._domain = domain
        self._subdomain_id = subdomain_id
        self._metadata = metadata
        
        if intersect_measures is None:
            self._intersect_measures = ()
        else:
            # Validate: all intersect measures must have "everywhere" subdomain_id
            if not all(m.subdomain_id() == "everywhere" for m in intersect_measures):
                raise NotImplementedError(
                    f"Currently, all intersect measures must have "
                    f"'everywhere' subdomain_id: got {intersect_measures}"
                )
            # Validate: nested intersect measures not allowed
            if not all(m.intersect_measures() == () for m in intersect_measures):
                raise ValueError(
                    f"All intersect measures must have empty intersect_measures"
                )
            # Validate: no metadata on intersect measures
            if not all(m.metadata() == {} for m in intersect_measures):
                raise ValueError(
                    f"All intersect measures must have empty metadata"
                )
            self._intersect_measures = tuple(
                sorted(intersect_measures, 
                       key=lambda m: m.ufl_domain()._ufl_sort_key_())
            )
```

**Constraints:**
- All intersect measures must have `subdomain_id="everywhere"`
- Intersect measures cannot themselves have intersect measures (no nesting)
- Intersect measures cannot have metadata
- Subdomain IDs can only be specified on the PRIMAL (primary) domain, not intersect measures

### Practical Example from test_submesh_solve.py (lines 98-100)

```python
# Problem: solve on mixed parent-submesh space
V0 = FunctionSpace(mesh, "CG", 2)      # Parent space
V1 = FunctionSpace(subm, "CG", 3)      # Submesh space
V = V0 * V1

u = TrialFunction(V)
v = TestFunction(V)
u0, u1 = split(u)
v0, v1 = split(v)

# Define measures that know about each other
dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))

# Couple them: u0 only on parent, u1 only on submesh, but they interact
a = inner(grad(u0), grad(v0)) * dx0 + inner(u0 - u1, v1) * dx1
L = inner(Constant(0.), v1) * dx1
```

### How TSFC Handles intersect_measures

From tsfc/driver.py (line 205):
- When preprocessing integrals, TSFC checks that all argument/coefficient domains are valid
- If not, it suggests using `intersect_measures` to declare the domain relationships
- The compiler then uses `submesh_*_*_map` objects to couple the integrals

---

## 7. Cell Closure and Entity Numbering

### Cell Closure (referenced in lines 1260, 1545)

```python
self.cell_closure  # Shape: (num_cells, num_points_in_closure)
# Points are in ascending order, with the cell itself last
self.submesh_parent.cell_closure
```

The cell closure is used to map between:
- **Submesh cells** → their corresponding **parent cells** (via `cell_closure[:, -1]`)

Example from line 1545:
```python
return self._submesh_make_entity_entity_map(
    self.cell_set, 
    self.submesh_parent.cell_set,
    self.cell_closure[:, -1],                    # Submesh cell points
    self.submesh_parent.cell_closure[:, -1],     # Parent cell points
    True  # child_parent_map=True
)
```

---

## 8. Distribution Parameters and Parallel Support

### Distribution for Submeshes (line 95)

```python
DISTRIBUTION_PARAMETERS_NOOP = {
    "partition": False,
    "overlap_type": (DistributedMeshOverlapType.NONE, 0),
}
"""Distribution parameters for derived meshes (RelabeledMesh/Submesh)."""
```

When creating a submesh (line 4960):
```python
submesh = Mesh(
    subplex,
    distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,  # Don't repartition
)
```

This ensures:
- The submesh uses the SAME partition as the parent
- No additional communication between processes
- Preserves parent-submesh DOF relationships

### Preservation of Original Distribution (line 4962-4963)

```python
submesh._distribution_parameters = mesh._distribution_parameters
```

The original distribution parameters are saved so the submesh "remembers" how its parent was distributed.

---

## 9. Required Overlap for Facet Submeshes (docstring, lines 4902-4912)

```
To make a submesh of co-dimension 1, the parent mesh must have
been overlapped with DistributedMeshOverlapType of
{None, VERTEX, RIDGE}; see distribution_parameters kwarg of Mesh().

To use interior facet integration on a submesh of co-dimension 1,
the parent mesh must have been overlapped with
DistributedMeshOverlapType of {VERTEX, RIDGE}, and the
facets of the parent mesh must have been labeled such that the
ridges (entities of co-dim 2) to be contained in the submesh are
shared by at most two facets.
```

This is a PETSc-level requirement for proper facet labeling in parallel.

---

## 10. Mark Entities Method (lines 924-1527)

### Purpose

Pre-mark entities on a mesh so they can later be extracted as a submesh:

```python
mesh.mark_entities(indicator_function, label_value, label_name=None)
```

### Implementation (MeshTopology version, lines 1484-1527)

```python
def mark_entities(self, tf, label_value, label_name=None):
    # tf: CoordinatelessFunction, must be scalar
    # label_value: integer ID
    # label_name: "Cell Sets" or "Face Sets" (auto-selected if None)
    
    elem = tf.function_space().ufl_element()
    
    if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
        # Mark cells
        height = 0
        label_name = label_name or CELL_SETS_LABEL
    elif (elem.family() == "HDiv Trace" and elem.degree() == 0) or \
         (elem.family() == "Lagrange" and elem.degree() == 1 and self.cell_dimension() == 1) or \
         (elem.family() == "Q" and elem.degree() == 2):
        # Mark facets
        height = 1
        label_name = label_name or FACE_SETS_LABEL
    else:
        raise ValueError("Must use DQ/DP for cells, P/HDiv Trace for facets")
    
    plex = self.topology_dm
    if not plex.hasLabel(label_name):
        plex.createLabel(label_name)
    plex.clearLabelStratum(label_name, label_value)
    label = plex.getLabel(label_name)
    section = tf.function_space().dm.getSection()
    array = tf.dat.data_ro_with_halos.real.astype(IntType)
    
    # Call PETSc routine to mark points
    dmcommon.mark_points_with_function_array(plex, section, height, 
                                             array, label, label_value)
```

### Typical Usage Pattern (from tests)

```python
# Step 1: Create a function to mark entities
mesh = RectangleMesh(10, 10, 1.0, 1.0)
x, y = SpatialCoordinate(mesh)
DQ0 = FunctionSpace(mesh, "DQ", 0)
indicator = Function(DQ0).interpolate(conditional(x > 0.5, 1, 0))

# Step 2: Mark entities
mesh.mark_entities(indicator, 999)

# Step 3: Extract submesh
submesh = Submesh(mesh, dim, 999)
```

---

## 11. Hierarchy and Submesh Ancestors

### Chain of Submeshes

You can create submeshes of submeshes:

```python
mesh = create_base_mesh()
submesh1 = Submesh(mesh, ..., subdomain_id=A)
submesh2 = Submesh(submesh1, ..., subdomain_id=B)
submesh3 = Submesh(submesh2, ..., subdomain_id=C)
```

Each maintains its parent reference:
- `submesh3.topology.submesh_parent = submesh2.topology`
- `submesh2.topology.submesh_parent = submesh1.topology`
- `submesh1.topology.submesh_parent = mesh.topology`

### Accessing Ancestors

```python
ancestors = submesh3.topology.submesh_ancestors
# Returns: (submesh3.topology, submesh2.topology, submesh1.topology, mesh.topology)
```

This is useful for finding the common ancestor between two submeshes.

---

## 12. Key Test Cases

### Basic Submesh (test_submesh_basics.py)

Shows that `submesh.topology.submesh_parent is parent.topology`:

```python
mesh = UnitIntervalMesh(2)
M = FunctionSpace(mesh, "DG", 0)
m = Function(M); m.dat.data[0] = 1
cell_marker = 100
parent = RelabeledMesh(mesh, [m], [cell_marker])
submesh = Submesh(parent, parent.topological_dimension, cell_marker)
assert submesh.topology.submesh_parent is parent.topology
```

### Base Integral on Submesh (test_submesh_base.py)

Shows that integrals on submesh match integrals on parent restricted to submesh:

```python
mesh = UnitSquareMesh(nelem, nelem, quadrilateral=True)
V = FunctionSpace(mesh, "Q", 4)
f = Function(V).interpolate(expr)
cond = conditional(x > .5, 1, conditional(y > .5, 1, 0))
target = assemble(f * cond * dx)  # Integral over submesh in parent

DQ0 = FunctionSpace(mesh, "DQ", 0)
indicator_function = Function(DQ0).interpolate(cond)
mesh.mark_entities(indicator_function, 999)
msub = Submesh(mesh, dim, 999)
Vsub = FunctionSpace(msub, "Q", 4)
fsub = Function(Vsub).interpolate(expr)
result = assemble(fsub * dx)  # Integral over submesh

assert abs(result - target) < 1e-12  # They match!
```

### Coupled Assembly (test_submesh_assemble.py)

Shows how to assemble on mixed parent-submesh spaces with `intersect_measures`:

```python
mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
x, y = SpatialCoordinate(mesh)
DQ0 = FunctionSpace(mesh, "DQ", 0)
indicator = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
mesh.mark_entities(indicator, 999)
subm = Submesh(mesh, dim, 999)

V0 = FunctionSpace(mesh, "CG", 1)
V1 = FunctionSpace(subm, "CG", 1)
V = V0 * V1

u = TrialFunction(V)
v = TestFunction(V)
u0, u1 = split(u)
v0, v1 = split(v)

# Define coupled measures
dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))

# Create coupled bilinear form
a = inner(u1, v0) * dx0(999) + inner(u0, v1) * dx1
A = assemble(a, mat_type="nest")

# Verify sparsity structure
assert A.M[0][1].nnz == expected_sparsity
```

### Coupled Solver (test_submesh_solve.py)

Shows solving a coupled parent-submesh PDE:

```python
mesh = RectangleMesh(nelem*2, nelem*2, 2., 1., quadrilateral=True)
indicator = Function(DQ0).interpolate(conditional(x < 1., 1, 0))
mesh.mark_entities(indicator, 999)
mesh = Submesh(mesh, dim, 999)  # Get submesh

V0 = FunctionSpace(mesh, "CG", 2)
V1 = FunctionSpace(subm, "CG", 3)
V = V0 * V1

u = TrialFunction(V)
v = TestFunction(V)
u0, u1 = split(u)
v0, v1 = split(v)

dx0 = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", subm),))
dx1 = Measure("dx", domain=subm, intersect_measures=(Measure("dx", mesh),))

# Coupled PDE
a = inner(grad(u0), grad(v0)) * dx0 + inner(u0 - u1, v1) * dx1
L = inner(Constant(0.), v1) * dx1
bc = DirichletBC(V.sub(0), g, bid)

solution = Function(V)
solve(a == L, solution, bcs=[bc])
```

---

## 13. Summary: When to Use What

### Use `RelabeledMesh` when:
- You want to CREATE CUSTOM SUBDOMAIN IDs on the ENTIRE mesh
- You don't need to subset; you just need to mark regions differently
- Example: Mark boundary regions with your own IDs, then use them in measures

```python
mesh = Mesh("mesh.msh")
indicator = Function(FunctionSpace(mesh, "DQ", 0)).interpolate(...)
mesh_with_labels = RelabeledMesh(mesh, [indicator], [my_id])
```

### Use `Submesh` when:
- You want to extract a PROPER SUBSET of the mesh
- You need separate function spaces on parent and submesh
- You want to solve equations on different regions
- Example: Domain decomposition, interface problems

```python
mesh = Mesh("mesh.msh")
indicator = Function(FunctionSpace(mesh, "DQ", 0)).interpolate(...)
mesh.mark_entities(indicator, region_id)
submesh = Submesh(mesh, mesh.topological_dimension, region_id)
V_submesh = FunctionSpace(submesh, "CG", 1)
```

### Use `intersect_measures` when:
- Assembling forms that couple parent and submesh spaces
- Need to specify how to integrate on different meshes
- Example: Mixed formulations, flux continuity constraints

```python
dx_parent = Measure("dx", domain=mesh, intersect_measures=(Measure("dx", submesh),))
dx_submesh = Measure("dx", domain=submesh, intersect_measures=(Measure("dx", mesh),))
a = form_on_parent * dx_parent + form_on_submesh * dx_submesh
```

---

## 14. Architecture Summary Diagram

```
MeshGeometry (parent_mesh)
├── MeshTopology
│   ├── topology_dm (PETSc DMPlex)
│   ├── cell_set, facet_set, ...
│   ├── cell_closure
│   └── submesh_parent = None
├── coordinates (Function)
└── submesh_parent = None

↓ Submesh(parent_mesh, subdim, label_id)
↓

MeshGeometry (submesh)
├── MeshTopology
│   ├── topology_dm (subplex from parent's plex)
│   ├── cell_set, facet_set, ... (SMALLER sets)
│   ├── cell_closure (SUBSET of parent's closure)
│   ├── submesh_parent = parent_mesh.topology
│   ├── submesh_child_cell_parent_cell_map (PyOP2 Map)
│   ├── submesh_child_exterior_facet_parent_*_facet_map (PyOP2 Maps)
│   └── submesh_parent_*_child_*_map (reverse maps)
├── coordinates (Function with FEWER DOFs)
└── submesh_parent = parent_mesh

FunctionSpace(submesh, "CG", 1)
├── mesh = submesh
├── dofs = SUBSET of FunctionSpace(parent_mesh, "CG", 1).dofs
└── Assembly uses parent-child maps for coupling
```

---

## 15. File Locations in Firedrake

Key files to consult:

- **`firedrake/mesh.py`** (5174 lines):
  - `Submesh()` function (lines 4863-4964)
  - `RelabeledMesh()` function (lines 4732-4831)
  - `MeshGeometry` class (lines 2339+)
  - `MeshTopology` class (lines 1070+)
  - Entity maps (lines 1543-1613)
  - `mark_entities()` method (lines 1484-1527)
  - Parent-submesh hierarchy (lines 954-960, 1531-1541)

- **`fenics-ufl/ufl/measure.py`**:
  - `Measure` class with `intersect_measures` parameter
  - Validation of intersect measures (lines 176-198)

- **Test files** (`tests/firedrake/submesh/`):
  - `test_submesh_basics.py`: Basic parent-submesh relationships
  - `test_submesh_base.py`: Integrals on submesh
  - `test_submesh_assemble.py`: Coupled assembly with `intersect_measures`
  - `test_submesh_solve.py`: Solving on mixed parent-submesh spaces
  - `test_submesh_interpolate.py`: Interpolation between parent and submesh

- **`tsfc/driver.py`** (around line 205):
  - Error message suggesting `intersect_measures`

---

## 16. Key Takeaways

1. **Submesh creates a STRUCTURAL SUBSET**: New topology with fewer entities, parent reference preserved

2. **RelabeledMesh creates RELABELED COPY**: Same topology, different labels, no parent reference (or rather, it IS the parent)

3. **Parent-child maps are CRITICAL**: Enable DOF mapping, interpolation, and coupled assembly

4. **intersect_measures is DECLARATIVE**: You tell the compiler which meshes appear in the form and how to integrate on each

5. **No automatic detection**: The compiler doesn't figure out domain relationships by itself; you must use `intersect_measures`

6. **Parallel safety**: Distribution parameters prevent repartitioning; submesh uses parent's partition

7. **Entity mapping via PETSc**: Uses DMPlex subpoint information to map between local point numberings

8. **Chain-able**: Can create submeshes of submeshes; hierarchy maintained via `submesh_ancestors`

