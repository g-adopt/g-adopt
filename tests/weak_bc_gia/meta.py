"""Case definition for the weak boundary condition tests of the viscoelastic solvers.

The cost of these tests is firedrake kernel compilation, which is serial within
a process. Splitting them into one step per (group, mesh) pair lets
`doit run_case` compile them across the machine, so the wall clock becomes the
slowest single step instead of the sum of all of them.

Every step is single-core: these are algebraic properties of the assembled
operator and small surface-load problems, none of which need MPI.
"""

# Both meshes use affine simplex cells, so the pointwise and the coupled
# formulation discretise the same continuous problem on them. Two and three
# dimensions is the only distinction the weak boundary terms make here; the
# cell-type and manifold paths are covered by tests/weak_bc_stokes.
MESHES = ["2D-tri", "3D-tet"]

GROUPS = [
    "block_symmetry",
    "pointwise_structure",
    "coupled_structure",
    "weak_u_symmetry",
]

# Ratios of the time step to the Maxwell time the refinement case runs at. The
# asymmetry a wrong symmetrising coefficient produces grows with this ratio, so
# one small and one large value bracket the behaviour.
REFINEMENT_DT_OVER_TAU = [0.25, 25.0]

steps = {}

for group in GROUPS:
    for mesh_key in MESHES:
        steps[f"{group}-{mesh_key}"] = {
            "entrypoint": "gia_symmetry.py",
            "args": f"--test {group} --mesh {mesh_key}",
            "outputs": [f"{group}-{mesh_key}.dat"],
        }

# The refinement case runs its own chain of unit squares and takes no mesh
# argument.
for dt_over_tau in REFINEMENT_DT_OVER_TAU:
    steps[f"refinement-{dt_over_tau}"] = {
        "entrypoint": "gia_refinement.py",
        "args": f"--dt-over-tau {dt_over_tau}",
        "outputs": [f"refinement-{dt_over_tau}.dat"],
    }

steps["iterative_preset"] = {
    "entrypoint": "gia_iterative.py",
    "outputs": ["iterative_preset.dat"],
}

# The coupled internal-variable solver under its static-condensation preset.
# Each entry is one process, so the three compile in parallel under
# `doit run_case`. "preset" runs the solves that compare the preset with a
# direct solve and the history layouts with each other, "operator" measures
# the symmetry and the reuse of the condensed operator, and "pc" runs the
# near-incompressible preconditioners and the taped solve.
CONDENSATION_ENTRIES = ["preset", "operator", "pc"]

# The (number of Maxwell elements, boundary conditions) pairs of the preset
# rows. Three elements is the case that Firedrake's `SCPC` refuses in a layout
# with one field per element. "weak" puts `un` on three sides, "strong" puts a
# component-wise Dirichlet condition there.
CONDENSATION_PRESET_CASES = [
    (n_elements, bc_kind)
    for bc_kind in ("weak", "strong")
    for n_elements in (1, 2, 3)
]

# The dimensions of the divergence-free mode rows and the coarse solvers of
# the patch multigrid rows.
CONDENSATION_SOLENOIDAL_DIMS = [2, 3]
CONDENSATION_MG_COARSE = ["gamg", "lu"]

for entry in CONDENSATION_ENTRIES:
    steps[f"condensation-{entry}"] = {
        "entrypoint": "gia_condensation.py",
        "args": f"--entry {entry}",
        "outputs": [f"condensation-{entry}.dat"],
    }

pytest = "local"
