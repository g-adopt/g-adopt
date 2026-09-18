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

pytest = "local"
