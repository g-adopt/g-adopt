"""Case definition for the weak boundary condition tests of the Stokes momentum term.

The cost of these tests is firedrake kernel compilation, which is serial within
a process. Splitting them into one step per (group, mesh) pair lets
`doit run_case` compile them across the machine, so the wall clock becomes the
slowest single step instead of the sum of all of them.

Every step is single-core: these are algebraic properties of the assembled
operator and small manufactured-solution problems, none of which need MPI.
"""

# The meshes each symmetry group runs on. Together they cover each distinct code
# path in the weak boundary terms once: simplex and tensor-product cells, two
# and three dimensions, extrusion, and manifold normals.
SYMMETRY_MESHES = ["2D-tri", "2D-quad", "3D-tet", "3D-extruded", "2D-cylinder"]
# The reference-form and variational-structure groups build every shipped
# approximation on each mesh, so they cost far more per mesh. They run on the
# smallest simplex mesh in each dimension, which is where a wrong coefficient
# shows up just as clearly.
REFERENCE_MESHES = ["2D-tri", "3D-tet"]
STRUCTURE_MESHES = ["2D-tri", "3D-tet", "2D-cylinder"]

GROUP_MESHES = {
    "nonlinear_viscosity": SYMMETRY_MESHES,
    "weak_u_symmetry": SYMMETRY_MESHES,
    "variational_structure": STRUCTURE_MESHES,
    "explicit_forms": REFERENCE_MESHES,
}

steps = {}

for group, mesh_keys in GROUP_MESHES.items():
    for mesh_key in mesh_keys:
        steps[f"{group}-{mesh_key}"] = {
            "entrypoint": "symmetry.py",
            "args": f"--test {group} --mesh {mesh_key}",
            "outputs": [f"{group}-{mesh_key}.dat"],
        }

# The manufactured-solution cases run their own refinement chain of unit squares
# and take no mesh argument.
for case in ["un", "u"]:
    steps[f"mms-{case}"] = {
        "entrypoint": "mms.py",
        "args": f"--case {case}",
        "outputs": [f"mms-{case}.dat"],
    }

steps["tosi"] = {
    "entrypoint": "tosi.py",
    "outputs": ["tosi.dat"],
}

pytest = "local"
