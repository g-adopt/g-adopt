"""Record the stored gate and forward references from the earlier GIA solver.

This script needs the earlier solver (`gadopt.gia_gravity`), which exists at
commit 2f4b1397 and not later. It writes `gia_gate_reference.json` and
`gia_forward_reference.json` of `tests/unit/data`. The README of this
directory gives the procedure. In short: run it serially from the root of a
checkout of 2f4b1397, with `tests/unit/cases_coord.py` a copy of the present
`tests/gia/gia_cases.py` (only its stored mesh and its coordinate helpers are
used), and with the output directory as the first argument. The module
`gia_solver_cases` that this script imports is the one of 2f4b1397, in
`tests/unit` of that checkout.

The gate references are the residual and two Jacobian actions of the
earlier solver at a coordinate-defined state; the forward references are
three steps of the earlier solver. Every value is reduced to the three
numbers of `probe_pairing` (norm, dot product with a coordinate-defined
probe, probe norm), so the values do not depend on the numbering of the
degrees of freedom. The new solver at 2f4b1397 is checked against the
earlier one in the same run.

Usage (from the root of the 2f4b1397 checkout):
    python record_all.py OUT_DIR [gate] [forward]
"""
import json
import sys
sys.path.insert(0, "tests/unit")
import gadopt  # noqa: E402,F401  (gadopt before firedrake)
import firedrake as fd  # noqa: E402
import gia_solver_cases as C  # noqa: E402  (2f4b1397 helpers: build_old, set_state)
import cases_coord as H  # noqa: E402  (present helpers: stored mesh, coordinates)
import test_gia_solver_gate as G2  # noqa: E402
import test_gia_solver_gate_3d as G3  # noqa: E402
import test_gia_solver as T  # noqa: E402

out_dir = sys.argv[1]
which = sys.argv[2:] or ["gate", "forward"]
meshes = H.annulus_meshes()


def by_name(new, old, layout, direction):
    od = fd.Function(old.solution_space)
    for i, name in enumerate(C.old_field_names(layout)):
        C.fill(od.subfunctions[i], direction.subfunctions[new.field_index[name]])
    return od


if "gate" in which:
    out = {"2d": {}, "3d": {}}
    worst = 0.0
    for case in G2.CASES:
        sea = C.SeaFields(meshes[1], ice=True)
        new, _, _ = C.build_new(meshes, case, sea=sea)
        old, _, layout = C.build_old(meshes, case, sea=sea)
        state = H.coordinate_state(new)
        C.set_state(new, old, layout, state,
                    previous_dt=0.07 if case.scheme == "bdf2" else None)
        directions = [H.coordinate_mixed_function(new, f"direction{k}", 1.0) for k in range(2)]
        r_old, a_old = C.residual_and_actions(old, [by_name(new, old, layout, d) for d in directions])
        index = {n: i for i, n in enumerate(C.old_field_names(layout))}
        out["2d"][case.label] = [H.fingerprint(v, index) for v in [r_old, *a_old]]
        r_new, a_new = C.residual_and_actions(new, directions)
        for v, ref in zip([r_new, *a_new], out["2d"][case.label]):
            d = H.fingerprint_differences(H.fingerprint(v, new.field_index), ref)
            worst = max(worst, max(d.values()))
            H.assert_fingerprints_agree(H.fingerprint(v, new.field_index), ref)
        print("2d", case.label, flush=True)
    base = fd.CubedSphereMesh(radius=1.0, refinement_level=1, degree=2)
    mesh = fd.ExtrudedMesh(base, layers=2, layer_height=0.5, extrusion_type="radial")
    mesh.cartesian = False
    for rep, model, family in G3.CASES:
        new, old, layout = G3.build(mesh, rep, model, family)
        names = C.old_field_names(layout)
        state = H.coordinate_state(new)
        C.set_state(new, old, layout, state, previous_dt=None)
        directions = [H.coordinate_mixed_function(new, f"direction{k}", 1.0) for k in range(2)]
        r_old, a_old = C.residual_and_actions(old, [by_name(new, old, layout, d) for d in directions])
        label = "-".join((rep, model, family))
        out["3d"][label] = [H.fingerprint(v, {n: i for i, n in enumerate(names)})
                            for v in [r_old, *a_old]]
        r_new, a_new = C.residual_and_actions(new, directions)
        for v, ref in zip([r_new, *a_new], out["3d"][label]):
            d = H.fingerprint_differences(H.fingerprint(v, new.field_index), ref)
            worst = max(worst, max(d.values()))
            H.assert_fingerprints_agree(H.fingerprint(v, new.field_index), ref)
        print("3d", label, flush=True)
    json.dump(out, open(f"{out_dir}/gia_gate_reference.json", "w"), indent=1)
    print(f"gate written; worst new-vs-old block difference {worst:.2e}")

if "forward" in which:
    out = {}
    for case in T.FORWARD:
        sub = meshes[1]
        sea = C.SeaFields(sub)
        old, z, layout = C.build_old(meshes, case, dt=C.dt_function(sub), sea=sea,
                                     declare_nullspace=True)
        names = C.old_field_names(layout)
        steps = []
        for k, h in enumerate(T.STEPS):
            old.dt.assign(h)
            if case.load:
                T.surface_ice(sea, k)
            old.solve()
            zs = z.subfunctions
            hist = old.time_integrator.current[0] if old.time_integrator else zs[1]
            steps.append({
                "displacement": H.probe_pairing(zs[0], "displacement"),
                "potential": H.probe_pairing(zs[layout.potential], "potential"),
                "history": H.probe_pairing(hist, "history"),
                "reals": {n: float(zs[i].dat.data_ro_with_halos[0]) for i, n in enumerate(names)
                          if n not in ("displacement", "history", "potential")}})
        out[case.label] = steps
        print("forward", case.label, flush=True)
    json.dump(out, open(f"{out_dir}/gia_forward_reference.json", "w"), indent=1)
    print("forward written")
