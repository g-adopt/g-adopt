"""Serial tests of the Extended Burgers part of `prepare_exp09.py`.

The tests run on one rank on a small box mesh near the surface radius, so
they need no GIAMIP mesh and no data file. They check:

- `load_eb_fit` reads the fit of decision S31 (`free_time_fits/3`) and
  refuses a fit with other EB parameters, a broken constraint or a zero
  modulus;
- `burgers_elements` gives mu_inf mu_bar_j and eta_0 mu_bar_j tau_bar_j
  below 90 km and the split elastic element in the lid, and the solver's
  reader `exp09_common.element_fields` returns the same columns;
- the element Maxwell times below the lid are tau_bar_j eta_0 / mu_inf at
  every node, and the instant modulus ratio is sum mu_bar below the lid and
  1 in the lid (`burgers_by_layer`);
- the lid check refuses a viscosity whose elastic cells are not the cells
  above 90 km.

Run (from the worktree root, PYTHONPATH as in CLAUDE.md):
    python -m pytest demos/glacial_isostatic_adjustment/giamip/exp09/test_prepare_exp09.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import prepare_exp09 as prepare  # noqa: E402  (imports gadopt first)
import exp09_common as common  # noqa: E402
from gadopt import (BoxMesh, Function, FunctionSpace,  # noqa: E402
                    SpatialCoordinate, dot, sqrt)

#: The lid depth in units of D.
LID = prepare.LID_DEPTH_M / common.D_SCALE


class FakeScales:
    """The one attribute of `exp09_common.Scales` that the EB code reads."""

    Re = 6371.0e3 / common.D_SCALE


def box_fields():
    """A box of 3 cell layers in z below Re, the top layer above 90 km.

    The box is thin in x and y (1e-3 D) and sits on the z axis, so the
    radius of every point is its z to about 1e-7 D, and the face between
    the first and the second cell layer is the 90 km sphere to that
    accuracy, far inside the cells' centroid spacing.

    Returns:
      `(mesh, fields)`: `shear_modulus` (DG1, varying with radius, as
      PREM does), `viscosity` (DG0: ETA_LID above 90 km, 3.16 below).
    """
    re = FakeScales.Re
    mesh = BoxMesh(2, 2, 3, 1e-3, 1e-3, 3 * LID)
    # Shift z so that the box spans [Re - 3 LID, Re].
    mesh.coordinates.dat.data[:, 2] += re - 3 * LID
    X = SpatialCoordinate(mesh)
    r = sqrt(dot(X, X))
    mu = Function(FunctionSpace(mesh, "DG", 1), name="shear_modulus")
    mu.interpolate(0.6 + 2.0 * (re - r))
    dg0 = FunctionSpace(mesh, "DG", 0)
    centroid = Function(dg0).interpolate(r)
    eta = Function(dg0, name="viscosity")
    eta.dat.data[:] = np.where(re - centroid.dat.data_ro < LID,
                               common.ETA_LID / common.ETA_BAR, 3.16)
    return mesh, {"shear_modulus": mu, "viscosity": eta}


def fit():
    """The S31 fit from the fit file of the directory."""
    return prepare.load_eb_fit(prepare.EB_FIT_FILE, prepare.EB_FIT_KEY)


def test_load_eb_fit_reads_the_free_time_fit():
    mu_bar, tau_bar, record = fit()
    with open(prepare.EB_FIT_FILE) as handle:
        entry = json.load(handle)["free_time_fits"]["3"]
    assert np.array_equal(mu_bar, entry["mu_bar"])
    assert np.array_equal(tau_bar, entry["tau_bar"])
    assert entry["free_times"] is True
    assert abs(np.sum(mu_bar * tau_bar) - 1.0) < 1e-12
    assert 0.69 < mu_bar.sum() < 0.71
    assert record["fit_file_sha256"] == common.sha256sum(prepare.EB_FIT_FILE)


@pytest.mark.parametrize("change", ["delta", "sum", "floor"])
def test_load_eb_fit_refuses_a_bad_fit(tmp_path, change):
    with open(prepare.EB_FIT_FILE) as handle:
        data = json.load(handle)
    entry = data["free_time_fits"]["3"]
    if change == "delta":
        data["parameters"]["Delta"] = 1.04
    elif change == "sum":
        entry["tau_bar"][-1] *= 1.01
    else:
        entry["mu_bar"][0] = 0.0
    path = tmp_path / "eb_fit.json"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        prepare.load_eb_fit(str(path), prepare.EB_FIT_KEY)


def test_burgers_elements_values_and_reader():
    mesh, fields = box_fields()
    mu_bar, tau_bar, _ = fit()
    moduli, viscosities, lid_cells, errors = prepare.burgers_elements(
        mesh, fields, FakeScales, mu_bar, tau_bar)
    # One third of the 2 x 2 x 3 x 6 tetrahedra lie in the lid.
    assert lid_cells == mesh.num_cells() // 3
    assert all(e < 1e-14 for e in errors.values())
    mu_inf = fields["shear_modulus"].dat.data_ro
    eta0 = fields["viscosity"].dat.data_ro
    lid0 = eta0 == common.ETA_LID / common.ETA_BAR
    w = mu_bar / mu_bar.sum()
    # The solver's reader returns the columns of the vector fields.
    mu_j, eta_j = common.element_fields({"shear_moduli": moduli,
                                         "viscosities": viscosities})
    assert len(mu_j) == len(eta_j) == 3
    dg1 = mu_j[0].function_space()
    lid1 = Function(dg1).interpolate(fields["viscosity"]).dat.data_ro \
        == common.ETA_LID / common.ETA_BAR
    for j in range(3):
        m = mu_j[j].dat.data_ro
        e = eta_j[j].dat.data_ro
        assert np.allclose(m[~lid1], mu_inf[~lid1] * mu_bar[j], rtol=1e-15)
        assert np.allclose(m[lid1], mu_inf[lid1] * w[j], rtol=1e-15)
        assert np.allclose(e[~lid0], eta0[~lid0] * mu_bar[j] * tau_bar[j],
                           rtol=1e-15)
        assert np.allclose(e[lid0], eta0[lid0] * w[j], rtol=1e-15)
        # The Maxwell time of element j at every DG1 node below the lid.
        tau = Function(dg1).interpolate(eta_j[j] / mu_j[j]).dat.data_ro
        tau_m = Function(dg1).interpolate(
            fields["viscosity"] / fields["shear_modulus"]).dat.data_ro
        assert np.allclose(tau[~lid1], tau_bar[j] * tau_m[~lid1],
                           rtol=1e-13)


def test_burgers_by_layer():
    mesh, fields = box_fields()
    mu_bar, tau_bar, _ = fit()
    moduli, viscosities, _, _ = prepare.burgers_elements(
        mesh, fields, FakeScales, mu_bar, tau_bar)
    re = FakeScales.Re
    layers = [("lid", re - LID, re), ("below", re - 3 * LID, re - LID)]
    cells = prepare.LayerCells(mesh, layers)
    out = prepare.burgers_by_layer(cells, layers, fields, moduli,
                                   viscosities)
    lid = out["lid"]["instant_modulus_ratio"]
    below = out["below"]["instant_modulus_ratio"]
    assert abs(lid["min"] - 1) < 1e-13 and abs(lid["max"] - 1) < 1e-13
    assert abs(below["mean"] - mu_bar.sum()) < 1e-13
    # Element 0 below the lid: tau_bar_0 * 3.16 / mu t_bar, about 0.07 t_bar
    # = 23 yr for mu about 0.6-0.7, so it is shorter than 50 yr and longer
    # than 10 yr everywhere; the lid elements are far longer than any step.
    e0 = out["below"]["elements"][0]
    assert 10.0 < e0["maxwell_time_yr"]["min"] < e0["maxwell_time_yr"]["max"] < 50.0
    assert e0["volume_fraction"]["below_50_yr"] == pytest.approx(1.0)
    assert e0["volume_fraction"]["below_10_yr"] == pytest.approx(0.0)
    for e in out["lid"]["elements"]:
        assert e["maxwell_time_yr"]["min"] > 1e18
        assert e["volume_fraction"]["below_2000_yr"] == 0.0


def test_lid_check_refuses_a_shifted_lid():
    mesh, fields = box_fields()
    mu_bar, tau_bar, _ = fit()
    # Make one cell below 90 km elastic: the depth rule and the viscosity
    # now disagree in one cell.
    eta = fields["viscosity"].dat.data
    first = int(np.flatnonzero(eta != common.ETA_LID / common.ETA_BAR)[0])
    eta[first] = common.ETA_LID / common.ETA_BAR
    with pytest.raises(RuntimeError, match="1 cells differ"):
        prepare.burgers_elements(mesh, fields, FakeScales, mu_bar, tau_bar)


def test_record_pieces_are_json(tmp_path):
    """The EB record and the field list go through the JSON writer."""
    _, _, record = fit()
    full = {"n_elements": 3, "burgers": record}
    full["fields"] = {
        name: (list(common.MANTLE_FIELDS[name]) if name in common.MANTLE_FIELDS
               else list(common.OPTIONAL_FIELDS[name][:2]) + ["vector of 3"])
        for name in common.field_names(full)}
    assert "shear_moduli" in full["fields"] and "viscosities" in full["fields"]
    path = tmp_path / "record.json"
    common.write_json(str(path), full)
    back = json.loads(path.read_text())
    assert back["burgers"]["mu_bar"] == list(record["mu_bar"])
    assert json.dumps(common.plain_json(full))
