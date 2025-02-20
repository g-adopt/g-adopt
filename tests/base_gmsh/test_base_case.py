import pandas as pd
from pathlib import Path
import hashlib
import pytest
import warnings

# Different variations of gmsh come up with different meshes from the geo file.
# If we see a mesh different to what we're expecting, warn the user and set
# the tests to xfail
b = Path(__file__).parent.resolve()
expected_md5 = "41b8157a5e18f467dbd1e7538d1236d6"
def hash_check():
    with open(b / "square.msh", "r") as f:
        mesh_md5 = hashlib.md5(f.read().encode()).hexdigest()
    if mesh_md5 != expected_md5:
        warn_str = f"Known good md5sum: {expected_md5}, Created mesh md5sum: {mesh_md5}"
        warnings.warn(UserWarning(warn_str))
    return mesh_md5 != expected_md5


@pytest.mark.xfail(hash_check(), reason="Mesh file has changed since known good output was created")
def test_base_case():
    df = pd.read_csv(b / "params.log", sep="\\s+", header=0).iloc[-1]
    expected = pd.read_pickle(b / "expected.pkl")
    kwargs = {"check_names": False}
    different_mesh = hash_check()
    if not different_mesh:
        kwargs |= {"rtol": 1e-4}
    pd.testing.assert_series_equal(df[["u_rms", "nu_top"]], expected, **kwargs)
    if different_mesh:
        assert abs(df.name - expected.name) <= 2
