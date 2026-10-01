from gadopt.gplates.gplatesfiles import (
    reconstructions,
    ensure_reconstruction,
    MissingReconstructionException,
)
import pytest
import urllib.request
import zipfile
from unittest.mock import patch

DUMMY_RECONSTRUCTION = {
    "plate_files": {
        "topology_filenames": ["a", "b"],
        "static_polygons": "c",
        "continental_polygons": "d",
    },
    "directory": "dummy",
    "url": "https://data.gadopt.org/github-actions/dummy.zip",
}

DUMMY_RECONSTRUCTION_NO_URL = {
    "plate_files": {
        "topology_filenames": ["a", "b"],
        "static_polygons": "c",
        "continental_polygons": "d",
    },
    "directory": "dummy",
}

DUMMY_RECONSTRUCTION_EXTRA_FILE = {
    "plate_files": {
        "topology_filenames": ["a", "b"],
        "static_polygons": "c",
        "continental_polygons": ["d", "e"],
    },
    "directory": "dummy",
    "url": "https://data.gadopt.org/github-actions/dummy.zip",
}

DUMMY_RECONSTRUCTION_NOT_ZIPFILE = {
    "directory": "dummy",
    "url": "https://data.gadopt.org/github-actions/not_a_zip_file.txt",
}

DUMMY_RECONSTRUCTION_WRONG_DIR = {
    "plate_files": {
        "topology_filenames": ["a", "b"],
        "static_polygons": "c",
        "continental_polygons": "d",
    },
    "directory": "wrong",
    "url": "https://data.gadopt.org/github-actions/dummy.zip",
}

DUMMY_RECONSTRUCTION_WRONG_URL = {
    "directory": "dummy",
    "url": "https://data.gadopt.org/404.zip",
}


@pytest.fixture
def mock_reconstruction(monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION)


def test_invalid_reconstruction(tmpdir):
    with pytest.raises(ValueError, match="Invalid reconstruction"):
        ensure_reconstruction("dummy", tmpdir)


def test_download_reconstruction(tmpdir, mock_reconstruction):
    ensure_reconstruction("dummy", tmpdir)
    assert (tmpdir / "dummy").isdir()
    assert all((tmpdir / "dummy" / fn).exists() for fn in "abcd")


def test_unzip_reconstruction(tmpdir, mock_reconstruction):
    # predownload reconstruction
    resp = urllib.request.urlopen(DUMMY_RECONSTRUCTION["url"])
    with open(tmpdir / "dummy.zip", "wb") as f:
        f.write(resp.read())
    ensure_reconstruction("dummy", tmpdir)
    assert (tmpdir / "dummy").isdir()
    assert all((tmpdir / "dummy" / fn).exists() for fn in "abcd")


def test_existing_reconstruction(tmpdir, mock_reconstruction):
    resp = urllib.request.urlopen(DUMMY_RECONSTRUCTION["url"])
    with open(tmpdir / "dummy.zip", "wb") as f:
        f.write(resp.read())
    zipfile.ZipFile(tmpdir / "dummy.zip").extractall()
    ensure_reconstruction("dummy", tmpdir)
    assert (tmpdir / "dummy").isdir()
    assert all((tmpdir / "dummy" / fn).exists() for fn in "abcd")


def test_bad_zipfile(tmpdir, mock_reconstruction):
    with open(tmpdir / "dummy.zip", "w") as f:
        f.write("this is not a zip file")
    ensure_reconstruction("dummy", tmpdir)
    assert (tmpdir / "dummy").isdir()
    assert all((tmpdir / "dummy" / fn).exists() for fn in "abcd")


def test_download_error(tmpdir, monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION_WRONG_URL)
    with pytest.raises(MissingReconstructionException, match="Error retrieving"):
        ensure_reconstruction("dummy", tmpdir)


def test_download_not_a_zipfile(tmpdir, monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION_NOT_ZIPFILE)
    with pytest.raises(MissingReconstructionException, match="Unexpected content type"):
        ensure_reconstruction("dummy", tmpdir)


def test_missing_files(tmpdir, monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION_EXTRA_FILE)
    with pytest.raises(MissingReconstructionException, match="An object with the expected name"):
        ensure_reconstruction("dummy", tmpdir)


def test_wrong_directory(tmpdir, monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION_WRONG_DIR)
    resp = urllib.request.urlopen(DUMMY_RECONSTRUCTION_WRONG_DIR["url"])
    with open(tmpdir / "dummy.zip", "wb") as f:
        f.write(resp.read())
    # ensure reconstruction is expecting this zip file to contain a directory
    # named 'wrong'. When that isn't there, it will trigger a re-download, which
    # will then fail with missing files. Make sure the download_reconstruction
    # function was called
    with patch("gadopt.gplates.gplatesfiles.download_reconstruction") as mock:
        with pytest.raises(
            MissingReconstructionException, match=r"Reconstruction \(wrong\) not found"
        ):
            ensure_reconstruction("dummy", tmpdir)
        mock.assert_called_once()


def test_no_url_provided(tmpdir, monkeypatch):
    monkeypatch.setitem(reconstructions, "dummy", DUMMY_RECONSTRUCTION_NO_URL)
    with patch("gadopt.gplates.gplatesfiles.extract_zip_reconstruction") as mock:
        with pytest.raises(
            MissingReconstructionException, match="Please download manually and extract to"
        ):
            ensure_reconstruction("dummy", tmpdir)
        mock.assert_not_called()


def test_empty_directory(tmpdir, mock_reconstruction):
    (tmpdir / "dummy").mkdir()
    with pytest.raises(MissingReconstructionException, match="An object with the expected name"):
        ensure_reconstruction("dummy", tmpdir)


def test_download_no_write_permission(mock_reconstruction):
    with pytest.raises(PermissionError, match="You have attempted"):
        ensure_reconstruction("dummy", "/a/b/c/d")
