from gadopt.gplates.gplatesfiles import (
    reconstructions,
    ensure_reconstruction,
    MissingReconstructionException,
)
import pytest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

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

TEST_DATA_PATH = Path(__file__).resolve().parent / "data"


@pytest.fixture
def mock_urlopen():

    with open(TEST_DATA_PATH / "dummy.zip", "rb") as f:
        zf = f.read()

    response = MagicMock()

    with (
        patch("urllib.request.urlopen", return_value=response) as urlopen,
        patch.object(
            type(response),
            "headers",
            new_callable=PropertyMock,
            create=True,
        ) as headers,
        patch.object(response, "read", return_value=zf) as read,
    ):
        headers.return_value = MagicMock()
        headers.return_value.get_content_type.return_value = "application/zip"
        yield urlopen, headers, read


@pytest.fixture
def mock_reconstruction():
    with patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION}) as r:
        yield r


def test_invalid_reconstruction(tmp_path):
    with pytest.raises(ValueError, match="Invalid reconstruction"):
        ensure_reconstruction("dummy", tmp_path)


def test_download_reconstruction(tmp_path, mock_reconstruction, mock_urlopen):
    urlopen, headers, read = mock_urlopen
    ensure_reconstruction("dummy", tmp_path)
    assert (tmp_path / "dummy").is_dir()
    assert all((tmp_path / "dummy" / fn).exists() for fn in "abcd")
    urlopen.assert_called_once_with("https://data.gadopt.org/github-actions/dummy.zip")
    headers.return_value.get_content_type.assert_called_once()
    read.assert_called_once()


def test_unzip_reconstruction(tmp_path, mock_reconstruction, mock_urlopen):
    # predownload reconstruction
    urlopen, _, _ = mock_urlopen
    (tmp_path / "dummy.zip").write_bytes((TEST_DATA_PATH / "dummy.zip").read_bytes())
    ensure_reconstruction("dummy", tmp_path)
    assert (tmp_path / "dummy").is_dir()
    assert all((tmp_path / "dummy" / fn).exists() for fn in "abcd")
    urlopen.assert_not_called()


def test_existing_reconstruction(tmp_path, mock_reconstruction, mock_urlopen):
    urlopen, _, _ = mock_urlopen
    (tmp_path / "dummy.zip").write_bytes((TEST_DATA_PATH / "dummy.zip").read_bytes())
    zipfile.ZipFile(tmp_path / "dummy.zip").extractall()
    ensure_reconstruction("dummy", tmp_path)
    assert (tmp_path / "dummy").is_dir()
    assert all((tmp_path / "dummy" / fn).exists() for fn in "abcd")
    urlopen.assert_not_called()


def test_bad_zipfile(tmp_path, mock_reconstruction, mock_urlopen):
    urlopen, headers, read = mock_urlopen
    with open(tmp_path / "dummy.zip", "w") as f:
        f.write("this is not a zip file")
    ensure_reconstruction("dummy", tmp_path)
    assert (tmp_path / "dummy").is_dir()
    assert all((tmp_path / "dummy" / fn).exists() for fn in "abcd")
    urlopen.assert_called_once_with("https://data.gadopt.org/github-actions/dummy.zip")
    headers.return_value.get_content_type.assert_called_once()
    read.assert_called_once()


def test_download_error(tmp_path, mock_urlopen):
    urlopen, _, _ = mock_urlopen
    urlopen.side_effect = urllib.error.HTTPError(
        url=DUMMY_RECONSTRUCTION_WRONG_URL["url"],
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=None,
    )
    with (
        pytest.raises(MissingReconstructionException, match="Error retrieving"),
        patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION_WRONG_URL}),
    ):
        ensure_reconstruction("dummy", tmp_path)


def test_download_not_a_zipfile(tmp_path, mock_urlopen):
    urlopen, headers, read = mock_urlopen
    headers.return_value.get_content_type.return_value = "text/plain"
    with (
        pytest.raises(MissingReconstructionException, match="Unexpected content type"),
        patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION_NOT_ZIPFILE}),
    ):
        ensure_reconstruction("dummy", tmp_path)
    urlopen.assert_called_once_with(
        "https://data.gadopt.org/github-actions/not_a_zip_file.txt"
    )
    read.assert_not_called()


def test_missing_files(tmp_path, mock_urlopen):
    urlopen, headers, read = mock_urlopen
    with (
        pytest.raises(
            MissingReconstructionException, match="An object with the expected name"
        ),
        patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION_EXTRA_FILE}),
    ):
        ensure_reconstruction("dummy", tmp_path)
    urlopen.assert_called_once_with("https://data.gadopt.org/github-actions/dummy.zip")
    headers.return_value.get_content_type.assert_called_once()
    read.assert_called_once()


def test_wrong_directory(tmp_path, mock_urlopen):
    urlopen, headers, read = mock_urlopen
    (tmp_path / "dummy.zip").write_bytes((TEST_DATA_PATH / "dummy.zip").read_bytes())
    # ensure reconstruction is expecting this zip file to contain a directory
    # named 'wrong'. When that isn't there, it will trigger a re-download, which
    # will then fail with missing files. Make sure it went through the motions to
    # download the reconstruction
    with (
        pytest.raises(
            MissingReconstructionException, match=r"Reconstruction \(wrong\) not found"
        ),
        patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION_WRONG_DIR}),
    ):
        ensure_reconstruction("dummy", tmp_path)
    urlopen.assert_called_once_with("https://data.gadopt.org/github-actions/dummy.zip")
    headers.return_value.get_content_type.assert_called_once()
    read.assert_called_once()


def test_no_url_provided(tmp_path, mock_urlopen):
    urlopen, _, _ = mock_urlopen
    with (
        pytest.raises(
            MissingReconstructionException,
            match="Please download manually and extract to",
        ),
        patch.dict(reconstructions, {"dummy": DUMMY_RECONSTRUCTION_NO_URL}),
    ):
        ensure_reconstruction("dummy", tmp_path)
    urlopen.assert_not_called()


def test_empty_directory(tmp_path, mock_reconstruction):
    (tmp_path / "dummy").mkdir()
    with pytest.raises(
        MissingReconstructionException, match="An object with the expected name"
    ):
        ensure_reconstruction("dummy", tmp_path)
