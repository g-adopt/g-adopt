from collections.abc import Sequence, Mapping
from pathlib import Path
import urllib.request
from urllib.error import URLError
import zipfile
from io import BytesIO

ReconstructionMetadataType = Mapping[str, Mapping[str, str | Mapping[str, str | Sequence[str]]]]

_default_muller2022_plate_files: Mapping[str, str | Sequence[str]] = {
    "rotation_filenames": ["optimisation/1000_0_rotfile_MantleOptimised.rot"],
    "topology_filenames": [
        "250-0_plate_boundaries.gpml",
        "410-250_plate_boundaries.gpml",
        "1000-410-Convergence.gpml",
        "1000-410-Divergence.gpml",
        "1000-410-Topologies.gpml",
        "1000-410-Transforms.gpml"
    ],
    "continental_polygons": "shapes_continents.gpml",
    "static_polygons": "shapes_static_polygons_Merdith_et_al.gpml",
}

reconstructions: ReconstructionMetadataType = {
    "Muller 2022 SE v1.2": {
        "plate_files": _default_muller2022_plate_files,
        "directory": "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2",
        "url": "https://earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2022_SE/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.zip",
    },
    "Muller 2022 SE v1.2.4": {
        "plate_files": _default_muller2022_plate_files,
        "directory": "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.4",
    },
    # Cao et al 2024 extends Muller 2022 to 1.8 Byrs.
    # DOI: 10.1016/j.gsf.2024.101922
    # Zenodo: 11536687
    "Cao 2024": {
        "plate_files": {
            "rotation_filenames": [
                "optimisation/1800_1000_rotfile_20240725_run3.rot",
                "optimisation/1000_0_rotfile_20240725_run3.rot"
            ],
            "topology_filenames":
                _default_muller2022_plate_files["topology_filenames"] +
                ["1800-1000_plate_boundaries.gpml", "TopologyBuildingBlocks.gpml"],
        },
        "directory": "1.8Ga_model_optimised_mantle_ref_frame_20240725",
    },
    "Zahirovic 2022": {
        "plate_files": {
            "rotation_filenames": [
                "Zahirovic2022_CombinedRotations_fixed_crossovers.rot",
            ],
            "topology_filenames": [
                "Zahirovic2022_ActiveDeformation.gpmlz",
                "Zahirovic2022_InactiveDeformation.gpmlz",
                "Zahirovic2022_PlateBoundaries.gpmlz",
            ],
        },
        "directory": "Zahirovic_2022",
    },
}


class MissingReconstructionException(Exception):
    """Custom exception for missing reconstructions

    An exception that provides a more helpful error message to users when a plate
    reconstruction is unable to be found or downloaded.

    Args:
        reconstruction: Name of the reconstruction. Must be a key in the reconstructions
        dict defined above
        path: Destination directory
        msg: Message to pass on to the user
    """
    def __init__(self, reconstruction: str, path: Path, msg: str):
        url = reconstructions[reconstruction].get("url")
        if url is None:
            exc_str = f"Error retrieving {reconstruction}: {msg}. Please download manually and extract to {path.parent}"
        else:
            exc_str = f"Error retrieving {reconstruction}: {msg}. Please download manually from {url} and extract to {path.parent}"
        super().__init__(exc_str)


def extract_zip_reconstruction(data: Path | bytes, path: Path) -> None:
    """Extracts a reconstruction

    Creates a ZipFile object from either a filename or downloaded bytes and extracts
    it to the provided path. Note that any exceptions from zipfile are intentionally
    left to the caller to handle.

    Args:
        data: Either a filename or a bytes object containing a zipfile
        path: Destination directory
    """
    if isinstance(data, bytes):
        zf = zipfile.ZipFile(BytesIO(data))
    else:
        zf = zipfile.ZipFile(data)
    zf.extractall(path)


def download_reconstruction(reconstruction: str, base_path: Path) -> None:
    """Download a plate reconstruction

    Uses urllib to retrieve a plate reconstruction. Checks that the downloaded data
    is of the correct type and passes it on to an extraction function. If a URL is
    not provided for the given reconstruction, return immediately.

    Args:
        reconstruction: Name of the reconstruction
        base_path: Destination directory

    Raises:
        MissingReconstructionError: The reconstruction could not be downloaded from the
        provided URL.
    """
    url = reconstructions[reconstruction].get("url")
    if url is None:
        return

    try:
        resp = urllib.request.urlopen(url)
    except URLError as e:
        # Something went wrong at a network level
        raise MissingReconstructionException(reconstruction, base_path, str(e))

    # Add more types and extraction mechanisms here if necessary
    if resp.headers.get_content_type() in (
        "application/zip",
        "application/x-zip-compressed",
    ):
        extract_zip_reconstruction(resp.read(), base_path.parent)
    else:
        # Downloaded something that wasn't a zipfile
        raise MissingReconstructionException(
            reconstruction,
            base_path,
            f"Unexpected content type: got {resp.headers.get_content_type()}, expected application/zip",
        )


def check_and_get_absolute_paths(base_path: Path, reconstruction: str) -> dict[str, list[str]]:
    """Checks that a reconstruction exists or downloads it

    If an object exists in base_path with the directory name expected by the provided
    reconstruction, check that all expected files are present. If no such object exists,
    check to see if there is an appropriate zip file in base_path and attempt to extract
    it. If extraction fails, or if there are no zip files, download the reconstruction
    from its url.

    Args:
        base_path: Top-level directory to extract the reconstruction
        reconstruction: Name of the reconstruction

    Raises:
        PermissionError: Attempted to download to a directory without write permissions
        MissingReconstructionException: The present/downloaded reconstruction did not contain
        the required files

    Returns:
        A dictionary with file types as keys and lists of filenames as values.
    """
    # Normalize all values to lists
    def to_list(value):
        if isinstance(value, str):
            return [value]
        elif isinstance(value, Sequence):
            return list(value)
        else:
            return [value]

    # Is the directory already there?
    if not base_path.exists():
        # Make sure the top-level directory is there
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Re-raise with a more useful message
            raise PermissionError(
                f"You have attempted download the reconstruction {reconstruction} to the directory {base_path.parent}, which you do not have permission to write to."
            )
        # Is there a zipfile present that we can extract? Try base_path/<directory>.zip
        # and if a URL has been provided, base_path/<url path basename>
        possible_zip_files = [base_path.with_suffix(".zip")]
        url = reconstructions[reconstruction].get("url")
        if url is not None:
            possible_zip_files.append(base_path / Path(url.split('?')[0]).name)
        for zippath in possible_zip_files:
            if zippath.exists():
                try:
                    extract_zip_reconstruction(zippath, base_path.parent)
                except zipfile.BadZipFile:
                    # The zipfile wasn't a zipfile
                    download_reconstruction(reconstruction, base_path)
                else:
                    # The zipfile extracted successfully but didn't contain the directory
                    # we needed
                    if not base_path.exists():
                        download_reconstruction(reconstruction, base_path)
                break
        else:
            download_reconstruction(reconstruction, base_path)

    filenames = reconstructions[reconstruction]["plate_files"]
    # Check if all files are present
    if not base_path.exists():
        raise MissingReconstructionException(
            reconstruction,
            base_path,
            f"Reconstruction ({reconstructions[reconstruction]['directory']}) not found",
        )

    all_files_present = all(
        (base_path / filename).exists()
        for files in filenames.values()
        for filename in to_list(files)
    )

    if not all_files_present:
        raise MissingReconstructionException(
            reconstruction,
            base_path,
            f"An object with the expected name ({reconstructions[reconstruction]['directory']}) exists, but does not contain all of the required files",
        )

    # Return absolute paths of the files, normalized to lists
    return {
        key: [str(base_path / filename) for filename in to_list(files)]
        for key, files in filenames.items()
    }


def ensure_reconstruction(reconstruction: str, base_path: str | Path) -> dict[str, list[str]]:
    """User entrypoint for reconstruction gathering

    Args:
        base_path: Top-level directory to extract the reconstruction
        reconstruction: Name of the reconstruction

    Raises:
        ValueError: Invalid reconstruction passed

    Returns:
        A dictionary with file types as keys and lists of filenames as values.
    """
    if reconstruction not in reconstructions:
        raise ValueError(f"Invalid reconstruction dataset {reconstruction}")

    base_path = Path(base_path) / reconstructions[reconstruction]["directory"]

    return check_and_get_absolute_paths(base_path, reconstruction)
