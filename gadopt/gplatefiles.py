from pathlib import Path
import zipfile
import pooch


def obtain_Muller_2022_SE(download_path: str | Path, download_mode: bool = False):
    muller_url = "https://earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2022_SE/Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2.zip"

    plate_reconstruction_files = {
        "rotation_filenames": ["optimisation/1000_0_rotfile_MantleOptimised.rot"],
        "topology_filenames": [
            "250-0_plate_boundaries.gpml",
            "410-250_plate_boundaries.gpml",
            "1000-410-Convergence.gpml",
            "1000-410-Divergence.gpml",
            "1000-410-Topologies.gpml",
            "1000-410-Transforms.gpml"
        ]
    }

    # This is the path to download and extract files
    download_path = Path(download_path)
    base_path = download_path / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2"

    # Check if all files are already present
    all_files_present = all((base_path / filename).exists() for filenames in plate_reconstruction_files.values() for filename in filenames)

    if download_mode:
        if not all_files_present:
            cache_plate_reconstruction_model(
                url=muller_url,
                download_dir=download_path,
                files_2_extract=[filename for filenames in plate_reconstruction_files.values() for filename in filenames]
            )
    else:
        if not all_files_present:
            raise FileNotFoundError("Some files are missing and download_mode is set to False. Cannot proceed without downloading the required files.")

    plate_reconstruction_files_with_path = {
        key: [str(base_path / filename) for filename in filenames]
        for key, filenames in plate_reconstruction_files.items()
    }

    return plate_reconstruction_files_with_path


def cache_plate_reconstruction_model(url: str, download_path: str | Path, files_2_extract: list[str]):
    # Define the URL of the file and the local storage path
    filename = Path(url).name
    filename_wo_extension = filename.rsplit(".zip", 1)[0]

    # This is where to download
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    # Download the file
    zip_path = pooch.retrieve(
        url=url,
        known_hash=None,  # No hash check
        path=download_path,
        fname=filename
    )

    files_2_extract = [Path(filename_wo_extension) / fi for fi in files_2_extract]

    # Unzip the downloaded file
    with zipfile.ZipFile(zip_path, 'r') as zipfi:
        zipfi.extractall(download_path, members=files_2_extract)
