from pathlib import Path


def check_and_get_absolute_paths(base_path: Path, filenames: dict):
    # Check if all files are present
    all_files_present = all((base_path / filename).exists() for files in filenames.values() for filename in files)

    if not all_files_present:
        raise FileNotFoundError("Some files are missing. Cannot proceed without downloading the required files.")

    # Return absolute paths of the files
    return {
        key: [str(base_path / filename) for filename in files]
        for key, files in filenames.items()
    }


def obtain_Muller_2022_SE_v1_2_4(base_path: str | Path):

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
    base_path = Path(base_path)
    base_path = base_path / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2"

    # Pass the base_path and filenames to the helper function
    return check_and_get_absolute_paths(base_path, plate_reconstruction_files)


def obtain_Muller_2022_SE_v1_2(base_path: str | Path):

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
    base_path = Path(base_path)
    base_path = base_path / "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2"

    # Pass the base_path and filenames to the helper function
    return check_and_get_absolute_paths(base_path, plate_reconstruction_files)
