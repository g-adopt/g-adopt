from pathlib import Path


def obtain_Muller_2022_SE(base_path: str | Path):

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

    # Check if all files are already present
    all_files_present = all((base_path / filename).exists() for filenames in plate_reconstruction_files.values() for filename in filenames)

    if not all_files_present:
        raise FileNotFoundError("Some files are missing. Cannot proceed without downloading the required files.")

    plate_reconstruction_files_with_path = {
        key: [str(base_path / filename) for filename in filenames]
        for key, filenames in plate_reconstruction_files.items()
    }

    return plate_reconstruction_files_with_path
