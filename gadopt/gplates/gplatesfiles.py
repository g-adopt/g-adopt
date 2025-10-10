from pathlib import Path

_default_muller2022_plate_files = {
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

reconstructions = {
    "Muller 2022 SE v1.2": {
        "plate_files": _default_muller2022_plate_files,
        "directory": "Muller_etal_2022_SE_1Ga_Opt_PlateMotionModel_v1.2",
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


def ensure_reconstruction(reconstruction: str, base_path: str | Path):
    if reconstruction not in reconstructions:
        raise ValueError(f"Invalid reconstruction dataset {reconstruction}")

    base_path = Path(base_path) / reconstructions[reconstruction]["directory"]

    return check_and_get_absolute_paths(base_path, reconstructions[reconstruction]["plate_files"])
