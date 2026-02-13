import argparse
import importlib


cases = {
    "smooth": {
        "cylindrical": {
            "freeslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "k": [2, 8],
                "n": [1, 4],
            },
            "zeroslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "k": [2, 8],
                "n": [1, 4],
            },
            "freesurface": {
                "cores": [1, 4, 6],
                "levels": [2**i for i in [1, 2, 3]],
                "k": [2, 8],
                "n": [1, 4],
            },
        },
        "spherical": {
            "freeslip": {
                "cores": [48, 96, 192],  # cascade lake
                "levels": [4, 5, 6],
                "l": [2, 8],
                "m": [2, 1],  # divide l by this value to get actual m
                "k": [3, 9],
                "permutate": False,
            },
            "zeroslip": {
                "cores": [48, 96, 192],
                "levels": [4, 5, 6],
                "l": [2, 8],
                "m": [2, 1],
                "k": [3, 9],
                "permutate": False,
            },
        },
    },
    "delta": {
        "cylindrical": {
            "freeslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "freeslip_dpc": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "zeroslip": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
            "zeroslip_dpc": {
                "cores": [4, 16, 24],
                "levels": [2**i for i in [2, 3, 4]],
                "n": [2, 8],
            },
        },
    },
}


def get_case(cases, config):
    config = config.split("_", maxsplit=2)
    while config:
        cases = cases[config.pop(0)]

    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="analytical",
        description="Run/submit analytical test cases",
    )
    parser.add_argument("case")
    parser.add_argument("params", type=int, nargs="*")

    args = parser.parse_args()

    from mpi4py import MPI

    try:
        model = importlib.import_module(args.case).model
    except ModuleNotFoundError:
        raise ValueError(f"unknown case {args.case}")

    errors = model(*args.params)
    if MPI.COMM_WORLD.rank == 0:
        config = get_case(cases, args.case)
        config.pop("cores")
        errfile_name = "errors-{}-{}.dat".format(
            args.case.replace("/", "_"),
            "-".join([f"{k}{v}" for k, v in zip(config.keys(), args.params)]),
        )

        with open(errfile_name, "w") as f:
            f.write(" ".join(str(x) for x in errors))
