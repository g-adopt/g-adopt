import argparse
import importlib

cases = {
    "tracy_2d": {
        "specified_head": {
            "dg0": {
                "levels": [51, 101, 201, 401],
                "cores": [1, 1, 4, 16],
                "degree": 0,
                "bc_type": "specified_head",
            },
            "dg1": {
                "levels": [51, 101, 201, 401],
                "cores": [1, 1, 4, 16],
                "degree": 1,
                "bc_type": "specified_head",
            },
            "dg2": {
                "levels": [76, 151, 301, 601],
                "cores": [2, 2, 8, 32],
                "degree": 2,
                "bc_type": "specified_head",
            },
        },
        "no_flux": {
            "dg1": {
                "levels": [51, 101, 201, 401],
                "cores": [1, 1, 4, 16],
                "degree": 1,
                "bc_type": "no_flux",
            },
        },
    },
}


def get_case(cases, name):
    """Look up a case by its flat name, e.g. 'tracy_2d_specified_head_dg1'.

    The cases dict is nested as cases[tracy_2d][specified_head][dg1].
    """
    # Split into module prefix and the rest
    parts = name.split("_")
    # First two parts form the module key (e.g. tracy_2d)
    module_key = "_".join(parts[:2])
    # Middle parts are the BC type (e.g. specified_head or no_flux)
    # Last part is the degree key (e.g. dg0, dg1, dg2)
    degree_key = parts[-1]
    bc_key = "_".join(parts[2:-1])
    return cases[module_key][bc_key][degree_key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="richards",
        description="Run Richards equation benchmark cases",
    )
    parser.add_argument("case")
    parser.add_argument("nodes", type=int)
    parser.add_argument("degree", type=int)
    parser.add_argument("bc_type", type=str)

    args = parser.parse_args()

    from mpi4py import MPI

    # Map case name to module: tracy_2d_specified_head_dg1 -> tracy_2d
    module_name = "_".join(args.case.split("_")[:2])
    try:
        model = importlib.import_module(module_name).model
    except ModuleNotFoundError:
        raise ValueError(f"unknown case {args.case} (module {module_name} not found)")

    errors = model(args.nodes, degree=args.degree, bc_type=args.bc_type)

    if MPI.COMM_WORLD.rank == 0:
        errfile = f"errors-{args.case}-nodes{args.nodes}-dq{args.degree}.dat"
        with open(errfile, "w") as f:
            f.write(" ".join(str(x) for x in errors))
        print(f"Wrote {errfile}")
