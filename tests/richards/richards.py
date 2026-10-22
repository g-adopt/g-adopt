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
                "levels": [76, 151, 301],
                "cores": [2, 2, 8],
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
    """Look up (module_name, config) by flat name, e.g. 'tracy_2d_specified_head_dg1'.

    The cases dict is nested as cases[tracy_2d][specified_head][dg1]. Rather than
    parsing the flat name (which is fragile because '_' is both the separator and
    appears inside the module/BC/degree keys), match it against the names the
    nesting actually produces.
    """
    for module_name, bc_dict in cases.items():
        for bc_type, degree_dict in bc_dict.items():
            for degree_key, config in degree_dict.items():
                if f"{module_name}_{bc_type}_{degree_key}" == name:
                    return module_name, config
    raise KeyError(f"unknown case {name}")


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

    # Resolve the module the case lives in (e.g. tracy_2d) without parsing.
    module_name, _ = get_case(cases, args.case)
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
