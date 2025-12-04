import argparse
import asyncio
import os
import sys

# Configuration for Richards equation test cases
cases = {
    "tracy_2d_specified_head": {
        "cores": [1],
        "levels": [1, 2, 3],
    },
    "tracy_2d_no_flux": {
        "cores": [1],
        "levels": [1, 2, 3],
    },
    "vauclin_2d": {
        "cores": [1],
        "levels": [1, 2],
    },
}


def get_case(cases, config):
    """Get case configuration by name."""
    return cases[config]


def param_sets(config):
    """Generate parameter combinations."""
    return zip(*config.values())


async def run_subproc(args, level, cores, params):
    """Run a single test case."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Import the appropriate test module
    case_name = args.case
    if case_name.startswith("tracy_2d"):
        import tracy_2d as test_module
        bc_type = case_name.split("_")[-1]  # 'specified' or 'no'
        if bc_type == "head":
            bc_type = "specified_head"
        elif bc_type == "flux":
            bc_type = "no_flux"
        else:
            # Handle tracy_2d_specified_head or tracy_2d_no_flux
            if "specified" in case_name:
                bc_type = "specified_head"
            elif "no" in case_name:
                bc_type = "no_flux"
            else:
                bc_type = "specified_head"
        results = test_module.model(level, bc_type=bc_type, do_write=args.write)
    elif case_name == "vauclin_2d":
        import vauclin_2d as test_module
        results = test_module.model(level, do_write=args.write)
    else:
        raise ValueError(f"Unknown case: {case_name}")

    # Write results to file
    if rank == 0:
        output_file = f"errors-{case_name}-levels{level}.dat"
        with open(output_file, "w") as f:
            f.write(" ".join(map(str, results)) + "\n")
        print(f"Results written to {output_file}")

    return 0


async def run_subcommand(args):
    """Execute test runs."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]

    for level in levels:
        print(f"Running {args.case} at level {level}...")
        await run_subproc(args, level, 1, {})

    return 0


def submit_subcommand(args):
    """Submit batch of tests."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]
    cores = case_config["cores"]

    for level in levels:
        for core_count in cores:
            cmd = args.template.format(cores=core_count)
            cmd += f" python3 {__file__} run {args.case}"
            if args.write:
                cmd += " --write"
            cmd += f" --level {level}"

            print(f"Submitting: {cmd}")
            if not args.dry_run:
                os.system(cmd)

    return 0


def count_subcommand(args):
    """Count number of test cases."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]
    cores = case_config["cores"]
    print(len(levels) * len(cores))
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Richards equation benchmark tests")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run a single test case")
    run_parser.add_argument("case", help="Test case name")
    run_parser.add_argument("--level", type=int, help="Refinement level")
    run_parser.add_argument("--write", action="store_true", help="Write VTK output")

    # Submit subcommand
    submit_parser = subparsers.add_parser("submit", help="Submit batch of tests")
    submit_parser.add_argument("case", help="Test case name")
    submit_parser.add_argument("-t", "--template", default="", help="Command template")
    submit_parser.add_argument("--write", action="store_true", help="Write VTK output")
    submit_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # Count subcommand
    count_parser = subparsers.add_parser("count", help="Count test cases")
    count_parser.add_argument("case", help="Test case name")

    args = parser.parse_args()

    if args.command == "run":
        if args.level is not None:
            # Run single level
            asyncio.run(run_subproc(args, args.level, 1, {}))
        else:
            # Run all levels
            asyncio.run(run_subcommand(args))
    elif args.command == "submit":
        submit_subcommand(args)
    elif args.command == "count":
        count_subcommand(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
