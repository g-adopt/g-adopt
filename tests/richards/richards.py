import argparse
import csv
import os
import sys

# Configuration for Richards equation test cases
# Core counts are mapped by level: level 2->1, level 3->2, level 4->4
cases = {
    "tracy_2d_specified_head_dg1": {
        "levels": [1, 2, 3, 4],
        "degree": 1,
        "bc_type": "specified_head",
        "space_type": "DG",
    },
    "tracy_2d_specified_head_dg2": {
        "levels": [1, 2, 3, 4],
        "degree": 2,
        "bc_type": "specified_head",
        "space_type": "DG",
    },
    "tracy_2d_no_flux_dg1": {
        "levels": [1, 2, 3, 4],
        "degree": 1,
        "bc_type": "no_flux",
        "space_type": "DG",
    },
    "tracy_2d_no_flux_dg2": {
        "levels": [1, 2, 3, 4],
        "degree": 2,
        "bc_type": "no_flux",
        "space_type": "DG",
    },
    "vauclin_2d": {
        "levels": [1, 2],
        "degree": 1,
        "bc_type": None,
        "space_type": "DG",
    },
}

# Map level to recommended core count (for DG1)
# DOFs quadruple each level, so cores should too to keep unknowns/core constant
LEVEL_TO_CORES_DG1 = {
    1: 1,
    2: 1,
    3: 4,
    4: 16,
}

# DG2 has 2x the DOFs of DG1, so use 2x the cores
LEVEL_TO_CORES_DG2 = {
    1: 2,
    2: 2,
    3: 8,
    4: 32,
}


def get_case(cases_dict, config):
    """Get case configuration by name."""
    return cases_dict[config]


def get_cores_for_level(level, degree=1):
    """Get recommended core count for a refinement level and polynomial degree."""
    if degree == 2:
        return LEVEL_TO_CORES_DG2.get(level, 1)
    return LEVEL_TO_CORES_DG1.get(level, 1)


def run_tracy_case(level, bc_type, degree, space_type='DG', do_write=False):
    """Run a Tracy 2D test case and return results."""
    import tracy_2d
    return tracy_2d.model(level, bc_type=bc_type, degree=degree, do_write=do_write, space_type=space_type)


def run_vauclin_case(level, do_write=False):
    """Run a Vauclin 2D test case and return results."""
    import vauclin_2d
    return vauclin_2d.model(level, do_write=do_write)


def write_results_csv(output_file, results, case_name, level, degree):
    """Write results to CSV file."""
    file_exists = os.path.exists(output_file)

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            if 'l2error_h' in results:
                # Tracy case
                writer.writerow([
                    'case', 'level', 'degree',
                    'l2error_h', 'l2error_theta', 'l2anal_h', 'l2anal_theta',
                    'max_intermediate_error_h', 'final_time', 'converged'
                ])
            else:
                # Vauclin case
                writer.writerow([
                    'case', 'level', 'degree',
                    'min_h', 'max_h', 'total_infiltration'
                ])

        # Write data row
        if 'l2error_h' in results:
            writer.writerow([
                case_name, level, degree,
                results['l2error_h'], results['l2error_theta'],
                results['l2anal_h'], results['l2anal_theta'],
                results['max_intermediate_error_h'], results['final_time'],
                results.get('converged', True)
            ])
        else:
            writer.writerow([
                case_name, level, degree,
                results['min_h'], results['max_h'], results['total_infiltration']
            ])


def run_single_case(args):
    """Run a single test case at a specific level."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    case_config = get_case(cases, args.case)
    level = args.level
    degree = case_config["degree"]
    bc_type = case_config["bc_type"]
    space_type = case_config.get("space_type", "DG")

    if rank == 0:
        print(f"Running {args.case} at level {level} with {space_type}{degree}...")

    # Run appropriate case
    if args.case.startswith("tracy_2d"):
        results = run_tracy_case(level, bc_type, degree, space_type=space_type, do_write=args.write)
    elif args.case == "vauclin_2d":
        results = run_vauclin_case(level, do_write=args.write)
    else:
        raise ValueError(f"Unknown case: {args.case}")

    # Write results
    if rank == 0:
        output_file = f"results_{args.case}.csv"
        write_results_csv(output_file, results, args.case, level, degree)
        print(f"Results appended to {output_file}")

        # Also print summary
        if 'l2error_h' in results:
            rel_error = results['l2error_h'] / results['l2anal_h']
            print(f"  Relative L2 error (h): {rel_error:.6e}")
            print(f"  Max intermediate error (h): {results['max_intermediate_error_h']:.6e}")

    return 0


def run_all_levels(args):
    """Run all levels for a case."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]

    # Clear existing results file
    output_file = f"results_{args.case}.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    for level in levels:
        args.level = level
        run_single_case(args)

    return 0


def submit_subcommand(args):
    """Submit batch of tests using task spooler."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]
    degree = case_config["degree"]

    for level in levels:
        cores = get_cores_for_level(level, degree)

        # Build command
        if args.template:
            cmd = args.template.format(cores=cores)
        else:
            cmd = ""

        if cores > 1:
            cmd += f" mpiexec -np {cores}"

        cmd += f" python3 {__file__} run {args.case} --level {level}"
        if args.write:
            cmd += " --write"

        cmd = cmd.strip()
        print(f"Submitting: {cmd}")
        if not args.dry_run:
            os.system(cmd)

    return 0


def count_subcommand(args):
    """Count number of test cases."""
    case_config = get_case(cases, args.case)
    levels = case_config["levels"]
    print(len(levels))
    return 0


def list_cases(args):
    """List all available test cases."""
    print("Available test cases:")
    for name, config in cases.items():
        levels = config["levels"]
        degree = config["degree"]
        space_type = config.get("space_type", "DG")
        bc = config.get("bc_type", "N/A")
        print(f"  {name}: levels={levels}, {space_type}{degree}, bc_type={bc}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Richards equation benchmark tests")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run a test case")
    run_parser.add_argument("case", help="Test case name")
    run_parser.add_argument("--level", type=int, help="Refinement level (run single level)")
    run_parser.add_argument("--write", action="store_true", help="Write VTK output")

    # Submit subcommand
    submit_parser = subparsers.add_parser("submit", help="Submit batch of tests")
    submit_parser.add_argument("case", help="Test case name")
    submit_parser.add_argument("-t", "--template", default="", help="Command template (e.g., 'tsp -N {cores} -f')")
    submit_parser.add_argument("--write", action="store_true", help="Write VTK output")
    submit_parser.add_argument("--dry-run", action="store_true", help="Dry run (print commands only)")

    # Count subcommand
    count_parser = subparsers.add_parser("count", help="Count test cases")
    count_parser.add_argument("case", help="Test case name")

    # List subcommand
    subparsers.add_parser("list", help="List available test cases")

    args = parser.parse_args()

    if args.command == "run":
        if args.level is not None:
            return run_single_case(args)
        else:
            return run_all_levels(args)
    elif args.command == "submit":
        return submit_subcommand(args)
    elif args.command == "count":
        return count_subcommand(args)
    elif args.command == "list":
        return list_cases(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
