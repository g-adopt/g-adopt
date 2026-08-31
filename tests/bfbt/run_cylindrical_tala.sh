#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd "${script_dir}/../.." && pwd)
python_bin=${FIREDRAKE_PYTHON:-/Users/rhodrid/Firedrake/venv-firedrake/bin/python}
mpi_exec=${MPIEXEC:-/opt/homebrew/bin/mpiexec}
mpi_ranks=${BFBT_MPI_RANKS:-2}
ncells=${BFBT_NCELLS:-32}
nlayers=${BFBT_NLAYERS:-8}
warm_repeats=${BFBT_WARM_REPEATS:-3}
suite=${BFBT_SUITE:-baseline}
density_contrast=${BFBT_DENSITY_CONTRAST:-1.65}
density_space=${BFBT_DENSITY_SPACE:-cg}
default_viscosity_contrast=10
if [[ ${suite} == tuned ]]; then
    default_viscosity_contrast=100
fi
radial_viscosity_contrast=${BFBT_RADIAL_VISCOSITY_CONTRAST:-${default_viscosity_contrast}}
lateral_viscosity_contrast=${BFBT_LATERAL_VISCOSITY_CONTRAST:-${default_viscosity_contrast}}
pressure_rtol=${BFBT_PRESSURE_RTOL:-1e-8}
velocity_rtol=${BFBT_VELOCITY_RTOL:-1e-10}
max_relative_residual=${BFBT_MAX_RELATIVE_RESIDUAL:-1e-7}
timestamp=$(date +%Y%m%dT%H%M%S)
results_dir=${BFBT_RESULTS_DIR:-${TMPDIR:-/tmp}/gadopt-bfbt-results/cylindrical_tala_${timestamp}}

if [[ ${suite} == all ]] && (( ncells < 16 || nlayers < 4 )); then
    printf '%s\n' "BFBT_SUITE=all requires at least 16 cells and 4 layers" >&2
    exit 2
fi
if [[ ${suite} == tuned ]] && (( ncells < 32 || nlayers < 8 )); then
    printf '%s\n' "BFBT_SUITE=tuned requires at least 32 cells and 8 layers" >&2
    exit 2
fi
if (( mpi_ranks < 2 )); then
    printf '%s\n' "BFBT_MPI_RANKS must be at least 2" >&2
    exit 2
fi

export PETSC_DIR=${PETSC_DIR:-/Users/rhodrid/Firedrake/petsc}
export PETSC_ARCH=${PETSC_ARCH:-arch-firedrake-default}
export HDF5_MPI=${HDF5_MPI:-ON}
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${results_dir}"
shopt -s nullglob
existing_results=(
    "${results_dir}"/*.json
    "${results_dir}"/*.log
    "${results_dir}"/*.tsv
)
shopt -u nullglob
if (( ${#existing_results[@]} > 0 )); then
    printf '%s\n' "results directory is not empty: ${results_dir}" >&2
    exit 2
fi

run_case() {
    local label=$1
    local ranks=$2
    local pressure_pc=$3
    shift 3
    local output_json="${results_dir}/${label}.json"
    local output_log="${results_dir}/${label}.log"
    local command=(
        "${python_bin}"
        "${script_dir}/cylindrical_tala.py"
        --pc "${pressure_pc}"
        --ncells "${ncells}"
        --nlayers "${nlayers}"
        --warm-repeats "${warm_repeats}"
        --density-contrast "${density_contrast}"
        --density-space "${density_space}"
        --radial-viscosity-contrast "${radial_viscosity_contrast}"
        --lateral-viscosity-contrast "${lateral_viscosity_contrast}"
        --pressure-rtol "${pressure_rtol}"
        --velocity-rtol "${velocity_rtol}"
        --json-file "${output_json}"
        "$@"
    )
    local status=0
    if (( ranks > 1 )); then
        "${mpi_exec}" -n "${ranks}" "${command[@]}" >"${output_log}" 2>&1 || status=$?
    else
        "${command[@]}" >"${output_log}" 2>&1 || status=$?
    fi
    if (( status != 0 )); then
        printf '%s\n' "failed ${label} (exit ${status}): ${output_log}" >&2
        return 1
    fi
    if ! "${python_bin}" -c '
import json
import math
import sys
result = json.load(open(sys.argv[1]))
expected_ranks = int(sys.argv[2])
expected_pc = sys.argv[3]
max_relative_residual = float(sys.argv[4])
assert result["mpi_size"] == expected_ranks
assert result["pc"] == expected_pc
assert result["velocity_ksp_reason"] > 0
assert result["pressure_ksp_reason"] > 0
assert math.isfinite(result["relative_equation_residual"])
assert result["relative_equation_residual"] <= max_relative_residual
assert all(sample["velocity_failures"] == 0 for sample in result["warm_work_samples"])
assert all(sample["pressure_failures"] == 0 for sample in result["warm_work_samples"])
assert all(sample["pressure_pc_inner_failures"] == 0 for sample in result["warm_work_samples"])
assert result["velocity_pc"]["pc_type"] == "gamg"
if expected_pc == "bfbt":
    assert all(sample["bfbt_inner_failures"] == 0 for sample in result["warm_work_samples"])
    assert result["bfbt_operators"]["right_nullspace_is_exact"]
    assert result["bfbt_operators"]["left_nullspace_is_exact"]
    assert result["bfbt_operators"]["exact_left_nullspace_attached"]
    assert result["pressure_pc_inner_ksp_type"] == result["bfbt_inner_ksp"]
    if result["bfbt_inner_ksp"] == "richardson":
        assert result["pressure_pc_inner_ksp_norm_type"] == "none"
        assert not result["pressure_pc_inner_ksp_residual_measured"]
        assert result["pressure_pc_inner_ksp_residual"] is None
        assert result["pressure_pc_inner_last_convergence_history"] is None
        assert result["pressure_pc_inner_ksp_max_it"] == result["bfbt_inner_max_it"]
        assert all(
            iteration == result["bfbt_inner_max_it"]
            for sample in result["warm_work_samples"]
            for iteration in sample["pressure_pc_inner_iterations_by_solve"]
        )
' "${output_json}" "${ranks}" "${pressure_pc}" "${max_relative_residual}"; then
        printf '%s\n' "invalid ${label}: ${output_json}" >&2
        return 1
    fi
    printf '%s\n' "completed ${label}: ${output_json}"
}

run_expected_nullspace_rejection() {
    local label=$1
    local ranks=$2
    local output_log="${results_dir}/${label}.log"
    local output_json="${results_dir}/${label}.json"
    rm -f "${output_json}"
    local command=(
        "${python_bin}"
        "${script_dir}/cylindrical_tala.py"
        --pc bfbt
        --ncells "${ncells}"
        --nlayers "${nlayers}"
        --warm-repeats 1
        --density-contrast "${density_contrast}"
        --density-space dq
        --radial-viscosity-contrast "${radial_viscosity_contrast}"
        --lateral-viscosity-contrast "${lateral_viscosity_contrast}"
        --pressure-rtol "${pressure_rtol}"
        --velocity-rtol "${velocity_rtol}"
        --json-file "${output_json}"
    )
    local status=0
    if (( ranks > 1 )); then
        "${mpi_exec}" -n "${ranks}" "${command[@]}" >"${output_log}" 2>&1 \
            || status=$?
    else
        "${command[@]}" >"${output_log}" 2>&1 || status=$?
    fi
    if (( status == 0 )); then
        printf '%s\n' "unexpected success ${label}: ${output_log}" >&2
        return 1
    fi
    if ! grep -a -q "no compatible transpose nullspace" "${output_log}"; then
        printf '%s\n' "wrong failure ${label} (exit ${status}): ${output_log}" >&2
        return 1
    fi
    printf '%s\n' "completed expected rejection ${label}: ${output_log}"
}

run_test() {
    local label=$1
    shift
    local output_log="${results_dir}/${label}.log"
    if "$@" >"${output_log}" 2>&1; then
        printf '%s\n' "completed ${label}: ${output_log}"
    else
        local status=$?
        printf '%s\n' "failed ${label} (exit ${status}): ${output_log}" >&2
        return 1
    fi
}

cd "${repo_dir}"
failures=0
case "${suite}" in
    baseline)
        run_case serial_mass 1 mass || failures=$((failures + 1))
        run_case serial_bfbt_gamg 1 bfbt || failures=$((failures + 1))
        run_case mpi_mass "${mpi_ranks}" mass || failures=$((failures + 1))
        run_case mpi_bfbt_gamg "${mpi_ranks}" bfbt || failures=$((failures + 1))
        ;;
    tuned)
        tuned_bfbt=(
            --bfbt-inner-ksp fgmres
            --bfbt-inner-rtol 1e-4
            --bfbt-inner-max-it 1000
            --bfbt-gamg-aggressive-coarsening 0
            --bfbt-gamg-agg-nsmooths 2
        )
        for ranks_label in serial mpi; do
            ranks=1
            if [[ ${ranks_label} == mpi ]]; then
                ranks=${mpi_ranks}
            fi
            run_case "${ranks_label}_mass_lu" "${ranks}" mass \
                --mass-inner-ksp preonly --mass-inner-pc lu \
                || failures=$((failures + 1))
            run_case "${ranks_label}_bfbt_dg0" "${ranks}" bfbt \
                "${tuned_bfbt[@]}" --bfbt-weight-degree 0 \
                || failures=$((failures + 1))
            run_case "${ranks_label}_bfbt_dg1" "${ranks}" bfbt \
                "${tuned_bfbt[@]}" --bfbt-weight-degree 1 \
                || failures=$((failures + 1))
            run_case "${ranks_label}_bfbt_dg1_richardson4" "${ranks}" bfbt \
                "${tuned_bfbt[@]}" --bfbt-weight-degree 1 \
                --bfbt-inner-ksp richardson --bfbt-inner-max-it 4 \
                || failures=$((failures + 1))
        done
        ;;
    all)
        run_test unit_serial "${python_bin}" -m pytest -q \
            tests/unit/test_bfbt_preconditioner.py || failures=$((failures + 1))
        run_test unit_mpi "${mpi_exec}" -n "${mpi_ranks}" \
            "${python_bin}" -m pytest -q tests/unit/test_bfbt_preconditioner.py \
            || failures=$((failures + 1))

        matched=(--pressure-rtol 1e-8 --velocity-rtol 1e-10)
        run_case serial_mass_tight 1 mass "${matched[@]}" \
            || failures=$((failures + 1))
        run_case serial_bfbt_lu_preonly 1 bfbt "${matched[@]}" \
            --bfbt-inner-ksp preonly --bfbt-inner-pc lu \
            || failures=$((failures + 1))
        run_case serial_bfbt_lu_tight 1 bfbt "${matched[@]}" \
            --bfbt-inner-pc lu --bfbt-inner-rtol 1e-10 \
            || failures=$((failures + 1))
        run_case serial_bfbt_gamg_loose 1 bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-2 || failures=$((failures + 1))
        run_case serial_bfbt_gamg_tight 1 bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            || failures=$((failures + 1))

        run_case serial_bfbt_gamg_agg0 1 bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            --bfbt-gamg-aggressive-coarsening 0 \
            || failures=$((failures + 1))
        run_case serial_bfbt_gamg_agg0_nsmooths2 1 bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            --bfbt-gamg-aggressive-coarsening 0 \
            --bfbt-gamg-agg-nsmooths 2 \
            || failures=$((failures + 1))
        run_case serial_bfbt_gamg_square_off 1 bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            --bfbt-gamg-aggressive-square-graph off \
            || failures=$((failures + 1))
        for nsmooths in 0 2; do
            run_case "serial_bfbt_gamg_nsmooths${nsmooths}" 1 bfbt \
                "${matched[@]}" --bfbt-inner-rtol 1e-10 \
                --bfbt-inner-max-it 500 \
                --bfbt-gamg-agg-nsmooths "${nsmooths}" \
                || failures=$((failures + 1))
        done
        for threshold in 0.01 0.05; do
            run_case "serial_bfbt_gamg_threshold${threshold}" 1 bfbt \
                "${matched[@]}" --bfbt-inner-rtol 1e-10 \
                --bfbt-inner-max-it 500 --bfbt-gamg-threshold "${threshold}" \
                || failures=$((failures + 1))
        done

        for contrast in 1 100; do
            run_case "serial_mass_jacobi_mu${contrast}" 1 mass "${matched[@]}" \
                --mass-inner-ksp preonly --mass-inner-pc jacobi \
                --radial-viscosity-contrast "${contrast}" \
                --lateral-viscosity-contrast "${contrast}" \
                || failures=$((failures + 1))
            run_case "serial_mass_lu_mu${contrast}" 1 mass "${matched[@]}" \
                --mass-inner-ksp preonly --mass-inner-pc lu \
                --radial-viscosity-contrast "${contrast}" \
                --lateral-viscosity-contrast "${contrast}" \
                || failures=$((failures + 1))
            run_case "serial_bfbt_mu${contrast}" 1 bfbt "${matched[@]}" \
                --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
                --radial-viscosity-contrast "${contrast}" \
                --lateral-viscosity-contrast "${contrast}" \
                || failures=$((failures + 1))
        done
        for inner_rtol in 1e-2 1e-4 1e-6 1e-8; do
            rtol_label=${inner_rtol//-/m}
            run_case "serial_bfbt_mu100_rtol_${rtol_label}" 1 bfbt \
                "${matched[@]}" --bfbt-inner-rtol "${inner_rtol}" \
                --bfbt-inner-max-it 1000 --bfbt-gamg-aggressive-coarsening 0 \
                --bfbt-gamg-agg-nsmooths 2 \
                --radial-viscosity-contrast 100 \
                --lateral-viscosity-contrast 100 \
                || failures=$((failures + 1))
        done

        run_case mpi_mass_tight "${mpi_ranks}" mass "${matched[@]}" \
            || failures=$((failures + 1))
        run_case mpi_bfbt_gamg_tight "${mpi_ranks}" bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            || failures=$((failures + 1))
        run_case mpi_bfbt_gamg_agg0 "${mpi_ranks}" bfbt "${matched[@]}" \
            --bfbt-inner-rtol 1e-10 --bfbt-inner-max-it 500 \
            --bfbt-gamg-aggressive-coarsening 0 \
            || failures=$((failures + 1))
        run_case mpi_bfbt_gamg_agg0_nsmooths2 "${mpi_ranks}" bfbt \
            "${matched[@]}" --bfbt-inner-rtol 1e-10 \
            --bfbt-inner-max-it 500 --bfbt-gamg-aggressive-coarsening 0 \
            --bfbt-gamg-agg-nsmooths 2 \
            || failures=$((failures + 1))
        run_case mpi_bfbt_mu100_rtol_1em4 "${mpi_ranks}" bfbt \
            "${matched[@]}" --bfbt-inner-rtol 1e-4 \
            --bfbt-inner-max-it 1000 --bfbt-gamg-aggressive-coarsening 0 \
            --bfbt-gamg-agg-nsmooths 2 \
            --radial-viscosity-contrast 100 \
            --lateral-viscosity-contrast 100 \
            || failures=$((failures + 1))
        run_expected_nullspace_rejection serial_dq_rejection 1 \
            || failures=$((failures + 1))
        run_expected_nullspace_rejection mpi_dq_rejection "${mpi_ranks}" \
            || failures=$((failures + 1))
        ;;
    *)
        printf '%s\n' "BFBT_SUITE must be baseline, tuned, or all" >&2
        exit 2
        ;;
esac

"${python_bin}" - "${results_dir}" "${suite}" "${ncells}" "${nlayers}" \
    "${warm_repeats}" "${mpi_ranks}" "${max_relative_residual}" \
    "${pressure_rtol}" "${velocity_rtol}" "${failures}" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
manifest = {
    "suite": sys.argv[2],
    "ncells": int(sys.argv[3]),
    "nlayers": int(sys.argv[4]),
    "warm_repeats": int(sys.argv[5]),
    "mpi_ranks": int(sys.argv[6]),
    "max_relative_residual": float(sys.argv[7]),
    "pressure_rtol": float(sys.argv[8]),
    "velocity_rtol": float(sys.argv[9]),
    "failures": int(sys.argv[10]),
}
manifest["status"] = "passed" if manifest["failures"] == 0 else "failed"
(results_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
header = (
    "case\tmpi_size\tpc\tviscosity_maximum\tpressure_iterations\t"
    "velocity_iterations\tinner_iterations\trelative_equation_residual\t"
    "warm_seconds\n"
)
rows = []
for path in sorted(results_dir.glob("*.json")):
    if path.name == "manifest.json":
        continue
    result = json.loads(path.read_text())
    work = result["warm_work_samples"][0]
    rows.append(
        "\t".join(
            str(value)
            for value in (
                path.stem,
                result["mpi_size"],
                result["pc"],
                result["viscosity_maximum"],
                work["pressure_iterations"],
                work["velocity_iterations"],
                work["pressure_pc_inner_iterations"],
                result["relative_equation_residual"],
                result["warm_seconds"],
            )
        )
        + "\n"
    )
(results_dir / "summary.tsv").write_text(header + "".join(rows))
PY

printf '%s\n' "results: ${results_dir}"
if (( failures > 0 )); then
    printf '%s\n' "${failures} configuration(s) failed" >&2
    exit 1
fi
