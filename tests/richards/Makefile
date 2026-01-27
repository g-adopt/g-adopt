include ../../rules.mk

# Tracy 2D test cases for CI (convergence tests with analytical solution)
tracy_cases := tracy_2d_specified_head_dg1 tracy_2d_specified_head_dg2 tracy_2d_no_flux_dg1 tracy_2d_no_flux_dg2

# Default CI target runs Tracy cases only (have analytical convergence tests)
cases := $(tracy_cases)

# All cases including Vauclin (not run in CI by default)
all_cases := $(tracy_cases) vauclin_2d

.PHONY: all tracy $(all_cases) clean check list thresholds

# CI target: run Tracy cases and pytest checks
all: $(cases) check

# Run all Tracy cases
tracy: $(tracy_cases)

# Run all cases including Vauclin
full: $(all_cases)

# Run all levels for a case sequentially
$(all_cases): richards.py tracy_2d.py vauclin_2d.py
	@$(call run_regular,python3 richards.py run $@,$@)

# Individual level targets for parallel execution via task spooler
# Usage: make submit_tracy_2d_specified_head_dg1
submit_%: richards.py
	@python3 richards.py submit $* -t "tsp -N {cores} -f"

# Dry run to see what would be submitted
dryrun_%: richards.py
	@python3 richards.py submit $* -t "tsp -N {cores} -f" --dry-run

# List available cases
list:
	@python3 richards.py list

clean:
	rm -rf *.csv *.pvd tracy_h_* tracy_analytical_* __pycache__ .sentinel.*

# Run pytest checks
check:
	python -m pytest test_richards.py -v -s

# Print baseline thresholds (run after baseline runs complete)
thresholds:
	python test_richards.py
