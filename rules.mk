# ==================================
# Makefile rules for executing tests
# ==================================
#
# To support running tests as a full suite (`make test` from project root),
# we have a `run-python` canned recipe to include in demo and test Makefiles.
# This handles "batch mode", where we want to run everything at once, with
# some kind of scheduling (through task-spooler), or "regular mode", which is
# used for ad-hoc or targeted testing, where we don't want to have extra
# utilities installed. In some very isolated cases, we may need to call the
# underlying macros directly.
#
# Variables
# ---------
#
# To configure the recipe, use per-target variables:
# target : var := value (or the deferred variant)
#
# As a matter of style, it makes it more distinct from a regular rule
# definition to include a space between the target and the first colon.
#
# Python script: This should be the *first dependency*, available in $<
# `category`:    Used for tsp categorisation, and gives more context
#                for timing information.
# `ncpus`:       If not specified, defaults to 1 and runs the target
#                *without* MPI. Any value greater than 1 will use mpiexec.
# `PETSC_FLAGS`: Could come from the environment, but can also be specified
#                per-target. Note that target-specific overrides will win
#                over the environment! These are flags for the PETSc options
#                database, such as `-log_view`.
# `exec_args`:   Per-target arguments to pass through to the Python script.
# `BATCH_MODE`:  This comes from the environment, and should be defined
#                e.g. in a CI environment where job scheduling needs to be
#                more granular than "run everything at once".
#                Note that there is a *2 hour timeout* on jobs in batch mode.
# `desc`:        A description of the job, if the script name is
#                insufficient. Could be used when invoking the same script
#                with several argument combinations.


# Run macros
# ----------
#
# These are the underlying macros for "regular" or "batch" mode
# and handle scheduling and printing timing information.
# Use `run_cmd` to respect the `BATCH_MODE` environment variable,
# otherwise use `run_regular` or `run_batch` directly. These should
# be used sparingly!
#
# These macros are designed to be used in a `$(call ...)` construct,
# where `$(1)` is the command to execute (usually through `exec_wrapper`),
# and `$(2)` is the description of the job to print in the timing summary.
# `ncpus` and `category` are described above.

run_regular = d=$$(date +%s); $(1) && echo "$(2) took $$(($$(date +%s)-d)) seconds"
run_batch = id=$$(tsp -f $(if $(filter-out 1,$(ncpus)),-N $(ncpus)) $(if $(category),-L $(category)) $(1)) && echo "[$(category)]$(2) took $$(tsp --print-total-time $$id)"
run_cmd = $(if $(BATCH_MODE),$(run_batch),$(run_regular))

# Execution macro
# ---------------
#
# This determines how to run the underlying script, and handles dispatch
# to mpiexec, and additional command line arguments.
# Should not be called directly.

exec_cmd = $(if $(filter-out 1,$(ncpus)),mpiexec -np $(ncpus)) python3 $< $(PETSC_FLAGS) $(exec_args)

# Execution wrapper
# -----------------
#
# When running in batch mode, we want an additional wrap inside the
# `timeout` utility to enforce a 2 hour time limit on any given task.
# This ensures that CI jobs don't exceed the 6 hour time limit on
# GitHub Actions so they can be torn down gracefully and provide
# output information.

exec_wrapper = $(if $(BATCH_MODE),timeout 2h $(exec_cmd),$(exec_cmd))

# Canned recipe
# -------------
#
# In *most* cases, this is the preferred method of executing tasks.
# For example:
#
#   target : var := value
#   target: script.py
#   	$(run-python)

define run-python =
@$(call run_cmd,$(exec_wrapper),$(or $(desc),$<))
endef

# Test-related variables
# ----------------------
#
# These are expansion rules that can be used to call the generic test_all.py
# with reference to the current test/demo.

current_dir = $(notdir $(patsubst %/,%,$(CURDIR)))
test_class = $(notdir $(patsubst %/,%,$(dir $(CURDIR))))
