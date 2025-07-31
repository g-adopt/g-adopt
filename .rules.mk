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
# To configure the recipe, use per-target variables. In many cases, we
# can use simply-expanded variables which get a value directly at the
# time of assignment, using the colon-equals syntax: target : var :=
# value
#
# However, some variables need recursively-expanded variables. An
# example is when we refer to an automatic variable like $* (the stem
# of a pattern rule) for e.g. the current case name. We can't use
# simple expansion, because $* doesn't exist at definition time. In
# this case, we define the value with a bare equals sign:
# target : var = value
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

exec_cmd = $(if $(filter-out 1,$(ncpus)),mpiexec -np $(ncpus)) python3 $(notdir $<) $(PETSC_FLAGS) $(exec_args)

# Canned recipe
# -------------
#
# In *most* cases, this is the preferred method of executing tasks.
#
# Note that this is also responsible for changing to the directory
# containing the script for execution. If you're not using this
# recipe, this responsibility falls on you!
#
# Example:
#
#   target : var := value
#   target: script.py
#   	$(run-python)

define run-python =
@(cd $(dir $<); $(call run_cmd,$(exec_cmd),$(or $(desc),$(notdir $<))))
endef

# ----------------------------------------------------------------------

# ======================================
# Makefile rules for recursive inclusion
# ======================================
#
# See the root Makefile or demos/Makefile for a bit of context into
# how these rules are used. In short, these are some supporting rules
# for including subdirectories or defining targets.

# Directory stack manipulation
# ----------------------------
#
# These two defines can be used with $(eval $(header)) and $(eval
# $(trailer)) respectively, as part of the directory stack
# manipulation. Just remember that you need to use them both, at the
# start and end of a file!

define header =
sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)
endef

define trailer =
d := $(dirstack_$(sp))
sp := $(basename $(sp))
endef

# Recursive inclusion
# -------------------
#
# This is the key part of the system. Defining this rule allows us to
# use $(eval ...) to automatically recurse into subdirectories from a
# list (rather than manually listing them all).
#
# Example:
#
#   $(foreach dir,$(dirs),$(eval $(include_subdir)))

define include_subdir =
dir := $(dir)
include $$(dir)/Makefile
endef

# Define standalone subdirectory targets
# --------------------------------------
#
# This is for defining the targets within e.g. demos or tests. It
# should be called within the "standalone execution" branch, and
# requires that make_targets (below) has been called in the standard
# branch of execution. See demos/Makefile for the usage example.

define subdir_targets =
.PHONY: $(1) clean-$(1)
$(1):
	$$(MAKE) -C .. $(2)/$(1)

clean-$(1):
	$$(MAKE) -C .. clean-$(2)/$(1)
endef

# Define current directory targets
# --------------------------------
#
# This is the other half of the above rule. It defines the targets
# corresponding to sub-categories. See demos/Makefile for the usage
# example.

define make_targets =
.PHONY: $(1) clean-$(1)
$(1): $$(TGT_$(1))

clean-$(1):
	rm -f $$(CLEAN_$(1))
	rm -rf $$(DIR_CLEAN_$(1))
endef

# Define default targets for the current file
# -------------------------------------------
#
# As opposed to the two rules above, the following rules is for test
# cases or demos specifically. After variables like $(TGT_$(d)) and
# $(CLEAN_$(d)) have been defined, include this rule with $(eval
# $(default_targets)) to automatically define the .done and .clean
# targets, allowing for standalone running.

define default_targets =
$(if $(TGT_$(d)),$(warning There are no defined targets for $(d)!))
$(if $(CLEAN_$(d))$(DIR_CLEAN_$(d)),,$(warning There are no defined files to clean for $(d)!))

$(d)/.done: $(TGT_$(d))
	@touch $@

.PHONY: $(d)/.clean
$(d)/.clean:
	@rm -f $$(addprefix $(d)/,$$(CLEAN_$(d)))
	@rm -rf $$(addprefix $(d)/,$$(DIR_CLEAN_$(d)))
endef
