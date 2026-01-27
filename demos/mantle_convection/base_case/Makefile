# Makefile stack management, refer to demos/Makefile for an idea of
# what this does and why it's needed.
$(eval $(header))

ifeq ($(d),)
# If we're working on a single case, it's probably a bit easier to be
# running a tighter development loop within its directory
# specifically. We define targets specific to this directory, so we
# can run "make check" to ensure the tests are still passing!

# We define some "standard" rules for demos, which allows us to run
# "make" and "make clean". However, because the actual command to run
# for "make check" changes depending on the actual case, we have to
# define that on a case-by-case basis!
include ../../.rules.mk
$(eval $(standalone))

.PHONY: check
check: .done
	python3 -m pytest ../../test_all.py -k mantle_convection/base_case

else

# This is the general (but not completely rigid) structure for
# defining the inputs and outputs of a case.

# TGT_$(d) should define the actual files that we require to be
# generated out of the test. Usually this is the actual results file
# that we're performing some kind of test on. There may be auxiliary
# outputs (checkpoints, visualisation), but unless they're required
# specifically, they don't need to go here.
TGT_$(d) := $(d)/params.log

# SRC_$(d) is just to make things cleaner: we can define any source
# files on which the case depends. Make will ensure that if these
# dependencies are changed (i.e. their modification time is *newer*
# than the latest result), the case will be run again. Nothing here is
# strictly necessary, but it can lead to a nicer development workflow
# if the dependencies are captured correctly.
SRC_$(d) := $(d)/base_case.py

# For the demos specifically, we want to record which source file
# contains the jupytext-compatible source for conversion to a
# notebook.
NOTEBOOK_SRC_$(d) := $(d)/base_case.py

# To give ourselves the best chance of reproducibility and
# cleanliness, we can define file- and directory-based artifacts that
# are produced from running the case. Because there can be a lot of
# entries here, these are written relative to the current directory
# (i.e. without the $(d) prefix). It's important that .done and all
# the files listed in $(TGT_$(d)) end up here.
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 params.log .done
# As a small insulation from foot-shooting, the following directories
# are removed with "rm -rf", whereas the above files are only removed
# with "rm -f".
DIR_CLEAN_$(d) := output

# Now we create the actual targets for running the test. The following
# rule defines the ".done" and ".clean" targets, which are used for
# standalone running.
$(eval $(default_targets))

# This is where a lot of the magic happens. The "run-python" canned
# recipe (defined in .rules.mk) is controlled by a few magic
# variables, and knows how to run python on a case for most
# situations, except where we require a submission manager.
$(d)/params.log : category := mantle_convection
$(d)/params.log: $(SRC_$(d))
	$(run-python)


# Once we are sure that all our variables are correctly populated, we
# can update the global target lists so that the top-level knows that
# this case exists.
TGT := $(TGT) $(TGT_$(d))
NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

$(eval $(trailer))
