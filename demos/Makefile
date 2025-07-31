# Welcome to the Makefile for all the demos
#
# This Makefile is part of a "recursive inclusion system". At the top,
# we maintain a directory stack, so that the $(d) variable ends up
# where it should be once we've processed this file. To refer to the
# current directory of a given Makefile, this is also the variable we
# should be using!
sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

# These are the subdirectories for recursive processing. In general,
# we try not to pollute the global variable "namespace", so we suffix
# things by the current directory.
subdirs_$(d) := mantle_convection multi_material glacial_isostatic_adjustment PDE_constrained_optimisation

ifeq ($(d),)
# If we execute "make" within the demos/ directory, it would be nice
# to do something sensible. We call the main Makefile, and can run the
# test cases, check their correctness, or clean all artifacts.
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all:
	$(MAKE) -C .. $(cwd)

.PHONY: check
check:
	python3 -m pytest -k "not ../tests"

.PHONY: clean
clean:
	$(MAKE) -C .. clean-$(cwd)

# Because each of the subdirectories is actually a "category", we
# provide an additional target to run the tests specific to that
# subdirectory directly.
#
# The documentation gets more sparse as we go, but you can check out
# the mantle_convection/Makefile for an idea of how the category-level
# Makefiles work.
include ../.rules.mk
$(foreach dir,$(subdirs_$(d)),$(eval $(call subdir_targets,$(dir),$(cwd))))

else
# Otherwise, we were included from the Makefile at the level above us,
# and should continue to build out the tree.
$(foreach dir,$(addprefix $(d)/,$(subdirs_$(d))),$(eval $(include_subdir)))

# For the demos specifically, we have a second purpose: to convert the
# demos to jupyter notebooks that serve as tutorials. This target
# performs the conversion, then builds an archive containing the
# notebooks and some supplementary files (a page index and flowchart,
# and any external images that were generated and recorded in
# $(NOTEBOOK_IMGS)).
notebook_files := $(NOTEBOOK_SRC:.py=.ipynb)
$(d)/artifact.tar: $(notebook_files) $(d)/.pages $(d)/.diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file $@ $(filter-out %.ipynb,$^)
	tar --transform='s|/.*/|/|' --append --file $@ $(notebook_files)
	$(foreach image,$(NOTEBOOK_IMGS),tar --append --file $@ $(image) ;)

# Here is the pattern rule for converting a notebook from its linked
# jupytext source file. The $< automatic variable is the name of the
# first prerequisite (i.e. the source file), and we also make sure to
# change to the directory containing the source file so path
# references, etc. make sense from the viewpoint of the script.
%.ipynb: %.py
	jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))

# Finally, we create the standalone build and clean targets for this
# directory.
TGT_$(d) := $(filter $(d)/%,$(TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

$(eval $(call make_targets,$(d)))
endif

# This is the bookkeeping for a stack that returns $(d) to where it
# was, so it continues to work after including sub-Makefiles.
d := $(dirstack_$(sp))
sp := $(basename $(sp))
