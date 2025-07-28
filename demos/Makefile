sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

subdirs := mantle_convection multi_material glacial_isostatic_adjustment PDE_constrained_optimisation

ifeq ($(d),)
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

# all direct-execution targets in this directory
.PHONY: all
all:
	$(MAKE) -C .. $(cwd)

.PHONY: check
check:
	python3 -m pytest -k "not ../tests"

.PHONY: clean
clean:
	$(MAKE) -C .. clean-$(cwd)

# create a target for each of the subdirs
include ../.rules.mk
$(foreach dir,$(subdirs),$(eval $(call subdir_targets,$(dir),$(cwd))))

else

$(foreach dir,$(addprefix $(d)/,$(subdirs)),$(eval $(include_subdir)))

notebook_files := $(NOTEBOOK_SRC:.py=.ipynb)
$(d)/artifact.tar: $(notebook_files) $(d)/.pages $(d)/.diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file $@ $(filter-out %.ipynb,$^)
	tar --transform='s|/.*/|/|' --append --file $@ $(notebook_files)
	$(foreach image,$(NOTEBOOK_IMGS),tar --append --file $@ $(image) ;)

TGT_$(d) := $(filter $(d)/%,$(TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

$(eval $(call make_targets,$(d)))
endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
