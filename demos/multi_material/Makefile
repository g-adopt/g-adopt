sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all:
	$(MAKE) -C .. $(cwd)

.PHONY: check
check: all
	pytest ../test_all.py . -k $(cwd)
else

subdirs_$(d) := compositional_buoyancy thermochemical_buoyancy

$(foreach dir,$(addprefix $(d)/,$(subdirs_$(d))),$(eval $(include_subdir)))

TGT_$(d) := $(filter $(d)/%,$(TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
