# Makefile stack management, refer to demos/Makefile for an idea of
# what this does and why it's needed. Or, if you're interested in how
# a specific case is implemented, check out base_case/Makefile.
sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)
# If we execute "make" within this directory, we provide targets to
# build and check all the contained cases.
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all:
	$(MAKE) -C .. $(cwd)

.PHONY: check
check: all
	pytest ../test_all.py . -k $(cwd)

else

# Otherwise, we're in the recursive inclusion process, and need to
# include all the cases we're responsible for.
subdirs_$(d) := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint gplates_global free_surface dynamic_topography

$(foreach dir,$(addprefix $(d)/,$(subdirs_$(d))),$(eval $(include_subdir)))

# These category-specific collections of targets are used at the level above.
TGT_$(d) := $(filter $(d)/%,$(TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
