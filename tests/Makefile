sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

subdirs := analytical_comparisons multi_material optimisation_checkpointing scalar_advection scalar_advection_diffusion Drucker-Prager_rheology adjoint 2d_cylindrical_TALA_DG viscoplastic_case_DG free_surface viscoelastic base_gmsh parallel_scaling

ifeq ($(d),)
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all:
	$(MAKE) -C .. $(cwd)

.PHONY: check
check:
	python3 -m pytest ../demos -m "not demo and not longtest"

.PHONY: longcheck
longcheck:
	python3 -m pytest -m "longtest"

.PHONY: clean
clean:
	$(MAKE) -C .. clean-$(cwd)

include ../.rules.mk
$(foreach dir,$(subdirs),$(eval $(call subdir_targets,$(dir),$(cwd))))

else

$(foreach dir,$(addprefix $(d)/,$(subdirs)),$(eval $(include_subdir)))

TGT_$(d) := $(filter $(d)/%,$(TGT))
HPC_TGT_$(d) := $(filter $(d)/%,$(HPC_TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

$(eval $(call make_targets,$(d)))

# define hpc tests, if there are any
ifneq ($(HPC_TGT_$(d)),)
hpc-tests: $(HPC_TGT_$(d))
endif

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
