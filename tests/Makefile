all: targets

include ../.rules.mk

define include_subdir =
dir := $(dir)
include $$(dir)/Makefile
endef

define subdir_targets =
.PHONY $(1) clean-$(1)
$(1): $$(TGT_$(1))

clean-$(1):
	rm -f $$(CLEAN_$(1))
	rm -rf $$(DIR_CLEAN_$(1))
endef

subdirs := analytical_comparisons multi_material optimisation_checkpointing scalar_advection scalar_advection_diffusion

$(foreach dir,$(subdirs),$(eval $(include_subdir)))

.PHONY: targets
targets: $(TGT)

.PHONY: clean
clean:
	rm -f $(CLEAN)
	rm -rf $(DIR_CLEAN)


# ##############

# cases := analytical_comparisons multi_material optimisation_checkpointing scalar_advection scalar_advection_diffusion Drucker-Prager_rheology adjoint 2d_cylindrical_TALA_DG viscoplastic_case_DG free_surface viscoelastic base_gmsh
# long_cases := analytical_comparisons parallel_scaling

# .PHONY: all longtest longtest_output $(cases) $(long_cases) clean check longcheck

# all: $(cases)

# longtest: $(long_cases)

# longtest_output: $(long_cases)

# # sort to remove duplicates which define both short and long tests
# $(sort $(cases) $(long_cases)):
# 	$(MAKE) -C $@ $(MAKECMDGOALS)

# clean: $(cases)
# 	rm -rf __pycache__

# check:
# 	python -m pytest . ../demos -m 'not demo and not longtest'

# longcheck:
# 	python -m pytest -m longtest
