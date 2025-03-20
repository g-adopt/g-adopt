tests := 2d_cylindrical_TALA_DG adjoint analytical_comparisons base_gmsh
tests += Drucker-Prager_rheology free_surface multi_material optimisation_checkpointing
tests += scalar_advection scalar_advection_diffusion viscoelastic viscoplastic_case_DG
long_tests := analytical_comparisons parallel_scaling
# sort to remove duplicates which define both short and long tests
all_tests := $(sort $(tests) $(long_tests))

.PHONY: all longtest longtest_output $(tests) $(long_tests) clean check longcheck

all: $(tests)

longtest: $(long_tests)

longtest_output: $(long_tests)

$(all_tests):
	$(MAKE) -C $@

clean:
	rm -rf __pycache__
	@$(foreach test,$(all_tests),$(MAKE) -C $(test) $(MAKECMDGOALS);)

check: $(tests)
	python -m pytest . ../demos -m 'not demo and not longtest'

longcheck: $(long_tests)
	python -m pytest -m longtest
