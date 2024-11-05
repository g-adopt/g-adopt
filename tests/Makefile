cases := 2d_cylindrical_TALA_DG adjoint analytical_comparisons Drucker-Prager_rheology
cases += free_surface multi_material optimisation_checkpointing
cases += scalar_advection_diffusion viscoelastic viscoplastic_case_DG
long_cases := analytical_comparisons parallel_scaling

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean check longcheck

all: $(cases)

longtest: $(long_cases)

longtest_output: $(long_cases)

# sort to remove duplicates which define both short and long tests
$(sort $(cases) $(long_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(cases)
	rm -rf __pycache__

check:
	python -m pytest . ../demos -m 'not demo and not longtest'

longcheck:
	python -m pytest -m longtest