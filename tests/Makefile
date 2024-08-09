cases := analytical_comparisons multi_material optimisation_checkpointing scalar_advection_diffusion Drucker-Prager_rheology free_surface
long_cases := analytical_comparisons parallel_scaling

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean

all: $(cases)

longtest: $(long_cases)

longtest_output: $(long_cases)

# sort to remove duplicates which define both short and long tests
$(sort $(cases) $(long_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(cases)
