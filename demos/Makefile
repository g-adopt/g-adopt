cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical analytical_comparisons Drucker-Prager_rheology 3d_spherical 3d_cartesian adjoint optimisation_checkpointing
long_cases := analytical_comparisons parallel_scaling

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean generate

all: $(cases)

longtest: $(long_cases)

longtest_output: $(long_cases)

# sort to remove duplicates which define both short and long tests
$(sort $(cases) $(long_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)
