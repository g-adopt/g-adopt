cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical analytical_comparisons Drucker-Prager_rheology 3d_spherical 3d_cartesian adjoint optimisation_checkpointing

.PHONY: all $(cases) clean generate

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)
