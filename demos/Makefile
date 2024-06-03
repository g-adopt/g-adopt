cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical analytical_comparisons Drucker-Prager_rheology 3d_spherical 3d_cartesian adjoint optimisation_checkpointing scalar_advection_diffusion multi_material
long_cases := analytical_comparisons parallel_scaling
demo_cases := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 3d_cartesian multi_material/compositional_buoyancy multi_material/thermochemical_buoyancy

.PHONY: all longtest longtest_output convert_demos $(cases) $(long_cases) clean generate

all: $(cases)

longtest: $(long_cases)

longtest_output: $(long_cases)

convert_demos: $(foreach case,$(demo_cases),$(case)/$(notdir $(case)).ipynb)

# sort to remove duplicates which define both short and long tests
$(sort $(cases) $(long_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))
