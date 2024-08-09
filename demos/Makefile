cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint multi_material gplates_global free_surface
demo_cases := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 3d_cartesian gplates_global multi_material/compositional_buoyancy multi_material/thermochemical_buoyancy free_surface

.PHONY: all convert_demos $(cases) $(demo_cases) clean generate

all: $(cases)

convert_demos: $(foreach case,$(demo_cases),$(case)/$(notdir $(case)).ipynb)

# sort to remove duplicates which define both tests and demos
$(sort $(cases) $(demo_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))
