cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint multi_material gplates_global free_surface viscoelastic
demo_cases := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA visualise_ALA_p_nullspace 2d_compressible_TALA viscoplastic_case 3d_cartesian gplates_global multi_material/compositional_buoyancy multi_material/thermochemical_buoyancy adjoint free_surface

notebook_files := $(foreach case,$(demo_cases),$(case)/$(notdir $(case)).ipynb) adjoint/adjoint_forward.ipynb

.PHONY: all convert_demos $(cases) $(demo_cases) clean generate

all: $(cases)

convert_demos: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages .diagram.mermaid
	tar --transform='s|.*/||' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar free_surface/temperature_warp.gif

# explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
adjoint/adjoint.ipynb: adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
free_surface/free_surface.ipynb: base_case/base_case.ipynb

# sort to remove duplicates which define both tests and demos
$(sort $(cases) $(demo_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))
