cases := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint gplates_global free_surface
demo_cases := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA visualise_ALA_p_nullspace 2d_compressible_TALA viscoplastic_case 3d_cartesian gplates_global adjoint free_surface
mc_dir="mantle_convection/"

mm_cases := multi_material
mm_demo_cases := multi_material/compositional_buoyancy multi_material/thermochemical_buoyancy

gia_cases := gia_base_case gia_2d_cylindrical
gia_demo_cases := gia_base_case gia_2d_cylindrical
gia_dir="glacial_isostatic_adjustment/"

notebook_files := $(foreach case,$(demo_cases),$(case)/$(notdir $(case)).ipynb) adjoint/adjoint_forward.ipynb

.PHONY: all convert_demos $(cases) $(demo_cases) clean generate $(mm_cases) $(mm_demo_cases)

all: $(cases) $(mm_cases) $(gia_cases)

convert_demos: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages .diagram.mermaid
	tar --transform='s|.*/||' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar free_surface/temperature_warp.gif

# explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
$(mc_dir)/adjoint/adjoint.ipynb: $(mc_dir)adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
$(mc_dir)free_surface/free_surface.ipynb: $(mc_dir)base_case/base_case.ipynb

# sort to remove duplicates which define both tests and demos
$(sort $(cases) $(demo_cases)):
	$(MAKE) -C $(mc_dir)$@ $(MAKECMDGOALS)

$(sort $(mm_cases) $(mm_demo_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

$(sort $(gia_cases) $(gia_demo_cases)):
	$(MAKE) -C $(gia_dir)$@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)
	rm -rf __pycache__

check:
	python3 -m pytest -m demo

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))
