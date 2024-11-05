mc_dir=mantle_convection
mc_demos := base_case 3d_cartesian viscoplastic_case free_surface
mc_demos += 2d_cylindrical 3d_spherical gplates_global
mc_demos += 2d_compressible_TALA 2d_compressible_ALA
mc_demos += adjoint

mm_dir=multi_material
mm_demos := compositional_buoyancy thermochemical_buoyancy

gia_dir=glacial_isostatic_adjustment
gia_demos := gia_base_case gia_2d_cylindrical

notebook_files := $(foreach case,$(mc_demos),$(mc_dir)/$(case)/$(notdir $(case)).ipynb)
notebook_files := $(foreach case,$(mm_demos),$(mm_dir)/$(case)/$(notdir $(case)).ipynb)
notebook_files := $(foreach case,$(gia_demos),$(gia_dir)/$(case)/$(notdir $(case)).ipynb)
notebook_files += ${mc_dir}/adjoint/adjoint_forward.ipynb
notebook_files += ${mc_dir}/visualise_ALA_p_nullspace/visualise_ALA_p_nullspace.ipynb

.PHONY: all convert_demos $(mc_demos) clean generate check $(mm_demos)

all: $(mc_demos) $(mm_demos) $(gia_demos)

convert_demos: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages .diagram.mermaid
	tar --transform='s|.*/||' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar free_surface/temperature_warp.gif

# explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
$(mc_dir)/adjoint/adjoint.ipynb: $(mc_dir)/adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
$(mc_dir)/free_surface/free_surface.ipynb: $(mc_dir)/base_case/base_case.ipynb

$(mc_demos):
	$(MAKE) -C $(mc_dir)/$@ $(MAKECMDGOALS)

$(mm_demos):
	$(MAKE) -C $(mm_dir)/$@ $(MAKECMDGOALS)

$(gia_demos):
	$(MAKE) -C $(gia_dir)/$@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: all
	rm -rf __pycache__

check:
	python3 -m pytest -m demo

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $< --run-path $(dir $(abspath $<))
