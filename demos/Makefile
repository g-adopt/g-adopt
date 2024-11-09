include makefile.demos.inc

.PHONY: all convert_demos generate clean check $(all_demo_paths) $(all_dirs)

all: $(all_demo_paths)

convert_demos: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages .diagram.mermaid
	tar --transform='s|/.*/|/|' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar $(mc_dir)/free_surface/temperature_warp.gif

# Explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
$(mc_dir)/adjoint/adjoint.ipynb: $(mc_dir)/adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
$(mc_dir)/free_surface/free_surface.ipynb: $(mc_dir)/base_case/base_case.ipynb

# Invoke as `make mantle_convection/base_case` to only run the mantle_convection/base_case
# demo or `make mantle_convection` to run all mantle convection demos.
$(all_demo_paths) $(all_dirs):
	$(MAKE) -C $@

generate:
	python3 generate_expected.py

clean:
	rm -rf __pycache__
	@$(foreach demo,$(all_demo_paths),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check:
	python3 -m pytest -m demo

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $<
