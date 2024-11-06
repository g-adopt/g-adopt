include makefile.cases.inc

.PHONY: all convert_demos generate clean check
.PHONY: $(mc_cases) $(mm_cases) $(gia_cases)
.PHONY: $(mc_dir) $(mm_dir) $(gia_dir)

all: $(mc_cases) $(mm_cases) $(gia_cases)

convert_demos: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages .diagram.mermaid
	tar --transform='s|.*/||' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar free_surface/temperature_warp.gif

# Explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
$(mc_dir)/adjoint/adjoint.ipynb: $(mc_dir)/adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
$(mc_dir)/free_surface/free_surface.ipynb: $(mc_dir)/base_case/base_case.ipynb

# Invoke as `make mantle_convection/base_case` to only run the mantle_convection/base_case
# demo or `make mantle_convection` to run all mantle convection demos.
$(mc_cases) $(mm_cases) $(gia_cases) $(mc_dir) $(mm_dir) $(gia_dir):
	$(MAKE) -C $@

generate:
	python3 generate_expected.py

clean:
	rm -rf __pycache__
	@$(foreach case,$(mc_cases) $(mm_cases) $(gia_cases),$(MAKE) -C $(case) $(MAKECMDGOALS);)

check:
	python3 -m pytest -m demo

# pattern rule for executing demo scripts as a notebook
%.ipynb: %.py
	python3 -m jupytext --to ipynb --execute $<
