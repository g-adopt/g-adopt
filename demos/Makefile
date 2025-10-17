include makefile.cases.inc

.PHONY: all convert_demos clean generate $(mc_cases) $(mm_cases) $(gia_cases) $(extra_cases) $(mc_dir) $(mm_dir) $(gia_dir)

all: $(mc_cases) $(mm_cases) $(gia_cases) $(extra_cases)

convert_demos: $(notebook_files) .pages $(diagram_files)
	tar --transform='s/.pages/CONTENTS.md/' --create --file artifact.tar .pages $(diagram_files)
	tar --transform='s|/.*/|/|' --append --file artifact.tar $(notebook_files)
	tar --append --file artifact.tar $(mc_dir)/free_surface/temperature_warp.gif
	tar --append --file artifact.tar $(gia_dir)/base_case/displacement_warp.gif
	tar --append --file artifact.tar $(gia_dir)/2d_cylindrical/displacement_warp.gif

# explicit dependencies between notebooks
# adjoint uses the checkpoint from the forward run
$(mc_dir)/adjoint/adjoint.ipynb: $(mc_dir)/adjoint/adjoint_forward.ipynb

# free surface has a comparison plot against the base case
$(mc_dir)/free_surface/free_surface.ipynb: $(mc_dir)/base_case/base_case.ipynb

# dynamic topography requires the checkpoint from the base case
$(mc_dir)/dynamic_topography/dynamic_topography.ipynb: $(mc_dir)/base_case/base_case.ipynb

### make mantle_convection/base_case should just run the mantle_convection/base_case demo
### make mantle_convection should run all mantle convection demos
$(all_cases) $(mc_dir) $(mm_dir) $(gia_dir):
	$(MAKE) -C $@

generate:
	python3 generate_expected.py

clean:
	rm -rf __pycache__
	@$(foreach case, $(all_cases), \
		$(MAKE) -C $(case) $(MAKECMDGOALS) ; \
	)

check:
	python3 -m pytest -m demo

# pattern rule for executing demo scripts as a notebook
# because jupyter is an insane jumble of components, there's absolutely no way to configure it
# to try to bind to its control port when it needs it. instead, the provisioner checks for a free
# port and writes that to a connection file. this is of course completely prone to race conditions.
#
# in the CI pipeline, where we're likely to be running several conversions in parallel, the
# `jupyter-nbconvert` command is a script that runs in an isolated network namespace, thus
# eliminating potential port conflicts. this does effectively require root in the CI environment
# because the loopback interface needs to be brought up in the namespace
%.ipynb: %.py
	jupytext --to ipynb $<
	jupyter-nbconvert --to notebook --execute --inplace \
		--TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["exercise"]' \
		$@
