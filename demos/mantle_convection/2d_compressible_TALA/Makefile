all: params.log

current_dir := $(notdir $(patsubst %/,%,$(CURDIR)))
test_class := $(notdir $(patsubst %/,%,$(dir $(CURDIR))))
ncpus := 4

params.log: 2d_compressible_TALA.py
	echo "running $<" >&2
	/usr/bin/time --format="$< finished in %E" tsp -L mantle_convection -N $(ncpus) -f mpiexec -np $(ncpus) python3 $< -Stokes_pc_factor_mat_solver_type superlu_dist

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state

check: params.log
	python3 -m pytest ../../test_all.py -k $(test_class)/$(current_dir)