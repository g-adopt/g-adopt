all: params.log

ncpus := 4

params.log: 2d_compressible_TALA.py
	echo "running $<" >&2
	/usr/bin/time --format="$< finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $< -Stokes_pc_factor_mat_solver_type superlu_dist

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
