all: params.log minimal_params.log

params.log: base_case.py
	mpiexec -np $(nthreads) python $<

minimal_params.log: minimal_base_case_test.py
	mpiexec -np $(nthreads) python $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log minimal_params.log
