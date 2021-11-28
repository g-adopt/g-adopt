all: params.log

params.log: base_case.py
	mpiexec -np $(nthreads) python $<

clean:
	rm -f *.vtu *.pvtu *.h5 params.log
