all: params.log

nthreads := 4

params.log: 2d_cylindrical.py
	mpiexec -np $(nthreads) python $<

clean:
	rm -f *.vtu *.pvtu *.h5 params.log
