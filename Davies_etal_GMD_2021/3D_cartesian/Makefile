all: params.log

nthreads := 8

params.log: 3d_cartesian.py
	mpiexec -np $(nthreads) python $<

clean:
	rm -f *.vtu *.pvtu *.h5 params.log
