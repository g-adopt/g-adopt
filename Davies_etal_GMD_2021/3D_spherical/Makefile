all: params.log

nthreads := 16

params.log: 3d_spherical.py
	mpiexec -np $(nthreads) python $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
