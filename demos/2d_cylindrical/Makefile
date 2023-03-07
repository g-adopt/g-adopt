all: params.log

ncpus := 4

params.log: 2d_cylindrical.py
	echo "running $<" >&2
	tsp -N $(ncpus) -nf mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
