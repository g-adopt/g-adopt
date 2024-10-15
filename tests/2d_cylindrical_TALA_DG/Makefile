all: params.log

ncpus := 1

params.log: 2d_cylindrical_TALA_DG.py
	echo "running $<" >&2
	/usr/bin/time --format="$< finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
