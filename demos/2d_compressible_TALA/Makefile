all: params.log

ncpus := 4

params.log: compressible_case_TALA.py
	echo "running $<" >&2
	tsp -N $(ncpus) -nf mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
