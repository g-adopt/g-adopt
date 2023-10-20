all: params.log

ncpus := 4

params.log: viscoplastic_case_dg_limiter.py
	echo "running $<" >&2
	/usr/bin/time --format="$< finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
