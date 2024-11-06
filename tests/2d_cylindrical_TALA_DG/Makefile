all: params.log

ncpus := 4

params.log: 2d_cylindrical_TALA_DG.py
	echo "running $<" >&2
	/usr/bin/time --format="$< finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state __pycache__

check: params.log
	python3 -m pytest ../../demos/test_all.py -k "../tests/2d_cylindrical_TALA_DG"