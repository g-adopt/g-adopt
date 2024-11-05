all: params.log

demo_name := $(notdir $(CURDIR))
ncpus := 16

params.log: $(demo_name).py
	$(info running $<)
	/usr/bin/time --format="$< finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state __pycache__

check: params.log
	python3 -m pytest ../../demos/test_all.py -k ../tests/$(demo_name)