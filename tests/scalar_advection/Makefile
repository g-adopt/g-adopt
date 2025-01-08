all: final_error.log

.PHONY: all clean

final_error.log: scalar_advection.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log
	rm -rf output

check: final_error.log
	python3 -m pytest
