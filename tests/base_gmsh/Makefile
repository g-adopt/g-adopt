all: params.log

.PHONY: all base_case.py

%.msh: %.geo
	gmsh -2 $<

params.log: base_case.py square.msh
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.msh
	rm -rf output

check: params.log
	python3 -m pytest