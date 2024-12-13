test_name := $(notdir $(CURDIR))

.PHONY: all clean

all: params.log

%.msh: %.geo
	gmsh -2 $<

params.log: $(test_name).py square.msh
	$(info running $<)
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.msh
	rm -rf output __pycache__

check: params.log
	python3 -m pytest
