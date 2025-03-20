include ../../rules.mk

.PHONY: all clean check

all: params.log

%.msh: %.geo
	gmsh -2 $<

params.log : category := tests/gmsh
params.log: base_case.py square.msh
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.msh
	rm -rf output __pycache__

check: params.log
	python3 -m pytest
