include ../../rules.mk

all: params.log

.PHONY: all clean

%.msh: %.geo
	gmsh -2 $<

params.log: category := tests/gmsh
params.log: run-base_case square.msh

run-base_case: square.msh

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.msh
	rm -rf output

check: params.log
	python3 -m pytest
