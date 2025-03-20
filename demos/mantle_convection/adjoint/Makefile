include ../../../rules.mk

.PHONY: all clean check

all: functional.txt

functional.txt : category := $(demo_type)
functional.txt: adjoint.py adjoint-demo-checkpoint-state.h5
	$(run-python)

adjoint-demo-checkpoint-state.h5 : category := $(demo_type)
adjoint-demo-checkpoint-state.h5 : ncpus := 2
adjoint-demo-checkpoint-state.h5: adjoint_forward.py
	$(run-python)

clean:
	rm -rf functional.txt *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint
	rm -rf output

check: functional.txt
	python3 -m pytest
