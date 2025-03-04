include ../../../rules.mk

.PHONY: all clean check

all: functional.txt

functional.txt : category := mantle_convection
functional.txt: adjoint.py adjoint-demo-checkpoint-state.h5
	$(run-python)

adjoint-demo-checkpoint-state.h5 : category := mantle_convection
adjoint-demo-checkpoint-state.h5 : ncpus := 2
adjoint-demo-checkpoint-state.h5: adjoint_forward.py
	$(run-python)

clean:
	rm -rf functional.txt *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint

check: functional.txt
	python3 -m pytest
