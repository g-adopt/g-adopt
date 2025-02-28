include ../../../rules.mk

all: functional.txt

functional.txt : category := mantle_convection
functional.txt: run-adjoint adjoint-demo-checkpoint-state.h5

run-adjoint: adjoint-demo-checkpoint-state.h5

adjoint-demo-checkpoint-state.h5 : category := mantle_convection
adjoint-demo-checkpoint-state.h5 : ncpus := 2
adjoint-demo-checkpoint-state.h5: run-adjoint_forward

clean:
	rm -rf functional.txt *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint

check: functional.txt
	python3 -m pytest
