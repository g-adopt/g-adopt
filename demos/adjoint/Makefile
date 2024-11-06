all: functional.txt

ncpus := 2

functional.txt: adjoint.py adjoint-demo-checkpoint-state.h5
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

adjoint-demo-checkpoint-state.h5: adjoint_forward.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -rf functional.txt *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint
