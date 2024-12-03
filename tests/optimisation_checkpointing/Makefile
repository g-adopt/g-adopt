all: full_optimisation_np1.dat full_optimisation_np2.dat

full_optimisation_np%.dat: helmholtz.py
	echo "running helmholtz checkpointing (np $*)" >&2
	/usr/bin/time --format="helmholtz checkpointing (np $*) took %E" tsp -N $* -f mpiexec -np $* python3 $<

clean:
	rm -rf optimisation_checkpoint_* __pycache__ *.h5 *.dat

check: full_optimisation_np1.dat full_optimisation_np2.dat
	python -m pytest
