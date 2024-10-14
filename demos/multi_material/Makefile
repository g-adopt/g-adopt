cases := compositional_buoyancy thermochemical_buoyancy free_surface
ncpus := 4

all: $(cases)

$(cases):
	echo "running $@.py" >&2
	cd $@ && /usr/bin/time --format="$@.py finished in %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $@.py

clean:
	rm -f */*.pvd */*.vtu */*.pvtu */*.h5 */params.log

