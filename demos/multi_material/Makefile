cases := compositional_buoyancy thermochemical_buoyancy

.PHONY: all $(cases)

all: $(cases)

$(cases):
	cd $@; /usr/bin/time --format="$@ finished in %E" tsp -f python3 $@.py

