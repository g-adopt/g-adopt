cases := $(shell python3 test_implicit_viscous_surface.py)

all: $(cases)

$(cases): implicit_viscous_surface.py
	echo "running implicit free surface coupling " >&2
#	tsp -N 1 -f python3 $< 
	python3 $< 
