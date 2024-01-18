cases := explicit implicit

all: $(cases)

$(cases): 
	echo "running $@ free surface coupling "
	tsp -N 1 -f python3 $@_viscous_surface.py
