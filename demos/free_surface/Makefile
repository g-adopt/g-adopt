cases := explicit implicit implicit_top_bottom
parallel := implicit_top_bottom_buoyancy implicit_cylindrical

ncpus := 8

all: $(cases) $(parallel)


$(cases): 
	echo "running $@ free surface coupling "
	tsp -N 1 -f python3 $@_viscous_surface.py

$(parallel):
	echo "running $@ free surface coupling "
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $@_viscous_surface.py
