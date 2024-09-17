serial_cases := explicit implicit implicit_top_bottom
parallel_cases := implicit_top_bottom_buoyancy implicit_cylindrical

ncpus := 8

all: $(serial_cases) $(parallel_cases)

$(serial_cases): 
	echo "running $@ free surface coupling "
	tsp -N 1 -f python3 $@_free_surface.py

$(parallel_cases):
	echo "running $@ free surface coupling "
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $@_free_surface.py
