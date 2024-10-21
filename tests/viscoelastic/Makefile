cases := zhong

all: $(cases)


$(cases): 
	echo "running viscoelastic $@ free surface formulation "
	tsp -N 1 -f python3 $@_viscoelastic_free_surface.py --case elastic
	tsp -N 1 -f python3 $@_viscoelastic_free_surface.py --case viscoelastic
	tsp -N 1 -f python3 $@_viscoelastic_free_surface.py --case viscous
