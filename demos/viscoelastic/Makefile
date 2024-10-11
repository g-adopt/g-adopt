cases := viscoelastic_loading 

all: $(cases)

$(cases): 
	echo "running $@  "
	tsp -N 1 -f python3 $@.py
