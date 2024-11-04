cases := compositional_buoyancy thermochemical_buoyancy

.PHONY: all $(cases)

all: $(cases)

$(cases):
	cd $@; /usr/bin/time --format="$@ finished in %E" tsp -f python3 $@.py

clean:
	@$(foreach case, $(cases), \
		rm -rf $(case)/*.h5 $(case)/*.pvd $(case)/params.log $(case)/*.msh $(case)/output ; \
	)

