all: params.log
current_dir := $(notdir $(patsubst %/,%,$(CURDIR)))
test_class := $(notdir $(patsubst %/,%,$(dir $(CURDIR))))

params.log: free_surface.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -L mantle_convection -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest ../../test_all.py -k $(test_class)/$(current_dir)