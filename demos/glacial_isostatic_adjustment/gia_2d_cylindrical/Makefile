all: params.log

demo_type := $(notdir $(realpath $(dir $(CURDIR))))
demo_name := $(notdir $(CURDIR))

params.log: $(demo_name).py
	$(info running $<)
	/usr/bin/time --format="$< finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif *.msh
	rm -rf density ice mesh output viscosity

check: params.log
	python3 -m pytest ../../test_all.py -k $(demo_type)/$(demo_name)
