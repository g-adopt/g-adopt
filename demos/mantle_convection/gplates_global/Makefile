all: params.log

demo_type := $(notdir $(realpath $(dir $(CURDIR))))
demo_name := $(notdir $(CURDIR))

params.log: gplates_global.py
	$(info running $<)
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest ../../test_all.py -k $(demo_type)/$(demo_name)