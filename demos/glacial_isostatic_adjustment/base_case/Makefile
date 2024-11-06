all: params.log
current_dir := $(notdir $(patsubst %/,%,$(CURDIR)))
test_class := $(notdir $(patsubst %/,%,$(dir $(CURDIR))))

params.log: base_case.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif
	rm -rf output

check: params.log
	python3 -m pytest ../../test_all.py -k $(test_class)/$(current_dir)
