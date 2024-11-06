all: params.log
current_dir := $(notdir $(patsubst %/,%,$(CURDIR)))

params.log: gplates_global.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest ../test_all.py -k $(current_dir)