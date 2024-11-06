all: functional_boundary.txt functional_field.txt

functional_%.txt: PDE_constrained_%.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm functional_*.txt
