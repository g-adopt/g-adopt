cases := functional_boundary.txt functional_field.txt

all: $(cases)

functional_%.txt: PDE_constrained_%.py
	$(info running $<)
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm $(cases) *.h5
	rm -rf __pycache__

check: $(cases)
	python3 -m pytest test_pde_constrained_optimisation.py
