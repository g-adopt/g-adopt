include ../../rules.mk

cases := functional_boundary.txt functional_field.txt

all: $(cases)

functional_%.txt : category := pde_constrained
$(cases): functional_%.txt: run-PDE_constrained_%

clean:
	rm $(cases) *.h5
	rm -rf __pycache__

check: $(cases)
	python3 -m pytest test_pde_constrained_optimisation.py
