include ../../rules.mk

cases := functional_boundary.txt functional_field.txt

.PHONY: all clean check

all: $(cases)

functional_%.txt : category := pde_constrained
$(cases): functional_%.txt: PDE_constrained_%.py
	$(run-python)

clean:
	rm $(cases) *.h5
	rm -rf __pycache__

check: $(cases)
	python3 -m pytest test_pde_constrained_optimisation.py
