include ../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := tests
params.log : ncpus := 16
params.log: 2d_cylindrical_TALA_DG.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../demos/test_all.py -k "../tests/2d_cylindrical_TALA_DG"
