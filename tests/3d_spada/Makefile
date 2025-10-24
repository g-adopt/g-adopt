include ../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := tests
params.log : ncpus := 8
params.log : exec_args = --reflevel 4 --DG0_layers 2
params.log: 3d_spada.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 *.dat params.log
	rm -rf output reference_state __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../demos/test_all.py -k "../tests/3d_spada"
