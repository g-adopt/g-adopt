include ../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := tests
params.log : ncpus := 4
params.log: viscoplastic_case_DG.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../demos/test_all.py -k "../tests/viscoplastic_case_DG"
