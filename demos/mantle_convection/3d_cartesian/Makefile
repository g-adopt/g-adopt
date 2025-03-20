include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := $(demo_type)
params.log : ncpus := 4
params.log: $(demo_name).py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state

check: params.log
	python3 -m pytest ../../test_all.py -k $(demo_type)/$(demo_name)
