include ../../rules.mk

.PHONY: all clean check

all: final_error.log

final_error.log : category := tests
final_error.log: scalar_advection_DGlim.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log
	rm -rf DG_*

check: final_error.log
	python3 -m pytest
