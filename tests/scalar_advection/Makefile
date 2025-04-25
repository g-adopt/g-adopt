include ../../rules.mk

.PHONY: all clean check

all: final_error.log

final_error.log : category := tests
final_error.log: scalar_advection.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log
	rm -rf output

check: final_error.log
	python3 -m pytest
