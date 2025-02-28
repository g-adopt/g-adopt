include ../../rules.mk

all: final_error.log

.PHONY: all clean

final_error.log: category := tests

final_error.log: run-scalar_advection

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log
	rm -rf output

check: final_error.log
	python3 -m pytest
