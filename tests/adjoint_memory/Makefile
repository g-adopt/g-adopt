include ../../rules.mk

scheduler := noscheduler fullmemory fullstorage

.PHONY: all clean check

all: $(addsuffix .conv,$(scheduler))

%.conv : category := tests/adjoint
%.conv : ncpus := 1
%.conv : exec_args = $*
%.conv : desc = $(exec_args)
%.conv: memory_test.py
	$(run-python)

clean:
	rm -f *.h5 *.conv *.dat
	rm -rf __pycache__
