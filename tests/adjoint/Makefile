include ../../rules.mk

cases := damping smoothing Tobs uobs uimposed
scheduler := noscheduler fullmemory fullstorage

cases_schedulers := $(foreach case,$(cases),$(foreach sched,$(scheduler),$(case)_$(sched)))

.PHONY: all clean check

all: $(addsuffix .conv,$(cases_schedulers))

%.conv : category := tests/adjoint
%.conv : ncpus := 2
%.conv : exec_args = $*
%.conv : desc = $(exec_args)
%.conv: taylor_test.py adjoint-demo-checkpoint-state.h5
	$(run-python)

adjoint-demo-checkpoint-state.h5 : category := tests/adjoint
adjoint-demo-checkpoint-state.h5 : ncpus := 2
adjoint-demo-checkpoint-state.h5 : desc := forward
adjoint-demo-checkpoint-state.h5: ../../demos/mantle_convection/adjoint/adjoint_forward.py
	$(run-python)

clean:
	rm -f *.h5 *.conv *.dat
	rm -rf __pycache__

check: $(addsuffix .conv,$(cases))
	python -m pytest
