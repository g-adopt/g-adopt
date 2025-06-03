include ../../rules.mk

cases := damping smoothing Tobs uimposed uobs
schedulers := noscheduler fullmemory

combo_cases := $(foreach c,$(cases),$(foreach s,$(schedulers),$(c)_$(s)))

.PHONY: all clean check

all: $(addsuffix .conv,$(combo_cases))

%.conv : category := tests/adjoint_2d_cylindrical
%.conv : ncpus := 16
%.conv : exec_args = $*
%.conv : desc = $(exec_args)
%.conv: inverse.py Checkpoint_State.h5
	$(run-python)
	touch $@

Checkpoint_State.h5 : category := tests/adjoint_2d_cylindrical
Checkpoint_State.h5 : ncpus := 16
Checkpoint_State.h5 : desc := forward
Checkpoint_State.h5: forward.py
	$(run-python)

check: $(addsuffix .conv,$(combo_cases))
	python3 -m pytest $(CURDIR)/test_cylindrical_adjoint.py

clean:
	rm -f *.conv
