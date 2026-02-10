cases := damping smoothing Tobs uobs uimposed
schedulers := noscheduler fullmemory fulldisk

combo_cases := $(foreach c,$(cases),$(foreach s,$(schedulers),$(c)_$(s)))
cores := 16

.PHONY: longtest check clean

longtest: $(addsuffix _functional.dat,$(combo_cases))

.ONESHELL:
SHELL = /bin/bash

Checkpoint_State.h5: forward.py
	mkdir -p batch_output
	gadopt_hpcrun -n ${cores} -N $@ -o batch_output/$(basename $@).out -e batch_output/$(basename $@).err -v PYOP2_SPMD_STRICT=1 python3 $<

%_functional.dat: inverse.py Checkpoint_State.h5
	echo "running spherical $@" >&2
	mkdir -p batch_output
	gadopt_hpcrun -n ${cores} -m 64GB -N $* -o batch_output/$*.out -e batch_output/$*.err -v PYOP2_SPMD_STRICT=1 python3 $< $*

longtest_output:
	[[ -d batch_output ]] && tail -n +1 batch_output/*

check: $(addsuffix _functional.dat,$(combo_cases))
	python -m pytest -m longtest

clean:
	rm -f -r *.conv *_functional.dat *_res.h5 firedrake_adjoint_checkpoint_* vtu-files Checkpoint_State.h5 batch_output
