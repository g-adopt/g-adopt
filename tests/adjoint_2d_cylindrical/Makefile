include ../../rules.mk

cases := damping smoothing Tobs uobs uimposed
schedulers := noscheduler fullmemory fulldisk

combo_cases := $(foreach c,$(cases),$(foreach s,$(schedulers),$(c)_$(s)))
cores := 16
project := xd2

.PHONY: longtest check clean

longtest: $(addsuffix _functional.dat,$(combo_cases))

.ONESHELL:
SHELL = /bin/bash

Checkpoint_State.h5: forward.py
	mkdir -p pbs_output
	qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W block=true -N $@ -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=${cores},walltime=03:00:00,mem=80GB,wd,jobfs=50GB -q normalsr -P $(project) -o pbs_output/$(basename $@).out -e pbs_output/$(basename $@).err -- ./run_gadi.sh python3 $<

%_functional.dat: inverse.py Checkpoint_State.h5
	echo "running spherical $@" >&2
	mkdir -p pbs_output
	qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W block=true -N $* -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=$(cores),walltime=04:00:00,mem=80GB,wd,jobfs=200GB -q normalsr -P $(project) -o pbs_output/$*.out -e pbs_output/$*.err -- ./run_gadi.sh python3 $< $*

check: $(addsuffix _functional.dat,$(combo_cases))
	python -m pytest -m longtest

clean:
	rm -f -r *.conv *_functional.dat *_res.h5 firedrake_adjoint_checkpoint_* vtu-files Checkpoint_State.h5 pbs_output
