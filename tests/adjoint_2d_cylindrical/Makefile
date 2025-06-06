include ../../rules.mk

cases := damping
schedulers := noscheduler fullmemory

combo_cases := $(foreach c,$(cases),$(foreach s,$(schedulers),$(c)_$(s)))
cores := 16
project := xd2

.PHONY: longtest clean check

longtest: $(addsuffix .conv,$(combo_cases))

.ONESHELL:
SHELL = /bin/bash
%.conv: inverse.py Checkpoint_State.h5
	echo "running spherical $@" >&2
	mkdir -p pbs_output
	qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -N adj_$* -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=${cores},walltime=03:00:00,mem=80GB,wd,jobfs=50GB -q normalsr -P $(project) -o pbs_output/$*.out -e pbs_output/$*.err -- mpiexec -np 8 python3 ./inverse.py $* &
	wait

Checkpoint_State.h5: forward.py clean
	mkdir -p pbs_output
	qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -N $@ -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=${cores},walltime=03:00:00,mem=80GB,wd,jobfs=50GB -q normalsr -P $(project) -o pbs_output/$*.out -e pbs_output/$*.err -- mpiexec -np 8 python3 ./forward.py $* $*
	wait

longcheck: $(addsuffix .conv,$(combo_cases))
	python -m pytest -m longtest_temp

clean:
	rm -f *.conv
