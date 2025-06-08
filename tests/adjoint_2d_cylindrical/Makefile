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
	jobid=$$(cat .fwd_job_id); \
	python3 driver.py submit -t "qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W block=true -W depend=afterok:$${jobid} -N {test_mode} -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=$(cores),walltime=04:00:00,mem={mem}GB,wd,jobfs=200GB -q normalsr -P $(project) -o pbs_output/{test_mode}.out -e pbs_output/{test_mode}.err -- ./run_gadi.sh " $*
	wait

Checkpoint_State.h5: forward.py clean
	mkdir -p pbs_output
	echo $@
	fwd_jobid=$$(qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -N $@ -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus=${cores},walltime=03:00:00,mem=80GB,wd,jobfs=50GB -q normalsr -P $(project) -o pbs_output/$(basename $@).out -e pbs_output/$(basename $@).err -- mpiexec python3 ./forward.py); \
	echo $$fwd_jobid > .fwd_job_id

longcheck: $(addsuffix .conv,$(combo_cases))
	python -m pytest -m longtest_temp

clean:
	rm -f -r *.conv pbs_output
