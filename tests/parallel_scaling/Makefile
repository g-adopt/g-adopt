levels := 5 6 7
levels := $(addsuffix .txt,$(addprefix profile_,$(levels)))

.PHONY: longtest clean

longtest: $(levels)

.ONESHELL:
SHELL = /bin/bash
profile_%.txt: scaling.py stokes_cubed_sphere.py
	echo "running parallel scaling on level $*"
	mkdir -p pbs_output
	python3 scaling.py submit -t "qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W block=true -N scaling_{level} -l storage=gdata/xd2+scratch/xd2,ncpus={cores},walltime=04:00:00,mem={mem}GB,wd,jobfs=200GB -q normalsr -P $(project) -o pbs_output/l{level}.out -e pbs_output/l{level}.err -- ./run_gadi.sh {level}" $*

clean:
	rm -rf pbs_output profile_*.txt level_*.out level_*.err

longtest_output:
	tail -n +1 level_*.{err,out}
