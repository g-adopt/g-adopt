cases := smooth/cylindrical/free_slip smooth/cylindrical/zero_slip delta/cylindrical/free_slip delta/cylindrical/zero_slip delta/cylindrical/free_slip_dpc delta/cylindrical/zero_slip_dpc
long_cases := smooth/cylindrical/free_surface smooth/spherical/free_slip smooth/spherical/zero_slip

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean

all: $(cases)

longtest: $(long_cases)

$(cases): analytical.py
	echo "running analytical $@" >&2
	/usr/bin/time --format="analytical $@ finished in %E" python3 analytical.py submit -t "tsp -N {cores} -f mpiexec -np {cores}" $@

.ONESHELL:
SHELL = /bin/bash
$(long_cases): analytical.py clean
	exec {out}<> <(:) # create new named FD
	qsub -W depend=on:$$(python3 analytical.py count $@) -N sentinel -W block=true -l ncpus=1,walltime=00:00:30,wd -q normal -P $(project) -- /bin/true >&"$$out" &
	read -u "$$out" sjob
	echo "running spherical $@, waiting on $$sjob" >&2
	mkdir -p pbs_output
	python3 analytical.py submit -t "qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W depend=beforeany:$$sjob -N analytical_{params} -l storage=gdata/xd2+scratch/xd2,ncpus={cores},walltime=01:00:00,mem={mem}GB,wd,jobfs=10GB -q normal -P $(project) -o pbs_output/$(subst /,-,$@)_l{level}_{params}.out -e pbs_output/$(subst /,-,$@)_l{level}_{params}.err -- ./run_gadi.sh" $@
	wait

clean:
	rm -rf pbs_output *.dat sentinel.*

longtest_output:
	[[ -d pbs_output ]] && tail -n +1 pbs_output/*
