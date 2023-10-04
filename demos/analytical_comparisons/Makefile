cases := smooth/cylindrical/free_slip smooth/cylindrical/zero_slip delta/cylindrical/free_slip delta/cylindrical/zero_slip delta/cylindrical/free_slip_dpc delta/cylindrical/zero_slip_dpc
spherical_cases := smooth/spherical/free_slip smooth/spherical/zero_slip

.PHONY: all $(cases) $(spherical_cases)

all: $(cases)

$(cases): analytical.py
	echo "running analytical $@" >&2
	/usr/bin/time --format="analytical $@ finished in %E" python3 analytical.py submit -t "tsp -N {cores} -f mpiexec -np {cores}" $@

project := xd2

.ONESHELL:
SHELL = /bin/bash
$(spherical_cases): analytical.py
	exec {out}<> <(:) # create new named FD
	qsub -W depend=on:$$(python3 analytical.py count $@) -N sentinel -W block=true -l ncpus=1,walltime=00:00:30,wd -q normal -P $(project) -- /bin/true >&"$$out" &
	read -u "$$out" sjob
	echo "running spherical $@, waiting on $$sjob" >&2
	python3 analytical.py submit -t "qsub -v GADOPT_CHECKOUT=$(gadopt_checkout) -W depend=beforeany:$$sjob -N analytical_{params} -l storage=gdata/xd2+scratch/xd2,ncpus={cores},walltime=01:00:00,mem={mem}GB,wd,jobfs=10GB -q normal -P $(project) -o pbs_output/l{level}_{params}.out -e pbs_output/l{level}_{params}.err -- ./run_gadi.sh" $@
	wait
