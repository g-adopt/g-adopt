include ../../rules.mk

cases := delta_cylindrical_zeroslip
long_cases := smooth_cylindrical_freesurface smooth_spherical_freeslip smooth_spherical_zeroslip

sentinels := $(addprefix .sentinel.,$(cases))

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean check longcheck

all: $(cases)

longtest: $(long_cases)

$(cases): %: .sentinel.%

tsp_string = $(if $(BATCH_MODE),tsp -N {cores} -f)
$(sentinels): analytical.py
	@$(call run_regular,python3 analytical.py submit -t "$(tsp_string) mpiexec -np {cores}" $(subst .sentinel.,,$@),analytical $(subst .sentinel.,,$@))
	@echo "done" > $@

.ONESHELL:
SHELL = /bin/bash
$(long_cases): analytical.py clean
	exec {out}<> <(:) # create new named FD
	qsub -W depend=on:$$(python3 analytical.py count $@) -N sentinel -W block=true -l ncpus=1,walltime=00:00:30,wd -q normal -P $(project) -- /bin/true >&"$$out" &
	read -u "$$out" sjob
	echo "running spherical $@, waiting on $$sjob" >&2
	mkdir -p pbs_output
	python3 analytical.py submit -t "qsub -v GADOPT_CHECKOUT=$(gadopt_checkout),GADOPT_SETUP=$(gadopt_setup) -W depend=beforeany:$$sjob -N analytical_{params} -l storage=gdata/xd2+scratch/xd2+gdata/fp50,ncpus={cores},walltime=01:00:00,mem={mem}GB,wd,jobfs=10GB -q normal -P $(project) -o pbs_output/$(subst /,-,$@)_l{level}_{params}.out -e pbs_output/$(subst /,-,$@)_l{level}_{params}.err -- ./run_gadi.sh" $@
	wait

clean:
	rm -rf pbs_output *.dat sentinel.* .sentinel.* __pycache__

longtest_output:
	[[ -d pbs_output ]] && tail -n +1 pbs_output/*

check: $(sentinels)
	python -m pytest -m 'not longtest'

longcheck:
	python -m pytest -m longtest
