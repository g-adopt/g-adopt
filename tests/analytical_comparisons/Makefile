include ../../rules.mk

cases := smooth_cylindrical_freeslip smooth_cylindrical_zeroslip delta_cylindrical_freeslip delta_cylindrical_zeroslip delta_cylindrical_freeslip_dpc delta_cylindrical_zeroslip_dpc
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
	echo "running spherical $@" >&2
	mkdir -p batch_output
	python3 analytical.py submit -H -o batch_output/$(subst /,-,$@)_l{level}_{params}.out -e batch_output/$(subst /,-,$@)_l{level}_{params}.err $@

clean:
	rm -rf batch_output *.dat sentinel.* .sentinel.* __pycache__

longtest_output:
	[[ -d batch_output ]] && tail -n +1 batch_output/*

check: $(sentinels)
	python -m pytest -m 'not longtest'

longcheck:
	python -m pytest -m longtest
