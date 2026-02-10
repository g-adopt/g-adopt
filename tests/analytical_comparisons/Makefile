long_cases := smooth_cylindrical_freesurface smooth_spherical_freeslip smooth_spherical_zeroslip

.PHONY: longtest longtest_output $(long_cases) clean longcheck

longtest: $(long_cases)

.ONESHELL:
SHELL = /bin/bash
$(long_cases): analytical.py clean
	echo "running spherical $@" >&2
	mkdir -p batch_output
	python3 analytical.py submit $@ -H -o batch_output/$(subst /,-,$@)_l{level}_{params}.out -e batch_output/$(subst /,-,$@)_l{level}_{params}.err

clean:
	rm -rf batch_output *.dat sentinel.* .sentinel.* __pycache__

longtest_output:
	[[ -d batch_output ]] && tail -n +1 batch_output/*

longcheck:
	python -m pytest -m longtest
