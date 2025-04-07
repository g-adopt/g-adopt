levels := 5 6 7
levels := $(addsuffix .txt,$(addprefix profile_,$(levels)))

.PHONY: longtest clean check

longtest: $(levels)

.ONESHELL:
SHELL = /bin/bash
profile_%.txt: scaling.py stokes_cubed_sphere.py
	echo "running parallel scaling on level $*"
	mkdir -p batch_output
	python3 scaling.py submit -H $*

clean:
	rm -rf batch_output profile_*.txt level_*.out level_*.err

longtest_output:
	tail -n +1 level_*.{err,out}

check: $(levels)
	python3 -m pytest