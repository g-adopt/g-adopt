include ../../rules.mk

cases := viscoelastic-compressible viscoelastic-compressible-burgers elastic-compressible
output_files := $(foreach case,$(cases),errors-$(case)-free-surface.dat)

.PHONY: all clean check

all: $(output_files)

errors-%-free-surface.dat : category := tests/gia
errors-%-free-surface.dat : exec_args = --case $*
errors-%-free-surface.dat : desc = $*
errors-%-free-surface.dat: compressible_viscoelastic_free_surface_coupled_nondimensional.py
	$(run-python)
	
clean:
	rm -rf errors*.dat *.csv __pycache__

check: $(output_files)
	python3 -m pytest
