include ../../rules.mk

cases := elastic viscoelastic viscous
output_files := $(foreach case,$(cases),errors-$(case)-zhong-free-surface.dat)

all: $(output_files)

errors-%-zhong-free-surface.dat: category := tests/viscoelastic
errors-%-zhong-free-surface.dat: exec_args = --case $*
errors-%-zhong-free-surface.dat: desc = $*

errors-%-zhong-free-surface.dat: zhong_viscoelastic_free_surface.py
	$(run-python)
	
clean:
	rm -rf errors*.dat __pycache__

check: $(output_files)
	python3 -m pytest
