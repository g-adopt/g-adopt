cases := elastic viscoelastic viscous
output_files := $(foreach case,$(cases),errors-$(case)-zhong-free-surface.dat)

all: $(output_files)

errors-%-zhong-free-surface.dat: zhong_viscoelastic_free_surface.py
	echo "running $< on $*" >&1
	/usr/bin/time --format="$< $* finished in %E" tsp -f python3 $< --case $*

clean:
	rm -rf errors*.dat __pycache__

check: $(output_files)
	python3 -m pytest
