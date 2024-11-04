cases := $(shell python3 test_scalar_advection_diffusion_DH27.py)

sentinels := $(addprefix .sentinel.,$(cases))

.PHONY: all $(cases) clean

all: $(cases)

$(cases): %: .sentinel.%

$(sentinels): scalar_advection_diffusion_DH27.py
	echo "running advection diffusion $(subst .sentinel.,,$@)" >&2
	tsp -N 1 -f python3 $< $(subst .sentinel.,,$@)
	echo 'done' > $@

clean:
	rm -f *.dat .sentinel.*
	rm -rf __pycache__

check: $(sentinels)
	python3 -m pytest