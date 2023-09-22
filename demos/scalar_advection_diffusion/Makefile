cases := $(shell python3 test_scalar_advection_diffusion_DH27.py)

all: $(cases)

$(cases): scalar_advection_diffusion_DH27.py
	echo "running advection diffusion $@" >&2
	tsp -N 1 -nf python3 $< $@

