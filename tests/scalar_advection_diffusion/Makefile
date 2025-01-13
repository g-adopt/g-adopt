cases := $(shell python3 test_scalar_advection_diffusion_DH27.py)

sentinels := $(addprefix .sentinel.,$(cases))

.PHONY: all $(cases) clean

all: $(cases) integrated_q.log integrated_q_DH219.log

$(cases): %: .sentinel.%

$(sentinels): scalar_advection_diffusion_DH27.py
	echo "running advection diffusion $(subst .sentinel.,,$@)" >&2
	tsp -N 1 -f python3 $< $(subst .sentinel.,,$@)
	echo 'done' > $@

integrated_q.log: scalar_advection_diffusion.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

integrated_q_DH219.log: scalar_advection_diffusion_DH219_skew.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.dat .sentinel.* *.log
	rm -rf __pycache__

check: $(sentinels)
	python3 -m pytest
