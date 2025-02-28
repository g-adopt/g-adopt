include ../../rules.mk

cases := $(shell python3 test_scalar_advection_diffusion_DH27.py)

sentinels := $(addprefix .sentinel.,$(cases))

.PHONY: all $(cases) clean

all: $(cases) integrated_q.log integrated_q_DH219.log

$(cases): %: .sentinel.%

$(sentinels): category := scalar_advection_diffusion_DH27
$(sentinels): exec_args = $(subst .sentinel.,,$@)
$(sentinels): desc = $(exec_args)
$(sentinels): scalar_advection_diffusion_DH27.py
	$(run-python)

integrated_q.log: category := tests
integrated_q.log: run-scalar_advection_diffusion

integrated_q_DH219.log: category := tests
integrated_q_DH219.log: run-scalar_advection_diffusion_DH219_skew

clean:
	rm -f *.dat .sentinel.* *.log
	rm -rf __pycache__

check: $(sentinels)
	python3 -m pytest
