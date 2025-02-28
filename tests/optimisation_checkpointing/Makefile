include ../../rules.mk

cases := full_optimisation_np1.dat full_optimisation_np2.dat

all: $(cases)

full_optimisation_np%.dat: category := optimisation_checkpointing
full_optimisation_np%.dat: ncpus = $*

$(cases): full_optimisation_np%.dat: helmholtz.py
	$(run-python)

clean:
	rm -rf optimisation_checkpoint_* __pycache__ *.h5 *.dat

check: $(cases)
	python -m pytest
