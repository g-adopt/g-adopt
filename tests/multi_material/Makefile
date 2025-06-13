include ../../rules.mk

cases := $(wildcard benchmarks/*)

.PHONY: all clean check

all: $(cases)

$(cases) : ncpus := 8
$(cases) : category := multi_material
$(cases) : exec_cmd = mpiexec -np $(ncpus) python3 $< $*
$(cases): desc = $*
$(cases): %: run_benchmark.py %/simulation.py
	$(run-python)

clean:
	rm -rf __pycache__
	@$(foreach case,$(cases),rm -rf $(case)/__pycache__;)
	@$(foreach case,$(cases),rm -rf $(case)/outputs;)
	@$(foreach case,$(cases),rm -f $(case)/*.pdf;)
	@$(foreach case,$(cases),rm -f $(case)/*.npz;)
	@$(foreach case,$(cases),rm -f $(case)/mesh/*.msh;)

check: $(cases)
	python3 -m pytest
