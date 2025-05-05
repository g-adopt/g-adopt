include ../../rules.mk

cases := $(wildcard benchmarks/*)
archive := output_0_reference.npz
outputs := $(foreach case,$(cases),$(case)/${archive})

.PHONY: all clean check

all: $(outputs)

%/${archive} : category := multi_material
%/${archive} : ncpus := 4
%/${archive} : exec_cmd = mpiexec -np $(ncpus) python3 run_benchmark.py $(dir $@) --without-plot
%/${archive} : desc = $*
%/${archive}: run_benchmark.py
	$(run-python)

clean:
	rm -rf __pycache__
	@$(foreach case,$(cases),rm -rf $(case)/__pycache__;)
	@$(foreach case,$(cases),rm -rf $(case)/outputs;)
	@$(foreach case,$(cases),rm -f $(case)/*.pdf;)
	@$(foreach case,$(cases),rm -f $(case)/*.npz;)
	@$(foreach case,$(cases),rm -f $(case)/mesh/*.msh;)

check: all
	python3 -m pytest
