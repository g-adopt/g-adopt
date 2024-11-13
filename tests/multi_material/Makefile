case_dir := benchmarks
cases := $(notdir $(wildcard ${case_dir}/*))
ncpus := 8

all: $(cases)

$(cases):
	$(info running $@ multi-material benchmark)
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 run_benchmark.py $@

clean:
	rm -rf __pycache__
	@$(foreach case,$(cases),rm -rf ${case_dir}/$(case)/__pycache__;)
	@$(foreach case,$(cases),rm -rf ${case_dir}/$(case)/outputs;)
	@$(foreach case,$(cases),rm -f ${case_dir}/$(case)/*.pdf;)
	@$(foreach case,$(cases),rm -f ${case_dir}/$(case)/*.npz;)
	@$(foreach case,$(cases),rm -f ${case_dir}/$(case)/mesh/*.msh;)

check: $(cases)
	python3 -m pytest
