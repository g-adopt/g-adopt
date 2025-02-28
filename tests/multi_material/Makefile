include ../../rules.mk

cases := crameri_2012 gerya_2003 robey_2019 schmalholz_2011 schmeling_2008 tosi_2015 trim_2023 van_keken_1997_isothermal van_keken_1997_thermochemical

all: $(cases)

$(cases): ncpus := 8
$(cases): category := multi_material
$(cases): exec_cmd = mpiexec -np $(ncpus) python3 $< benchmarks.$*

$(cases): %: run_benchmark.py benchmarks/%.py
	@$(call run_cmd,$(exec_wrapper),$*)

clean:
	@$(foreach case, $(cases), \
		rm -rf $(case) ; \
	)
	rm -rf __pycache__ benchmarks/*.msh benchmarks/__pycache__

check: $(cases)
	python3 -m pytest
