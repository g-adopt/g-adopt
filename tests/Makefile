long_cases := analytical_comparisons parallel_scaling adjoint_2d_cylindrical

.PHONY: all longtest longtest_output $(long_cases) check longcheck

# don't run anything by default
all:
	@echo "specify a target"

longtest: $(long_cases)

longtest_output: $(long_cases)

$(long_cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

check:
	python -m pytest . ../demos -m 'not demo and not longtest'

longcheck:
	python -m pytest -m longtest
