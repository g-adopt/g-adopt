.PHONY: lint test longtest longtest_output convert_demos

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos"
	@python3 -m flake8 demos

test:
	$(MAKE) -C demos

longtest:
	$(MAKE) -C demos longtest

longtest_output:
	$(MAKE) -C demos longtest_output

# convert demo Python scripts to executed notebooks
convert_demos:
	$(MAKE) -C demos convert_demos
