.PHONY: lint longtest longtest_output

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos and tests"
	@python3 -m flake8 demos tests

longtest:
	$(MAKE) -C tests longtest

longtest_output:
	$(MAKE) -C tests longtest_output

check:
	python -m pytest -m 'not longtest'
