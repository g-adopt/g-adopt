.PHONY: lint

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos and tests"
	@python3 -m flake8 demos tests

check:
	python -m pytest -m 'not longtest'
