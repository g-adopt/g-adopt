.PHONY: lint test

lint:
	@echo "Linting Davies et. al"
	@python3 -m flake8 Davies_etal_GMD_2021
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos"
	@python3 -m flake8 demos

test:
	$(MAKE) -C demos
