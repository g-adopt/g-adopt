.PHONY: lint test

lint:
	@echo "Linting Davies et. al"
	@python3 -m flake8 Davies_etal_GMD_2021

test:
	$(MAKE) -C demos
