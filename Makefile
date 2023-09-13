.PHONY: lint test

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos"
	@python3 -m flake8 demos

test:
	$(MAKE) -C demos
