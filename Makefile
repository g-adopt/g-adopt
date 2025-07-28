.PHONY: lint all-tests demos clean check

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos and tests"
	@python3 -m flake8 demos tests

include .rules.mk

dir := demos
include $(dir)/Makefile

# dir := tests
# include $(dir)/Makefile

all-tests: demos tests

check: all-tests
	python3 -m pytest -m "not longtest"

clean:
	rm -f $(CLEAN)
	rm -rf $(DIR_CLEAN)
