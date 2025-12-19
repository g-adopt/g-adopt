# Welcome to the G-ADOPT Makefile!
#
# This top-level file drives all the automation in the repository. We
# have, of course, the famous "lint" target that checks for some
# degree of code style consistency. The main show is the ability to
# run the tests and demos, and convert the demos to notebooks.
#
# For an explanation of how the individual cases are run, check out
# the .rules.mk file: this dispatches between different methods
# depending on whether we're using a "batch mode" and also the number
# of cores required.

# The following tasks are not linked to any real files and are thus
# always considered "out of date" and executed when requested.
.PHONY: lint all-tests clean check

lint:
	@echo "Linting module code"
	@python3 -m flake8 gadopt
	@echo "Linting demos and tests"
	@python3 -m flake8 demos tests

# Here begins the fun of the testing system. In order to give Make a
# global view of the project, and to allow for dependencies between
# different cases, we're using a non-recursive structure. However, we
# don't want to encode all of our logic within the one Makefile! Each
# Makefile is responsible for including all of its children, but also
# keeping track of *where* that child is with respect to the top-level
# directory of the project.
#
# For example, a test would need to define a target like
# "tests/example/.done", but we keep a directory stack so that test
# only has to refer to "$(d)/.done". Look into the following Makefiles
# for a bit more information on how this mechanism is implemented.
include .rules.mk

dir := demos
include $(dir)/Makefile

dir := tests
include $(dir)/Makefile

# At this point, all the file targets within the project are
# defined. We can just define a few more meta-targets: one to run all
# the test cases.
all-tests: demos tests

# ...one to run the cases, *and* run pytest (which will include the unit tests)
check: all-tests
	python3 -m pytest -m "not longtest"

# ...and one to clean up *all* test artifacts.
clean:
	rm -f $(CLEAN)
	rm -rf $(DIR_CLEAN)
