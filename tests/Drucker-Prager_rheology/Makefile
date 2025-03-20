include ../../rules.mk

cases := $(shell python3 test_spiegelman.py)

.PHONY: all clean check

all: $(addprefix spiegelman_,$(cases))

spiegelman_% : category := Drucker-Prager_rheology
spiegelman_% : exec_args = $*
spiegelman_% : desc = $(exec_args)
spiegelman_%: spiegelman.py
	$(run-python)

clean:
	rm -rf __pycache__
	@$(foreach case,$(addprefix spiegelman_,$(cases)),rm -rf $(case);)

check: $(addprefix spiegelman_,$(cases))
	python -m pytest
