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
	rm -rf $(addprefix spiegelman_,$(cases)) __pycache__

check: $(addprefix spiegelman_,$(cases))
	python -m pytest
