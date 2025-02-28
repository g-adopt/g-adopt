include ../../rules.mk

cases := $(shell python3 test_spiegelman.py)

all: $(cases)

$(cases): category := Drucker-Prager_rheology
$(cases): exec_args = $@
$(cases): desc = $(exec_args)
$(cases): spiegelman.py
	$(run-python)

clean:
	rm -rf $(addprefix spiegelman_,$(cases)) __pycache__

check: $(addprefix spiegelman_,$(cases))
	python -m pytest
