include ../../rules.mk

cases := tracy_2d_specified_head tracy_2d_no_flux vauclin_2d
sentinels := $(addprefix .sentinel.,$(cases))

.PHONY: all $(cases) clean check

all: $(cases)

$(cases): %: .sentinel.%

$(sentinels): richards.py tracy_2d.py vauclin_2d.py
	@$(call run_regular,python3 richards.py run $(subst .sentinel.,,$@),richards $(subst .sentinel.,,$@))
	@echo "done" > $@

clean:
	rm -rf *.dat *.pvd *.vtu sentinel.* .sentinel.* __pycache__

check: $(sentinels)
	python -m pytest test_richards.py -v
