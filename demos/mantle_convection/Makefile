include ../makefile.cases.inc

cases := $(notdir $(mc_cases))

.PHONY: all $(cases) clean check

all: $(cases)

# dynamic topography must come after base case
dynamic_topography: .WAIT base_case

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach case, $(cases), \
		$(MAKE) -C $(case) $(MAKECMDGOALS); \
	)

check: $(cases)
	python -m pytest $(CURDIR)/../test_all.py -k $(mc_dir)
