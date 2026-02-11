include ../makefile.cases.inc

cases := $(notdir $(gia_cases))

.PHONY: all $(cases) clean check

all: $(cases)

# Ensure 2d_cylindrical_lvv completes before adjoint_ice starts
adjoint_ice: | 2d_cylindrical_lvv

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach case, $(cases), \
		$(MAKE) -C $(case) $(MAKECMDGOALS); \
	)

check: $(cases)
	python -m pytest $(CURDIR)/../test_all.py -k $(gia_dir)
