include ../makefile.cases.inc

cases := $(notdir $(gia_cases))

.PHONY: all $(cases) clean check

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach case, $(cases), \
		$(MAKE) -C $(case) $(MAKECMDGOALS); \
	)

check: $(cases)
	python -m pytest $(CURDIR)/../test_all.py -k $(gia_dir)
