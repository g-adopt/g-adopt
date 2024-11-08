include ../makefile.cases.inc
cases := $(notdir $(gia_cases))

.PHONY: all $(cases)

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach case, $(cases), \
		$(MAKE) -C $(case) $(MAKECMDGOALS); \
	)

check:
	python -m pytest ../test_all.py -k $(gia_dir)