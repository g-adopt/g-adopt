include ../makefile.demos.inc

.PHONY: all $(mm_demos) clean check

all: $(mm_demos)

$(mm_demos):
	$(MAKE) -C $@

clean:
	@$(foreach demo,$(mm_demos),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check: $(mm_demos)
	python -m pytest ../test_all.py -k $(mm_dir)
