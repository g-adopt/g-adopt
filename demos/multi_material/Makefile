include ../makefile.demos.inc

.PHONY: all $(mm_demos)

all: $(mm_demos)

$(mm_demos):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach demo,$(mm_demos),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check:
	python -m pytest ../test_all.py -k $(mm_dir)
