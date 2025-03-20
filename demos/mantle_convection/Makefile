include ../makefile.demos.inc

.PHONY: all $(mc_demos) clean check

all: $(mc_demos)

$(mc_demos):
	$(MAKE) -C $@

clean:
	@$(foreach demo,$(mc_demos),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check: $(mc_demos)
	python -m pytest ../test_all.py -k $(mc_dir)
