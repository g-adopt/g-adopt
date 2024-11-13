include ../makefile.demos.inc

.PHONY: all $(mc_demos)

all: $(mc_demos)

$(mc_demos):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach demo,$(mc_demos),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check:
	python -m pytest ../test_all.py -k $(mc_dir)
