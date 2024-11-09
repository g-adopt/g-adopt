include ../makefile.demos.inc

.PHONY: all $(gia_demos)

all: $(gia_demos)

$(gia_demos):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean:
	@$(foreach demo,$(gia_demos),$(MAKE) -C $(demo) $(MAKECMDGOALS);)

check:
	python -m pytest ../test_all.py -k $(gia_dir)