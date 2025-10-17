include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := mantle_convection
params.log: dynamic_topography.py
	$(MAKE) -C ../base_case
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
