include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := multi_material
params.log : ncpus := 4
params.log: thermochemical_buoyancy.py
	$(run-python)

clean:
	rm -f *.msh *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
