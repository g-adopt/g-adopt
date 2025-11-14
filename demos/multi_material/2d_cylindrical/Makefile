include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := multi_material
params.log : ncpus := 4
params.log: 2d_cylindrical.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf level_set mesh output RaB temp

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
