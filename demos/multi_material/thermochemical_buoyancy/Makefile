include ../../../rules.mk

all: params.log

params.log : category := multi_material
params.log: run-thermochemical_buoyancy

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
