include ../../../rules.mk

all: params.log

params.log : category := mantle_convection
params.log : ncpus := 4
params.log: run-3d_cartesian

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output reference_state

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
