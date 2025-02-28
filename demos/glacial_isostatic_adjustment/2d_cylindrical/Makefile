include ../../../rules.mk

all: params.log

params.log : category := gia
params.log: run-2d_cylindrical

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif
	rm -rf output density ice mesh viscosity __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
