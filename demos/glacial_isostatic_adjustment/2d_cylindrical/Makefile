include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := gia
params.log: 2d_cylindrical.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif
	rm -rf output ice mesh viscosity __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
