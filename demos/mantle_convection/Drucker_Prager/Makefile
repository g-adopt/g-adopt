include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := mantle_convection
params.log: Drucker_Prager.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
