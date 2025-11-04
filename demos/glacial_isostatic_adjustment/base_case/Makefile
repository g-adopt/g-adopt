include ../../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := gia
params.log: base_case.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif
	rm -rf output

check: params.log
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
