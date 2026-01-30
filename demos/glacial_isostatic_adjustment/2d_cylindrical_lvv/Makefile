include ../../../rules.mk

.PHONY: all clean check

all: 2d_cylindrical_lvv.done

2d_cylindrical_lvv.done : category := gia
2d_cylindrical_lvv.done: 2d_cylindrical_lvv.py
	$(run-python)
	@touch $@

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.done *.gif
	rm -rf output ice mesh viscosity __pycache__

check: 2d_cylindrical_lvv.done
	python3 -m pytest $(CURDIR)/../../test_all.py -k $(test_class)/$(current_dir)
