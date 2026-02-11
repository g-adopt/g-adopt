include ../../../rules.mk

.PHONY: all clean check

all: functional.txt

functional.txt : category := gia
functional.txt: adjoint_ice.py forward-2d-cylindrical-disp-vel.h5
	$(run-python)

forward-2d-cylindrical-disp-vel.h5: ../2d_cylindrical_lvv/2d_cylindrical_lvv.done
	ln -s ../2d_cylindrical_lvv/forward-2d-cylindrical-disp-vel.h5

../2d_cylindrical_lvv/2d_cylindrical_lvv.done:
	$(MAKE) -C ../2d_cylindrical_lvv

clean:
	rm -rf functional.txt taylor_test_minconv.txt *.h5 *.pvd adj_ice ice optimisation_checkpoint output updated* viscosity

check: functional.txt
	python3 -m pytest
