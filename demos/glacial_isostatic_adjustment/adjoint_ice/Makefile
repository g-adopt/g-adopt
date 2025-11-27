include ../../../rules.mk

.PHONY: all clean check

all: functional.txt

functional.txt : category := gia
functional.txt: adjoint_ice.py forward-2d-cylindrical-disp-vel.h5
	$(run-python)

forward-2d-cylindrical-disp-vel.h5: ../2d_cylindrical_lvv/2d_cylindrical_lvv.py
	$(MAKE) -C ../2d_cylindrical_lvv
	cp ../2d_cylindrical_lvv/forward-2d-cylindrical-disp-vel.h5 .

clean:
	rm -rf functional.txt taylor_test_minconv.txt *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint

check: functional.txt
	python3 -m pytest
