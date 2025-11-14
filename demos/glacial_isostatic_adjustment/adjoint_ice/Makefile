include ../../../rules.mk

.PHONY: all clean check

all: functional.txt

functional.txt : category := gia
functional.txt: adjoint_ice.py forward-2d-cylindrical-disp-vel.h5 2d_cylindrical_lvv.py
	$(run-python)

forward-2d-cylindrical-disp-vel.h5 : category := gia
forward-2d-cylindrical-disp-vel.h5 : ncpus := 1
forward-2d-cylindrical-disp-vel.h5: 2d_cylindrical_lvv.py
	$(run-python)

2d_cylindrical_lvv.py:
	cp ${CURDIR}/../2d_cylindrical_lvv/2d_cylindrical_lvv.py .

clean:
	rm -rf functional.txt taylor_test_minconv.txt 2d_cylindrical_lvv.py *.h5 *.pvd solutions solution visualisation_vtk optimisation_checkpoint

check: functional.txt
	python3 -m pytest
