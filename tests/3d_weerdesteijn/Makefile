include ../../rules.mk

.PHONY: all clean check

all: params.log

weerdesteijn_box_refined_surface_%km_nondim.msh: weerdesteijn_box_refined_surface_nondim.geo
	gmsh -2 $< -setnumber refined_dx $* -o $@

params.log : category := tests
params.log : ncpus := 8
params.log : exec_args = --refined_surface --bulk_shear_ratio 1.94 --dt_years 5000 --DG0_layers 2
params.log: 3d_weerdesteijn.py weerdesteijn_box_refined_surface_20km_nondim.msh 
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 *.msh *.dat params.log
	rm -rf output reference_state __pycache__

check: params.log
	python3 -m pytest $(CURDIR)/../../demos/test_all.py -k "../tests/3d_weerdesteijn"
