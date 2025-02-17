all: functional.txt taylor_test_minconv.txt

ncpus := 1

functional.txt: adjoint_ice.py forward-2d-cylindrical-disp-incdisp.h5
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -L gia -f python3 $<

taylor_test_minconv.txt: adjoint_ice.py forward-2d-cylindrical-disp-incdisp.h5
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -L gia -f python3 $<

forward-2d-cylindrical-disp-incdisp.h5: ../2d_cylindrical/2d_cylindrical.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -L gia -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<
	cp ../2d_cylindrical/forward-2d-cylindrical-disp-incdisp.h5 .

clean:
	rm -rf functional.txt taylor_test_minconv.txt *.h5 *.pvd adj_ice ice viscosity output updated_ice_thickness updated_out optimisation_checkpoint

check: functional.txt taylor_test_minconv.txt
	python3 -m pytest
