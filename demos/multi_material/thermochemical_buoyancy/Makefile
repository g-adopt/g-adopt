all: params.log

params.log: thermochemical_buoyancy.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -L multi_material -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output