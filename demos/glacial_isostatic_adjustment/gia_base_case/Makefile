all: params.log

params.log: gia_base_case.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log *.gif
