all: params.log

params.log: free_surface.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 *.gif params.log
