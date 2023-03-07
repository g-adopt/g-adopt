all: params.log

params.log: base_case.py
	echo "running $<" >&2
	tsp -fn python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
