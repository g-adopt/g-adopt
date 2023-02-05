all: params.log

params.log: 2d_cylindrical.py
	echo "running $<" >&2
	mpiexec -np 4 python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
