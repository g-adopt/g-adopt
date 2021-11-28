all: params.log

params.log: compressible_case.py
	mpiexec -np 4 python $<

clean:
	rm -f *.vtu *.pvtu *.h5 params.log
