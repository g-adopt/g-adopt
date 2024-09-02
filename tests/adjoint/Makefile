cases := damping smoothing Tobs uobs

all: $(addsuffix .conv,$(cases))

ncpus := 2

%.conv: taylor_test.py adjoint-demo-checkpoint-state.h5
	echo "running $< on case $*" >&2
	/usr/bin/time --format="$< on case $* took %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $< $*

adjoint-demo-checkpoint-state.h5: ../../demos/adjoint/adjoint_forward.py
	echo "running forward adjoint case" >&2
	/usr/bin/time --format="forward adjoint case took %E" tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.h5 *.conv *.dat
