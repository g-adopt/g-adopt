cases := damping smoothing Tobs uobs

all: $(addsuffix .conv,$(cases))

ncpus := 4

%.conv: taylor_test.py Checkpoint_State.h5
	echo "running $< on case $*" >&2
	tsp -L mantle_convection -N $(ncpus) -f mpiexec -np $(ncpus) python3 $< $*

Checkpoint_State.h5: forward.py
	echo "running forward adjoint case" >&2
	tsp -L mantle_convection -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -rf *.h5 *.conv vtu-files
