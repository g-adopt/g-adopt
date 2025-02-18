cases := damping smoothing Tobs uobs

all: $(addsuffix .conv,$(cases))

ncpus := 104

%.conv: inverse.py Checkpoint_State.h5
	echo "running $< on case $*" >&2
	mpiexec -np $(ncpus) python3 $< $*

Checkpoint_State.h5: forward.py
	echo "running forward adjoint case" >&2
	mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.h5 *.conv
