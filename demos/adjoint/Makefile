cases := damping smoothing Tobs uobs

all: $(addsuffix .conv,$(cases)) checkpointing

ncpus := 2

%.conv: taylor_test.py Checkpoint_State.h5
	echo "running $< on case $*" >&2
	tsp -N $(ncpus) -nf mpiexec -np $(ncpus) python3 $< $*

Checkpoint_State.h5: forward.py
	echo "running forward adjoint case" >&2
	tsp -N $(ncpus) -nf mpiexec -np $(ncpus) python3 $<

checkpointing: helmholtz.py
	echo "running helmholtz checkpointing" >&2
	tsp -N 1 -nf python3 $<

clean:
	rm -f *.h5 *.conv *.dat
	rm -rf optimisation_checkpoint
