all: checkpointing

checkpointing: helmholtz.py
	echo "running helmholtz checkpointing" >&2
	tsp -N 1 -nf python3 $<

clean:
	rm -rf optimisation_checkpoint
