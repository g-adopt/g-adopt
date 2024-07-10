all: checkpointing

checkpointing: helmholtz.py
	echo "running helmholtz checkpointing" >&2
	/usr/bin/time --format="helmholtz checkpointing took %E" tsp -f python3 $<

clean:
	rm -rf optimisation_checkpoint
