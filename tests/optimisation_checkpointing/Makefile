all: checkpointing

checkpointing: helmholtz.py
	echo "running helmholtz checkpointing" >&2
	/usr/bin/time --format="helmholtz checkpointing took %E" tsp -f python3 $<
	echo 'done' > checkpointing

clean:
	rm -rf optimisation_checkpoint __pycache__ *.h5 *.dat checkpointing

check: checkpointing
	python -m pytest