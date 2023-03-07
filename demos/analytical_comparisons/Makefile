cases := smooth/cylindrical/free_slip

.PHONY: all $(cases)

all: $(cases)

$(cases): analytical.py
	echo "running analytical $@" >&2
	python3 analytical.py submit -t "tsp -N {cores} -f mpiexec -np {cores}" $@
