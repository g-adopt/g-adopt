cases := smooth/cylindrical/free_slip smooth/cylindrical/zero_slip delta/cylindrical/free_slip delta/cylindrical/zero_slip delta/cylindrical/free_slip_dpc delta/cylindrical/zero_slip_dpc

.PHONY: all $(cases)

all: $(cases)

$(cases): analytical.py
	echo "running analytical $@" >&2
	python3 analytical.py submit -t "tsp -N {cores} -f mpiexec -np {cores}" $@
