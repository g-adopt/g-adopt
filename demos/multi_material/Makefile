cases := gerya_2003 robey_2019

ncpus := 8

all: $(cases)

$(cases):
	echo "running $@ multi-material benchmark"
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 run_benchmark.py benchmarks.$@
