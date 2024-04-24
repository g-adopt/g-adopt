cases := crameri_2012 gerya_2003 robey_2019 schmalholz_2011 schmeling_2008 trim_2023

ncpus := 8

all: $(cases)

$(cases):
	echo "running $@ multi-material benchmark"
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 run_benchmark.py benchmarks.$@
