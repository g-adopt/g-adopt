serial_cases := explicit implicit implicit_top_bottom
parallel_cases := implicit_top_bottom_buoyancy implicit_cylindrical

ncpus := 8

serial_sentinels := $(addprefix .sentinel.,$(serial_cases))
parallel_sentinels := $(addprefix .sentinel.,$(parallel_cases))

.PHONY: all clean check $(serial_cases) $(parallel_cases)

all: $(serial_cases) $(parallel_cases)

#Allow 'make <easy name>' to effectively alias to 'make <long ugly file name>'
$(serial_cases) $(parallel_cases): %: .sentinel.%

$(serial_sentinels):
	echo "running $(subst .sentinel.,,$@) free surface coupling "
	tsp -N 1 -f python3 $(subst .sentinel.,,$@)_free_surface.py
	echo "done" > $@

$(parallel_sentinels):
	echo "running $(subst .sentinel.,,$@) free surface coupling "
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 $(subst .sentinel.,,$@)_free_surface.py
	echo "done" > $@

clean:
	rm -rf *.dat .sentinel.* __pycache__

check: $(serial_sentinels) $(parallel_sentinels)
	python3 -m pytest
