include ../../rules.mk

serial_cases := explicit implicit implicit_top_bottom
parallel_cases := implicit_top_bottom_buoyancy implicit_cylindrical

serial_sentinels := $(addprefix .sentinel.,$(serial_cases))
parallel_sentinels := $(addprefix .sentinel.,$(parallel_cases))

.PHONY: all clean check $(serial_cases) $(parallel_cases)

all: $(serial_cases) $(parallel_cases)

#Allow 'make <easy name>' to effectively alias to 'make <long ugly file name>'
$(serial_cases) $(parallel_cases): %: .sentinel.%

$(serial_sentinels) $(parallel_sentinels) : category := tests/free_surface
$(parallel_sentinels) : ncpus := 8

$(serial_sentinels) $(parallel_sentinels): .sentinel.%: %_free_surface.py
	$(run-python)
	@echo "done" > $@

clean:
	rm -rf *.dat .sentinel.* __pycache__

check: $(serial_sentinels) $(parallel_sentinels)
	python3 -m pytest
