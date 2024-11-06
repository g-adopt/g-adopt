cases := $(shell python3 test_spiegelman.py)

all: $(cases)

$(cases): spiegelman.py
	echo "running spiegelman $@" >&2
	/usr/bin/time --format="spiegelman $@ finished in %E" tsp -f python3 $< $@

clean:
	rm -rf $(addprefix spiegelman_,$(cases)) __pycache__

check: $(addprefix spiegelman_,$(cases))
	python -m pytest