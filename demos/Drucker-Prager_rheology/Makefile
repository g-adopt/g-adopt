cases := $(shell python3 test_spiegelman.py)

all: $(cases)

$(cases): spiegelman.py
	echo "running spiegelman $@" >&2
	tsp -fn python3 $< $@

clean:
	rm -rf $(addprefix spiegelman_,$(cases))
