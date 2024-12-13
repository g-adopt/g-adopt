test_name := $(notdir $(CURDIR))

.PHONY: all clean

all: final_error.log

final_error.log: $(test_name).py
	$(info running $<)
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log
	rm -rf output CG* __pycache__

check: final_error.log
	python3 -m pytest
