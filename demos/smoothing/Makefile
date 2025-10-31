include ../../rules.mk

.PHONY: all clean check

all: params.log

params.log : category := smoothing
params.log: smoothing.py smoothing-example.h5
	$(run-python)

smoothing-example.h5:
	@echo "Copying checkpoint file from tests directory..."
	@[ ! -f smoothing-example.h5 ] && cp ../../tests/adjoint_2d_cylindrical/Checkpoint230.h5 smoothing-example.h5 || true

clean:
	rm -f *.pvd *.vtu *.pvtu params.log
	rm -rf output original_temperature

check: params.log
	python3 -m pytest $(CURDIR)/../test_all.py -k $(test_class)/$(current_dir)
