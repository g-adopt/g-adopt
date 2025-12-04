include ../../rules.mk

.PHONY: all clean check

all: final_error.log final_error_adaptive.log num_steps_adaptive.log dt_stats_adaptive.log

final_error.log : category := tests
final_error.log: scalar_advection.py
	$(run-python)

final_error_adaptive.log num_steps_adaptive.log dt_stats_adaptive.log : category := tests
final_error_adaptive.log num_steps_adaptive.log dt_stats_adaptive.log: scalar_advection_adaptive.py
	$(run-python)

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 *.log
	rm -f *.log
	rm -rf output

check: all
	python3 -m pytest
