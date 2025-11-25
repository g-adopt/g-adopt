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
	rm -f *.pvd *.vtu *.pvtu *.h5 final_error.log final_error_adaptive.log
	rm -f num_steps_adaptive.log dt_stats_adaptive.log
	rm -rf output

check: final_error.log final_error_adaptive.log num_steps_adaptive.log dt_stats_adaptive.log
	python3 -m pytest
