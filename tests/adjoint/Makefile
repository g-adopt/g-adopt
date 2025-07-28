sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
cwd := $(shell realpath --strip --relative-to=../.. $(CURDIR))

.PHONY: all
all: $(d)/.done

$(d)/.done:
	$(MAKE) -C ../.. $(cwd)/.done

.PHONY: check
check: $(d)/.done
	python3 -m pytest

else

CASES_COMPONENTS_$(d) := damping smoothing Tobs uobs uimposed
CASES_SCHEDULERS_$(d) := noscheduler fullmemory fullstorage
CASES_$(d) := $(foreach case,$(CASES_COMPONENTS_$(d)),$\
	$(foreach scheduler,$(CASES_SCHEDULERS_$(d)),$\
		$(case)_$(scheduler)))
TGT_$(d) := $(CASES_$(d):%=$(d)/%.conv)
SRC_$(d) := $(d)/taylor_test.py
CLEAN_$(d) := *.h5 *.conv *.dat
DIR_CLEAN_$(d) := __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(d)/adjoint-demo-checkpoint-state.h5: demos/mantle_convection/adjoint/adjoint-demo-checkpoint-state.h5
	@ln -sf $< $@

$(TGT_$(d)) : category := tests/adjoint
$(TGT_$(d)) : ncpus := 2
$(TGT_$(d)) : exec_args = $*
$(TGT_$(d)) : desc = $(exec_args)
$(TGT_$(d)): $(d)/%.conv: $(SRC_$(d)) $(d)/adjoint-demo-checkpoint-state.h5
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
