sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all: $(d)/.done

$(d)/.done:
	$(MAKE) -C .. $(cwd)/.done

.PHONY: check
check: $(d)/.done
	python3 -m pytest

else

TGT_$(d) := $(d)/final_error.log $(d)/final_error_adaptive.log
SRC_$(d) := $(d)/scalar_advection.py $(d)/scalar_advection_adaptive.py
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 final_error.log .done
DIR_CLEAN_$(d) := output

$(d)/.done: $(TGT_$(d))
	@touch $@

$(d)/final_error.log: category := tests
$(d)/final_error.log: $(d)/scalar_advection.py
	$(run-python)

$(d)/final_error_adaptive.log: category := tests
$(d)/final_error_adaptive.log: $(d)/scalar_advection_adaptive.py
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
