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

TGT_$(d) := final_error.log
SRC_$(d) := $(d)/scalar_advection.py
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 final_error.log .done
DIR_CLEAN_$(d) := output

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)): category := tests
$(TGT_$(d)): $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
