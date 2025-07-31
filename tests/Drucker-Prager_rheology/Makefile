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

CASES_$(d) := $(addprefix spiegelman_,$(shell python3 $(d)/test_spiegelman.py))
TGT_$(d) := $(CASES_$(d):%=$(d)/%/newton.txt)
SRC_$(d) := $(d)/spiegelman.py
CLEAN_$(d) := .done
DIR_CLEAN_$(d) := __pycache__ $(CASES_$(d))

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)) : category := Drucker-Prager_rheology
$(TGT_$(d)) : exec_args = $*
$(TGT_$(d)) : desc = $(exec_args)
$(TGT_$(d)): $(d)/spiegelman_%/newton.txt: $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
