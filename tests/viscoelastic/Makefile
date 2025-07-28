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

CASES_$(d) := elastic viscoelastic viscous
TGT_$(d) := $(CASES_$(d):%=$(d)/errors-%-zhong-free-surface.dat)
SRC_$(d) := $(d)/zhong_viscoelastic_free_surface.py
CLEAN_$(d) := *.dat
DIR_CLEAN_$(d) := __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)) : category := tests/viscoelastic
$(TGT_$(d)) : exec_args = --case $*
$(TGT_$(d)) : desc = $*
$(TGT_$(d)): $(d)/errors-%-zhong-free-surface.dat: $(SRC_$(d))
	$(run-python)

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
