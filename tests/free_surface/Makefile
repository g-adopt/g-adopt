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

CASES_SERIAL_$(d) := explicit implicit implicit_top_bottom
CASES_PARALLEL_$(d) := implicit_top_bottom_buoyancy implicit_cylindrical
CASES_$(d) := $(CASES_SERIAL_$(d)) $(CASES_PARALLEL_$(d))
TGT_SERIAL_$(d) := $(addprefix $(d)/.sentinel.,$(CASES_SERIAL_$(d)))
TGT_PARALLEL_$(d) := $(addprefix $(d)/.sentinel.,$(CASES_PARALLEL_$(d)))
TGT_$(d) := $(TGT_SERIAL_$(d)) $(TGT_PARALLEL_$(d))
SRC_$(d) := $(addsuffix _free_surface.py,$(CASES_$(d)))
CLEAN_$(d) := *.dat .sentinel.* .done
DIR_CLEAN_$(d) := __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_PARALLEL_$(d)) : ncpus := 8
$(TGT_$(d)) : category := tests/free_surface
$(TGT_$(d)): $(d)/.sentinel.%: $(d)/%_free_surface.py
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
