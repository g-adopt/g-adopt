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
	python3 -m pytest ../../demos/test_all.py -k $(shell realpath --strip --relative-to=../../demos $(CURDIR))

else

TGT_$(d) := $(d)/params.log
SRC_$(d) := $(d)/2d_cylindrical_TALA_DG.py
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 params.log .done
DIR_CLEAN_$(d) := output reference_state __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)): category := tests
$(TGT_$(d)): ncpus := 16
$(TGT_$(d)): $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
