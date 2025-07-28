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

CASES_$(d) := full_optimisation_np1 full_optimisation_np2
TGT_$(d) := $(CASES_$(d):%=$(d)/%.dat)
SRC_$(d) := $(d)/helmholtz.py
CLEAN_$(d) := *.dat *.h5 .done
DIR_CLEAN_$(d) := optimisation_checkpoint_* __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)): category := optimisation_checkpointing
$(TGT_$(d)): ncpus = $*
$(TGT_$(d)): $(d)/full_optimisation_np%.dat: $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
