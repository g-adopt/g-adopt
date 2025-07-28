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

CASES_$(d) := $(shell python3 $(d)/test_scalar_advection_diffusion_DH27.py)
TGT_$(d) := $(addprefix $(d)/.sentinel.,$(CASES_$(d)))
SRC_$(d) := $(d)/scalar_advection_diffusion_DH27.py
CLEAN_$(d) := *.dat *.log .sentinel.*
DIR_CLEAN_$(d) := __pycache__

$(TGT_$(d)) : category := scalar_advection_diffusion_DH27
$(TGT_$(d)) : exec_args = $(patsubst .sentinel.%,%,$(notdir $@))
$(TGT_$(d)) : desc = $(exec_args)
$(TGT_$(d)): $(d)/scalar_advection_diffusion_DH27.py
	$(run-python)

TGT_$(d) += $(d)/integrated_q.log $(d)/integrated_q_DH219.log
SRC_$(d) += $(d)/scalar_advection_diffusion.py $(d)/scalar_advection_diffusion_DH219_skew.py

$(d)/integrated_q.log : category := tests
$(d)/integrated_q.log: $(d)/scalar_advection_diffusion.py
	$(run-python)

$(d)/integrated_q_DH219.log : category := tests
$(d)/integrated_q_DH219.log: $(d)/scalar_advection_diffusion_DH219_skew.py
	$(run-python)


$(d)/.done: $(TGT_$(d))
	@touch $@

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
