sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

$(d)/.done: $(d)/params.log
	@touch $@

$(d)/params.log : category := gia
$(d)/params.log: $(d)/2d_cylindrical.py $(d)/unstructured_annulus.py
	$(run-python)

TGT_$(d) := $(d)/.done
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 *.gif params.log .done
DIR_CLEAN_$(d) := output ice mesh viscosity __pycache__

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

# XXX placeholder for old check routine
ifdef undefined
.PHONY: check
check: $(d)/.done
	python3 -m pytest ../../test_all.py -k mantle_convection/base_case
endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
