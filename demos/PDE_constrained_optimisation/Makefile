sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
# current directory relative to root makefile
cwd := $(shell realpath --strip --relative-to=.. $(CURDIR))

.PHONY: all
all: $(d)/.done

$(d)/.done:
	$(MAKE) -C .. $(cwd)/.done

.PHONY: check
check: $(d)/.done $(d)/test_pde_constrained_optimisation.py
	python3 -m pytest $(d)/test_pde_constrained_optimisation.py

else

TGT_$(d) := $(d)/.done
SRC_$(d) := $(d)/PDE_constrained_boundary.py $(d)/PDE_constrained_field.py
NOTEBOOK_SRC_$(d) := $(d)/PDE_constrained_boundary.py $(d)/PDE_constrained_field.py
CLEAN_$(d) := $(CASES_$(d)) *.h5
DIR_CLEAN_$(d) := __pycache__

CASES_$(d) := $(d)/functional_boundary.txt $(d)/functional_field.txt

$(d)/.done: $(CASES_$(d))
	@touch $@

$(CASES_$(d)) : category := pde_constrained
$(CASES_$(d)): $(d)/functional_%.txt: $(d)/PDE_constrained_%.py
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
