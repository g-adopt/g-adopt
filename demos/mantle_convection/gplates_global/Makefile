sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
# current directory relative to root makefile
cwd := $(shell realpath --strip --relative-to=../../.. $(CURDIR))

.PHONY: all
all: $(d)/.done

$(d)/.done:
	$(MAKE) -C ../../.. $(cwd)/.done

.PHONY: check
check: $(d)/.done
	python3 -m pytest ../../test_all.py -k $(cwd)

else

TGT_$(d) := $(d)/params.log
SRC_$(d) := $(d)/gplates_global.py
NOTEBOOK_SRC_$(d) := $(d)/gplates_global.py
CLEAN_$(d) := *.pvd *.h5 params.log .done
DIR_CLEAN_$(d) := output

$(d)/.done: $(TGT_$(d))
	@touch $@

$(d)/params.log : category := mantle_convection
$(d)/params.log: $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
