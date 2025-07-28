sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
# current directory relative to root makefile
cwd := $(shell realpath --strip --relative-to=../.. $(CURDIR))

.PHONY: all
all: $(d)/.done

$(d)/.done:
	$(MAKE) -C ../.. $(cwd)/.done

.PHONY: check
check: $(d)/.done
	python3 -m pytest ../../test_all.py -k $(cwd)

else

TGT_$(d) := $(d)/.done
SRC_$(d) := $(d)/base_case.py
NOTEBOOK_SRC_$(d) := $(d)/base_case.py
NOTEBOOK_IMGS_$(d) := $(d)/displacement_warp.gif
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 *.gif params.log .done
DIR_CLEAN_$(d) := output

$(d)/.done: $(d)/params.log
	@touch $@

$(d)/params.log : category := gia
$(d)/params.log: $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))
NOTEBOOK_IMGS := $(NOTEBOOK_IMGS) $(NOTEBOOK_IMGS_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
