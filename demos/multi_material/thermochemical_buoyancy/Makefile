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

else

$(d)/.done: $(d)/params.log
	@touch $@

$(d)/params.log : category := multi_material
$(d)/params.log: $(d)/thermochemical_buoyancy.py
	$(run-python)

TGT_$(d) := $(d)/.done
CLEAN_$(d) := *.pvd *.vtu *.pvtu *.h5 params.log .done
DIR_CLEAN_$(d) := output

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
