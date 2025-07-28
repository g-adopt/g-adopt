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
	python3 -m pytest

else

TGT_$(d) := $(d)/functional.txt
SRC_$(d) := $(d)/adjoint.py $(d)/adjoint_forward.py
NOTEBOOK_SRC_$(d) := $(d)/adjoint.py $(d)/adjoint_forward.py
CLEAN_$(d) := functional.txt *.h5 *.pvd .done
DIR_CLEAN_$(d) := solutions solution visualisation_vtk optimisation_checkpoint

$(d)/.done: $(TGT_$(d))
	@touch $@

$(d)/functional.txt : category := mantle_convection
$(d)/functional.txt: $(d)/adjoint.py $(d)/adjoint-demo-checkpoint-state.h5
	$(run-python)

$(d)/adjoint-demo-checkpoint-state.h5 : category := mantle_convection
$(d)/adjoint-demo-checkpoint-state.h5 : ncpus := 2
$(d)/adjoint-demo-checkpoint-state.h5: $(d)/adjoint_forward.py
	$(run-python)

# we also need the ordering between the notebooks to ensure
# the forward is executed first
$(d)/adjoint.ipynb: $(d)/adjoint_forward.ipynb

TGT := $(TGT) $(TGT_$(d))
NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
