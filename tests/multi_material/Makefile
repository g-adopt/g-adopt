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

CASES_$(d) := $(notdir $(wildcard $(d)/benchmarks/*))
TGT_$(d) := $(CASES_$(d):%=$(d)/benchmarks/%/output_0_reference.npz)
SRC_$(d) := $(d)/run_benchmark.py
CLEAN_$(d) := .done $(CASES_$(d):%=benchmarks/%/*.pdf) $(CASES_$(d):%=benchmarks/%/*.npz) $(CASES_$(d):%=benchmarks/%/mesh/*.msh)
DIR_CLEAN_$(d) := __pycache__ $(CASES_$(d):%=benchmarks/%/__pycache) $(CASES_$(d):%=benchmarks/%/outputs)

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)) : category := multi_material
$(TGT_$(d)) : ncpus := 8
$(TGT_$(d)) : exec_cmd = mpiexec -np $(ncpus) python3 $(notdir $<) benchmarks/$* --without-plot
$(TGT_$(d)) : desc = $*
$(TGT_$(d)): $(d)/benchmarks/%/output_0_reference.npz: $(SRC_$(d))
	$(run-python)

TGT := $(TGT) $(TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
