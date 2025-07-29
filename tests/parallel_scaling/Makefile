sp := $(sp).x
dirstack_$sp) := $(d)
d := $(dir)

ifeq ($(d),)

d := .
cwd := $(shell realpath --strip --relative-to=../.. $(CURDIR))

.PHONY: all
all: $(d)/.hpc_done

$(d)/.hpc_done:
	$(MAKE) -C ../.. $(cwd)/.hpc_done

.PHONY: check
check: $(d)/.hpc_done
	python3 -m pytest

else

# refinement levels
HPC_CASES_$(d) := 5 6 7
HPC_TGT_$(d) := $(HPC_CASES_$(d):%=$(d)/profile_%.txt)
SRC_$(d) := $(d)/scaling.py $(d)/stokes_cubed_sphere.py
CLEAN_$(d) := profile_*.txt level_*.out level_*.err
DIR_CLEAN_$(d) := batch_output

$(d)/.hpc_done: $(HPC_TGT_$(d))
	@touch $@

$(d)/batch_output:
	@mkdir $@

$(HPC_TGT_$(d)): $(d)/profile_%.txt: $(SRC_$(d)) $(d)/batch_output $(d)/run.template
	(cd $(dir $<); gadopt_hpcrun --debug -n $$(python3 scaling.py get_ncpus $*) -v LEVEL=$* -N scaling_$* -o batch_output/l$*.out -e batch_output/$*.err --template-file ./run.template python3 scaling.py run $*

HPC_TGT := $(HPC_TGT) $(HPC_TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
