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
	python3 -m pytest -m "not longtest"

else

CASES_$(d) := smooth_cylindrical_freeslip smooth_cylindrical_zeroslip delta_cylindrical_freeslip delta_cylindrical_zeroslip delta_cylindrical_freeslip_dpc delta_cylindrical_zeroslip_dpc
HPC_CASES_$(d) := smooth_cylindrical_freesurface smooth_spherical_freeslip smooth_spherical_zeroslip
TGT_$(d) := $(addprefix $(d)/.sentinel.,$(CASES_$(d)))
HPC_TGT_$(d) := $(addprefix $(d)/.sentinel.,$(HPC_CASES_$(d)))
SRC_$(d) := $(d)/analytical.py
CLEAN_$(d) := *.dat .sentinel.* .done
DIR_CLEAN_$(d) := batch_output __pycache__

$(d)/.done: $(TGT_$(d))
	@touch $@

$(TGT_$(d)) : case = $(patsubst .sentinel.%,%,$(notdir $@))
$(TGT_$(d)) : tsp_string := $(if $(BATCH_MODE),tsp -N {cores} -f )
$(TGT_$(d)): $(SRC_$(d))
	@(cd $(dir $<); $(call run_regular,python3 analytical.py submit -t "$(tsp_string)mpiexec -np {cores}" $(case),analytical $(case)))
	@touch $@

batch_output:
	mkdir batch_output

$(HPC_TGT_$(d)) : case = $(patsubst .sentinel.%,%,$(notdir $@))
$(HPC_TGT_$(d)): $(SRC_$(d)) batch_output clean-$(d)
	(cd $(dir $<); python3 analytical.py submit $(case) -H -o batch_output/$(case)_l{level}_{params}.out -e batch_output/$(case)_l{level}_{params.err}))

TGT := $(TGT) $(TGT_$(d))
HPC_TGT := $(HPC_TGT) $(HPC_TGT_$(d))
CLEAN := $(CLEAN) $(addprefix $(d)/,$(CLEAN_$(d)))
DIR_CLEAN := $(DIR_CLEAN) $(addprefix $(d)/,$(DIR_CLEAN_$(d)))

endif

d := $(dirstack_$(sp))
sp := $(basename $(sp))
