sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

subdirs_$(d) := base_case 2d_cylindrical

$(foreach dir,$(addprefix $(d)/,$(subdirs_$(d))),$(eval $(include_subdir)))

TGT_$(d) := $(filter $(d)/%,$(TGT))
CLEAN_$(d) := $(filter $(d)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(d)/%,$(DIR_CLEAN))

d := $(dirstack_$(sp))
sp := $(basename $(sp))
