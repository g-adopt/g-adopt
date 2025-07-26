sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

subdirs_$(d) := base_case 2d_cylindrical

$(foreach dir,$(addprefix $(d)/,$(subdirs_$(d))),$(eval $(include_subdir)))

TGT_$(d) := $(filter $(dir)/%,$(TGT))
CLEAN_$(d) := $(filter $(dir)/%,$(CLEAN))
DIR_CLEAN_$(d) := $(filter $(dir)/%,$(DIR_CLEAN))

d := $(dirstack_$(sp))
sp := $(basename $(sp))
