sp := $(sp).x
dirstack_$(sp) := $(d)
d := $(dir)

NOTEBOOK_SRC_$(d) := visualise_ALA_p_nullspace.py

NOTEBOOK_SRC := $(NOTEBOOK_SRC) $(NOTEBOOK_SRC_$(d))

d := $(dirstack_$(sp))
sp := $(basename $(sp))
