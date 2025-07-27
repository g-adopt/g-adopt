all: targets

include ../.rules.mk

define include_subdir =
dir := $(dir)
include $$(dir)/Makefile
endef

define subdir_targets =
.PHONY: $(1) clean-$(1)
$(1): $$(TGT_$(1))

clean-$(1):
	rm -f $$(CLEAN_$(1))
	rm -rf $$(DIR_CLEAN_$(1))
endef

subdirs := mantle_convection multi_material glacial_isostatic_adjustment PDE_constrained_optimisation

$(foreach dir,$(subdirs),$(eval $(include_subdir)))

.PHONY: targets
targets: $(TGT)

notebook_files := $(NOTEBOOK_SRC:.py=.ipynb)
artifact.tar: $(notebook_files) .pages .diagram.mermaid
	tar --transform='s/.pages/CONTENTS.md/' --create --file $@ $(filter-out %.ipynb,$^)
	tar --transform='s|/.*/|/|' --append --file $@ $(notebook_files)
	$(foreach image,$(NOTEBOOK_IMGS),tar --append --file $@ $(image) ;)

$(foreach dir,$(subdirs),$(eval $(call subdir_targets,$(dir))))

.PHONY: clean
clean:
	rm -f $(CLEAN)
	rm -rf $(DIR_CLEAN)
