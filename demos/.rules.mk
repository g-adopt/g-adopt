# Standalone demos
# ----------------
#
# This rule dispatches to the .done and .clean targets that are part
# of $(default_targets) for running a demo standalone within its own directory.

define standalone =
cwd := $(shell realpath --strip --relative-to=../../.. $(CURDIR))

.PHONY: all
all: .done

# This is the important rule that dispatches to the "real" Make
# workflow to run the case if necessary.
.done:
	$$(MAKE) -C ../../.. $$(cwd)/.done

.PHONY: clean
clean:
	$$(MAKE) -C ../../../ $$(cwd)/.clean
endef
