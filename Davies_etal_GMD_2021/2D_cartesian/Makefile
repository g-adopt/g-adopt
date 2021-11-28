cases := base_case compressible_case viscoplastic_case

nthreads := 1
export nthreads

.PHONY: all $(cases) clean

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(cases)
