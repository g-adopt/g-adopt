cases := base_case 2d_compressible viscoplastic_case 2d_cylindrical analytical_comparisons Drucker-Prager_rheology

.PHONY: all $(cases) clean generate

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python3 generate_expected.py

clean: $(cases)
