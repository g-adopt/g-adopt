cases := 2D_cartesian 2D_cylindrical 3D_cartesian 3D_spherical

.PHONY: all $(cases) clean generate

all: $(cases)

$(cases):
	$(MAKE) -C $@ $(MAKECMDGOALS)

generate:
	python test_all.py

clean: $(cases)
