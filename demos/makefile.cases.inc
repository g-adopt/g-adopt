mc_test_names := adjoint_2d_cylindrical
mc_demo_names :=
mc_dir := mantle_convection
mc_demos := $(addprefix $(mc_dir)/,$(mc_demo_names))
mc_cases := $(addprefix $(mc_dir)/,$(mc_test_names))

mm_test_names :=
mm_demo_names := compositional_buoyancy thermochemical_buoyancy
mm_dir:= multi_material
mm_demos := $(addprefix $(mm_dir)/,$(mm_demo_names))
mm_cases := $(addprefix $(mm_dir)/,$(mm_test_names))

gia_test_names :=
gia_demo_names := base_case 2d_cylindrical
gia_dir := glacial_isostatic_adjustment
gia_demos := $(addprefix $(gia_dir)/,$(gia_demo_names))
gia_cases := $(addprefix $(gia_dir)/,$(gia_test_names))

# other cases not tied to specific physics/domains
extra_cases :=

# sort to remove duplicates which define both tests and demos
all_cases := $(sort $(mc_cases))

notebook_files := $(foreach case,$(mc_demos) $(mm_demos) $(gia_demos),$(case)/$(notdir $(case)).ipynb) $(mc_dir)/adjoint/adjoint_forward.ipynb PDE_constrained_optimisation/PDE_constrained_boundary.ipynb PDE_constrained_optimisation/PDE_constrained_field.ipynb
