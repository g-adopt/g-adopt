mc_test_names := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint gplates_global free_surface dynamic_topography Drucker_Prager
mc_demo_names := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA visualise_ALA_p_nullspace 2d_compressible_TALA viscoplastic_case 3d_cartesian gplates_global adjoint free_surface dynamic_topography Drucker_Prager
mc_dir := mantle_convection
mc_demos := $(addprefix $(mc_dir)/,$(mc_demo_names))
mc_cases := $(addprefix $(mc_dir)/,$(mc_test_names))

mm_test_names := 2d_cylindrical compositional_buoyancy free_surface thermochemical_buoyancy
mm_demo_names := 2d_cylindrical compositional_buoyancy free_surface thermochemical_buoyancy
mm_dir:= multi_material
mm_demos := $(addprefix $(mm_dir)/,$(mm_demo_names))
mm_cases := $(addprefix $(mm_dir)/,$(mm_test_names))

gia_test_names := base_case 2d_cylindrical 2d_cylindrical_lvv adjoint_ice
gia_demo_names := base_case 2d_cylindrical 2d_cylindrical_lvv adjoint_ice
gia_dir := glacial_isostatic_adjustment
gia_demos := $(addprefix $(gia_dir)/,$(gia_demo_names))
gia_cases := $(addprefix $(gia_dir)/,$(gia_test_names))

# other cases not tied to specific physics/domains
extra_cases := PDE_constrained_optimisation smoothing

# sort to remove duplicates which define both tests and demos
all_cases := $(sort $(mc_demos) $(mc_cases) $(mm_demos) $(mm_cases) $(gia_demos) $(gia_cases) $(extra_cases))

notebook_files := $(foreach case,$(mc_demos) $(mm_demos) $(gia_demos),$(case)/$(notdir $(case)).ipynb) $(mc_dir)/adjoint/adjoint_forward.ipynb PDE_constrained_optimisation/PDE_constrained_boundary.ipynb PDE_constrained_optimisation/PDE_constrained_field.ipynb smoothing/smoothing.ipynb

diagram_files := mantle_convection/.diagram.mermaid glacial_isostatic_adjustment/.diagram.mermaid
