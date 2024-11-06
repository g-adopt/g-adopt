mc_tests := base_case 2d_compressible_ALA 2d_compressible_TALA viscoplastic_case 2d_cylindrical 3d_spherical 3d_cartesian adjoint gplates_global free_surface
mc_demos := base_case 2d_cylindrical 3d_spherical 2d_compressible_ALA visualise_ALA_p_nullspace 2d_compressible_TALA viscoplastic_case 3d_cartesian gplates_global adjoint free_surface
mc_dir := mantle_convection
mc_demos := $(addprefix $(mc_dir)/,$(mc_demos))
# sort to remove duplicates which define both tests and demos
mc_cases := $(addprefix $(mc_dir)/,$(sort $(mc_tests) $(mc_demos)))

mm_tests := compositional_buoyancy thermochemical_buoyancy
mm_demos := compositional_buoyancy thermochemical_buoyancy
mm_dir:= multi_material
mm_demos := $(addprefix $(mm_dir)/,$(mm_demos))
# sort to remove duplicates which define both tests and demos
mm_cases := $(addprefix $(mm_dir)/,$(sort $(mm_tests) $(mm_demos)))

gia_tests := base_case 2d_cylindrical
gia_demos := base_case 2d_cylindrical
gia_dir := glacial_isostatic_adjustment
gia_demos := $(addprefix $(gia_dir)/,$(gia_demos))
# sort to remove duplicates which define both tests and demos
gia_cases := $(addprefix $(gia_dir)/,$(sort $(gia_tests) $(gia_demos)))

notebook_files := $(foreach case,$(mc_demos) $(mm_demos) $(gia_demos),$(case)/$(notdir $(case)).ipynb) adjoint/adjoint_forward.ipynb