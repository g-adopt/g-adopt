mc_dir := mantle_convection
mc_demos := base_case 3d_cartesian viscoplastic_case free_surface
mc_demos += 2d_cylindrical 3d_spherical gplates_global
mc_demos += 2d_compressible_TALA 2d_compressible_ALA
mc_demos += adjoint
mc_cases := $(addprefix $(mc_dir)/, $(mc_demos))

mm_dir := multi_material
mm_demos := compositional_buoyancy thermochemical_buoyancy
mm_cases := $(addprefix $(mm_dir)/, $(mm_demos))

gia_dir := glacial_isostatic_adjustment
gia_demos := base_case 2d_cylindrical
gia_cases := $(addprefix $(gia_dir)/, $(gia_demos))

notebook_files := $(foreach case,$(mc_cases),$(case)/$(notdir $(case)).ipynb)
notebook_files += $(foreach case,$(mm_cases),$(case)/$(notdir $(case)).ipynb)
notebook_files += $(foreach case,$(gia_cases),$(case)/$(notdir $(case)).ipynb)
notebook_files += ${mc_dir}/adjoint/adjoint_forward.ipynb
notebook_files += ${mc_dir}/visualise_ALA_p_nullspace/visualise_ALA_p_nullspace.ipynb