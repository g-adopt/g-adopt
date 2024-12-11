demos := PDE_constrained_optimisation
demo_paths := $(demos)

mc_dir := mantle_convection
mc_demos := base_case 3d_cartesian viscoplastic_case free_surface
mc_demos += 2d_cylindrical 3d_spherical gplates_global
mc_demos += 2d_compressible_TALA 2d_compressible_ALA
mc_demos += adjoint
mc_demo_paths := $(addprefix $(mc_dir)/, $(mc_demos))

mm_dir := multi_material
mm_demos := compositional_buoyancy free_surface thermochemical_buoyancy
mm_demo_paths := $(addprefix $(mm_dir)/, $(mm_demos))

gia_dir := glacial_isostatic_adjustment
gia_demos := base_case 2d_cylindrical
gia_demo_paths := $(addprefix $(gia_dir)/, $(gia_demos))

all_dirs := $(mc_dir) $(mm_dir) $(gia_dir)
all_demos := $(demos) $(mc_demos) $(mm_demos) $(gia_demos)
all_demo_paths := $(demo_paths) $(mc_demo_paths) $(mm_demo_paths) $(gia_demo_paths)

notebook_files := $(foreach demo,$(all_demo_paths),$(demo)/$(notdir $(demo)).ipynb)
notebook_files += PDE_constrained_optimisation/PDE_constrained_boundary.ipynb
notebook_files += PDE_constrained_optimisation/PDE_constrained_field.ipynb
notebook_files += ${mc_dir}/adjoint/adjoint_forward.ipynb
notebook_files += ${mc_dir}/visualise_ALA_p_nullspace/visualise_ALA_p_nullspace.ipynb
notebook_files := $(filter-out PDE_constrained_optimisation/PDE_constrained_optimisation.ipynb,$(notebook_files))
