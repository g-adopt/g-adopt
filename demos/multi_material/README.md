Multi-material benchmark simulations underpinned by the conservative level-set approach.

Simulations can be run using
`mpiexec -n NCORES python3 run_benchmark.py benchmarks.MODULE_NAME`

**benchmarks.crameri_2012**
Isostatic relaxation of a topography cosine perturbation using the sticky-air approach.
- Compositional buoyancy
- Two material interfaces with homogeneous stratification
- Sticky air

**benchmarks.gerya_2003**
Sinking of a hard rectangular block into a lower-viscosity medium.
- Compositional buoyancy

**benchmarks.robey_2019**
Thermochemical convection in a density-stratified fluid.
- Compositional and thermal buoyancy

**benchmarks.schmalholz_2011**
Necking and detachment of a slab deforming through a power-law viscous rheology.
- Compositional buoyancy
- Non-linear rheology

**benchmarks.schmeling_2008**
Spontaneous subduction driven by compositional buoyancy using the sticky-air approach.
- Compositional buoyancy
- Two material interfaces with inhomogeneous stratification
- Sticky air

**benchmarks.tosi_2015**
Viscoplastic thermal convection with a passive material interface.
- Thermal buoyancy
- Passive material interface

**benchmarks.trim_2023**
Thermochemical, temporally periodic flow using a manufactured solution.
- Compositional and thermal buoyancy
- Manufactured solution

**benchmarks.van_keken_1997_isothermal**
Isothermal Rayleigh-Taylor instability.
- Compositional buoyancy

**benchmarks.van_keken_1997_thermochemical**
Entrainment of a thin, compositionally dense layer by thermal convection.
- Compositional and thermal buoyancy