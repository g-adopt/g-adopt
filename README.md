G-ADOPT
=======

This repository contains material and examples relating to the G-ADOPT
Platform, a computational platform for inverse geodynamics, being
developed and maintained by researchers from the Research School of
Earth Sciences at the Australian National University. It builds on a
recent surge in accessible observational datasets and advances in
inversion methods using sophisticated adjoint techniques that provide
a mechanism for fusing these observations with dynamics, physics and
chemistry.

G-ADOPT is supported and funded by the Australian Research Data
Commons (ARDC), with additional partner contributions from AuScope,
the NCI and Geosciences Australia.

Installation
------------

G-ADOPT is available on PyPI as `gadopt`, and requires a working
[Firedrake](https://www.firedrakeproject.org/) installation. To bring
in the optional nonlinear optimisation dependencies, install the
`gadopt[optimisation]` variant. See [the G-ADOPT
website](https://gadopt.org/install/) for more detailed installation
instructions, including directions for getting started with the demo
notebooks.

Citing
------

If you use this software in your work, please cite the software using the following metadata and the two articles below:
<details>
<summary> APA references</summary>
   
    Gibson, A., Davies, R., Kramer, S., Ghelichkhan, S., Turner, R., Duvernay, T., & Scott, W. (2024). G-ADOPT (Version v2.2.0) [Computer software]. https://doi.org/10.5281/zenodo.5644391 
    
    Davies, D. R., Kramer, S. C., Ghelichkhan, S., & Gibson, A. (2022). Towards automatic finite-element methods for geodynamics via Firedrake. Geoscientific Model Development, 15(13), 5127–5166. doi:10.5194/gmd-15-5127-2022
    
    Ghelichkhan, S., Gibson, A., Davies, D. R., Kramer, S. C., & Ham, D. A. (2024). Automatic adjoint-based inversion schemes for geodynamics: reconstructing the evolution of Earth's mantle in space and time. Geoscientific Model Development, 17(13), 5057-5086.
</details>

<details>
<summary> Bibtex references</summary>
    
    @software{Gibson_G-ADOPT_2024,
    author = {Gibson, Angus and Davies, Rhodri and Kramer, Stephan and Ghelichkhan, Sia and Turner, Ruby and Duvernay, Thomas and Scott, Will},
    doi = {10.5281/zenodo.5644391},
    month = feb,
    title = {{G-ADOPT}},
    url = {https://github.com/g-adopt/g-adopt},
    version = {v2.2.0},
    year = {2024}
    }
    
    @Article{Davies_Towards_2022,
    AUTHOR = {Davies, D. R. and Kramer, S. C. and Ghelichkhan, S. and Gibson, A.},
    TITLE = {Towards automatic finite-element methods for geodynamics via Firedrake},
    JOURNAL = {Geoscientific Model Development},
    VOLUME = {15},
    YEAR = {2022},
    NUMBER = {13},
    PAGES = {5127--5166},
    URL = {https://gmd.copernicus.org/articles/15/5127/2022/},
    DOI = {10.5194/gmd-15-5127-2022}
    }
    
   @Article{gmd-17-5057-2024,
   AUTHOR = {Ghelichkhan, S. and Gibson, A. and Davies, D. R. and Kramer, S. C. and Ham, D. A.},
   TITLE = {Automatic adjoint-based inversion schemes for geodynamics: reconstructing the evolution of Earth's mantle in space and time},
   JOURNAL = {Geoscientific Model Development},
   VOLUME = {17},
   YEAR = {2024},
   NUMBER = {13},
   PAGES = {5057--5086},
   URL = {https://gmd.copernicus.org/articles/17/5057/2024/},
   DOI = {10.5194/gmd-17-5057-2024}
   }
</details>


Please also cite Firedrake using the instructions [here](https://www.firedrakeproject.org/citing.html).
