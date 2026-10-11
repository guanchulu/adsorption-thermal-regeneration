# CRAFTED - An exploratory database of simulated adsorption isotherms of metal-organic frameworks

## Overview

The files in this repository compose the **C**harge-dependent, **R**eproducible, **A**ccessible, **F**orcefield-dependent, and **T**emperature-dependent **E**xploratory **D**atabase (**CRAFTED**) of adsorption isotherms. This dataset contains the simulation of CO2 and N2 adsorption isotherms on 690 metal-organic frameworks taken from the CoRE-MOF-2014 database and 667 covalent organic frameworks taken from the CURATED-COFs database. The simulations were performed with two force fields (UFF and DREIDING), six partial charge schemes (no charges, Qeq, EQeq, DDEC, MPNN, and PACMOF), and three temperatures (273, 298, 323 K).

### Contents

- `CIF_FILES/` contains 6 folders (NEUTRAL, DDEC, EQeq, Qeq, MPNN, and PACMOF), each one with 1357 CIF files;
- `FORCEFIELDS/` contains 2 folders (UFF and DREIDING) with the definition of the forcefields;
- `INPUT_FILES/` contains 97,704 input files for the GCMC simulations;
- `ISOTHERM_FILES/` contains 97,704 adsorption isotherms resulting from the GCMC simulation;
- `ENTHALPY_FILES/` contains 97,704 enthalpies of adsorption from the isotherms;
- `RAC_DBSCAN/` contains the RAC and geometrical descriptors to perform the t-NSE + DBSCAN analysis;

### Licenses

The 690 MOF-related CIF files in the `DDEC` folder were downloaded from [CoRE-MOF-2014](https://doi.org/10.5281/zenodo.3986573) and are licensed under the terms of the Creative Commons Attribution 4.0 International license ([`CC-BY-4.0`](https://creativecommons.org/licenses/by/4.0/legalcode)).
The 667 COF-related CIF files in the `NEUTRAL` folder were downloaded from [CURATED-COFs](https://github.com/danieleongari/CURATED-COFs) and are licensed under the terms of the MIT license ([`MIT`](https://github.com/danieleongari/CURATED-COFs/blob/master/LICENSE)).

> Dalar Nazarian, Jeffrey S. Camp, & David S. Sholl. (2016). Computation-Ready Experimental Metal-Organic Framework (CoRE MOF) 2014 DDEC Database [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.3986573>
>
> Ongari, Daniele, et al. "Building a consistent and reproducible database for adsorption evaluation in covalent–organic frameworks." ACS Central Science 5.10 (2019): 1663-1675. <https://doi.org/10.1021/acscentsci.9b00619>
>
> Ongari, Daniele, Leopold Talirz, and Berend Smit. "Too many materials and too many applications: An experimental problem waiting for a computational solution." ACS Central Science 6.11 (2020): 1890-1900. <https://doi.org/10.1021/acscentsci.0c00988>

The `CO2.def` and `N2.def` forcefield files were downloaded from [RASPA](https://github.com/iRASPA/RASPA2/tree/master/molecules/ExampleDefinitions) and are licensed under the terms of the [MIT](https://github.com/iRASPA/RASPA2/blob/master/COPYING) license.

> Dubbeldam, David, et al. "RASPA: molecular simulation software for adsorption and diffusion in flexible nanoporous materials." Molecular Simulation 42.2 (2016): 81-101. <https://doi.org/10.1080/08927022.2015.1010082>

The remaining MOF-related CIF files in the `PACMOF`, `MPNN`, `Qeq`, `EQeq` and `NEUTRAL` folders were derived from those in the `DDEC` folder and are licensed under the terms of the Creative Commons Attribution 4.0 International license ([`CC-BY-4.0`](https://creativecommons.org/licenses/by/4.0/legalcode)) from the CoRE-MOF-2014 subset.
The remaining COF-related CIF files in the `PACMOF`, `MPNN`, `Qeq`, `EQeq` and `DDEC` folders were derived from those in the `NEUTRAL` folder and are licensed under the terms of the MIT license ([`MIT`](https://github.com/danieleongari/CURATED-COFs/blob/master/LICENSE)) from the CURATED-COFs subset.

All remaining files were created by us, and are licensed under the terms of the [`CDLA-Sharing-1.0`](https://cdla.dev/sharing-1-0/) license.

### Software requirements

In order to create a Python environment capable of running the Jupyter notebooks, please install [`conda`](https://docs.conda.io/en/latest/miniconda.html) and execute

```Shell
conda env create --file environment.yml
```

### Usage instructions

Execute the command below to run JupyterLab in the appropriate Python environment.

```Shell
conda run --name crafted jupyter-lab
```
