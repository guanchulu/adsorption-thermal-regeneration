# Sources and data transformations

## Public simulation archive

CRAFTED v2.0.1, published 13 November 2023:
https://doi.org/10.5281/zenodo.10120180

Archive SHA256:
`f29f7b3d6b7d6c51fd113138bc4a811c448d4dcce66ed37fddbfa4141a238dde`

There are 97,704 matched isotherm–enthalpy records for 690 MOFs and 667 COFs. Force-field/charge variants are repeated simulation conditions. The normalized input tables and saved predictions preserve source identifiers. License and transformation notices accompany the files.

## Literature-derived adsorption data

NIST/ARPA-E SRD205:
https://doi.org/10.18434/T43882

Frozen source:
https://github.com/NIST-ISODB/isodb-library/tree/9e1ee472967d66ec7845e23c5d5b986eca5ff020

The retained multitemperature subset has 1,186 curves and 25,162 reported points. The 32 three-temperature holdout groups and the 122 two-temperature response groups overlap; their counts must not be added.

## Published 13X parameterization

Santori, G.; Luberti, M. Thermodynamics of thermally-driven adsorption compression. Sustainable Materials and Technologies 10 (2016), 1–9.
https://doi.org/10.1016/j.susmat.2016.09.001

Author manuscript:
https://www.pure.ed.ac.uk/ws/portalfiles/portal/27896214/Adsorption_Compression_Low_Grade_Heat.pdf

Table 8 supplies the dual-site equilibrium parameters; the published dry-solid heat capacity is assumed. The current calculation uses a different pressure-first regeneration path. Source identity and conversions are in `sources/public_case_provenance.txt` and `data/public_case_13x.json`.

## Gas heat capacity

NIST Chemistry WebBook CO2 Shomate data:
https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Units=SI&Mask=1#Thermo-Gas

Coefficient implementation: `code/path_model.py`.

## File integrity

The release data ZIP contains `DATA_FILE_MANIFEST.json` with every payload's SHA256 and byte count. Release-level checksums are in SHA256SUMS.txt.

