# Data licenses and attribution

## CRAFTED 2.0.1

Source: Felipe Lopes Oliveira, Conor Cleeton, Rodrigo Neumann Barros Ferreira, Binquan Luan, Amir H. Farmahini, Lev Sarkisov and Mathias Steiner. CRAFTED: An exploratory database of simulated adsorption isotherms of nanoporous materials, version 2.0.1 (2023).

DOI: https://doi.org/10.5281/zenodo.10120180

The original archive is preserved without modification in the release data ZIP:
`data/crafted_raw/CRAFTED-2.0.1.tar.xz`.

The source adsorption and enthalpy data are distributed under the **Community Data License Agreement – Sharing, Version 1.0**. Its unmodified text is in `licenses/archive_1_LICENSE` and `data/crafted_raw/archive_1_LICENSE`. That license continues to apply to the source data and enhanced/normalized CRAFTED data supplied here. No restriction in this repository supersedes those rights.

The original archive also contains MOF CIFs under CC BY 4.0, COF CIFs under MIT, and RASPA molecule definitions under MIT. Preserve the original internal attribution and license notices when using those components.

**Modification notice:** `data/crafted_*.parquet` and CRAFTED observations in `results/crafted_*.parquet`, `results/crafted_full/` and the Figure 5 source CSVs are reformatted, joined, selected or augmented by the scripts in this repository. Pressure is converted from Pa to bar; source adsorption enthalpy and its sign-transformed comparison observable are distinguished by column names. Structure, gas, force-field, charge and temperature identifiers are retained. The raw archive is not changed. The companion NOTICE files describe these transformations at file level.

## NIST adsorption data

Source: D. W. Siderius, V. K. Shen, R. D. Johnson III and R. D. van Zee (editors), NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials, SRD205.
https://doi.org/10.18434/T43882

The frozen upstream commit is `9e1ee472967d66ec7845e23c5d5b986eca5ff020`. Its source URL and checksums are supplied in `data/snapshot_manifest.json`. Reported material identifiers, source DOIs and uptake conventions remain attached to the extracted data. Cite the underlying experimental publications as well as the database when using their observations.

## 13X model and heat capacities

Only the parameter values, conversions and calculated results are redistributed. The original article PDF and screenshots are not included. See DATA_SOURCES.md for the primary-source links.

## Study-specific work

This deposit makes the study code and calculated outputs publicly accessible. The authors have not selected a separate reuse license for original study code and independent model outputs. The source-data and enhanced-data licenses described above remain in force.

