# Rehosted StarDist 2D models

Small StarDist 2D models offered in StarDist2D's model lists: the Fiji, napari and QuPath tools (*Model*, and
*Start from* in *Train a model*), the notebook (sections 3.4 and 6) and the command line. Each zip holds
`config.json`, `thresholds.json`, `weights_best.h5` (unchanged from the source) and a `README.md` naming the source
and licence. StarDist2D downloads a model the first time it is used, from this folder of release
1.2.0, through StarDist's own registry (`StarDist2D.from_pretrained`), which checks the SHA-256 in
`src/stardist2d/model_catalogue.py` and caches the model in `~/.keras/models/StarDist2D/`.

- The **Cell Migration Lab (CML)** models come from the model folders inside the lab's Zenodo datasets (section 3.1
  downloads the whole datasets, 100-580 MB each; a model alone is about 5.8 MB). All are CC BY 4.0: cite the record.
- The **BioImage Model Zoo (BMZ)** model is the entry `sincere-microbe`; its `config.json` and `thresholds.json` are
  the entry's StarDist configuration, its weights the entry's `stardist_weights.h5`. MIT licence.

All are single-channel models (32 rays, grid 2), so they also work as a start for *Train a model* with its defaults.

| Shown as | File (`<name>.zip`) | Images | Authors | Cite | Licence | SHA-256 |
|---|---|---|---|---|---|---|
| CML: HUVEC nuclei (fluorescence) | `cml_huvec_nuclei` | fluorescence | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.10617532 | cc-by-4.0 | `47a5f56f83e718bf…` |
| CML: fluorescent cells | `cml_fluorescent_cells` | fluorescence | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.10572310 | cc-by-4.0 | `3137961901aaef20…` |
| CML: AsPC1 cells, Lifeact (fluorescence) | `cml_aspc1_lifeact` | fluorescence | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.13442128 | cc-by-4.0 | `901b36a42736391e…` |
| CML: MiaPaCa2 cells, CD44 (fluorescence) | `cml_miapaca2_cd44` | fluorescence | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.13442877 | cc-by-4.0 | `b3139f57c1b3eb03…` |
| CML: tumour cell nuclei, DAPI (fluorescence) | `cml_tumour_cell_nuclei_dapi` | fluorescence | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.13443221 | cc-by-4.0 | `e655d37d941c4a44…` |
| CML: neutrophils (brightfield) | `cml_neutrophils_bf` | brightfield | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.10572231 | cc-by-4.0 | `2b60ebd7e91feb47…` |
| CML: monocytes (brightfield) | `cml_monocytes_bf` | brightfield | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.10572200 | cc-by-4.0 | `eac8ab3f31634c69…` |
| CML: cancer cells 20x (brightfield) | `cml_cancer_cells_bf_20x` | brightfield | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.10572122 | cc-by-4.0 | `d648ae903db074f3…` |
| CML: cancer cells 10x (brightfield) | `cml_cancer_cells_bf_10x` | brightfield | Follain, Gautier, Ghimire, Sujan, Pylvänäinen, Joanna, Ivaska, Johanna, Jacquemet, Guillaume | doi:10.5281/zenodo.13304399 | cc-by-4.0 | `97d5ff4eb4e2b0f2…` |
| BMZ: CHO mitotic rounding (brightfield) | `bmz_cho_mitotic_rounding_bf` | brightfield | Estibaliz Gómez de Mariscal, Mario Del Rosario | BioImage Model Zoo entry sincere-microbe (https://bioimage.io/#/?id=sincere-microbe) | MIT | `ac86a1954e5153e0…` |

The *fluorescent cells* dataset also holds a copy of StarDist's `2D_versatile_fluo` in a subfolder; the model here is
the lab's own model at the top of its model folder.

To add a model: put its zip here (files at the top of the zip, as above), add a `CatalogueModel` with its SHA-256 to
`model_catalogue.py` and its label to `CatalogueModels` in `src/stardist2d_lc_tools/__init__.py`;
`tests/test_model_catalogue.py` checks that the three agree.
