# StarDist 2D example dataset (small, shipped with the installer)

A subset of **"ZeroCostDL4Mic - Stardist 2D example training and test dataset (light)"**,
Johanna Jukkala and Guillaume Jacquemet, Zenodo, 2023,
[doi:10.5281/zenodo.7949940](https://doi.org/10.5281/zenodo.7949940), licence
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- 6 training image/mask pairs and the 2 test pairs of that record (1024×1024, 16-bit images, 8-bit masks).
- Pixels are unchanged; files are renamed (no spaces) and saved with lossless zlib compression.
  `original_names.csv` gives each file's original name.
- It lets the notebook run section 3.1 without internet (offline installs). The full record (45 training pairs)
  is downloaded instead when the notebook is online and you ask for it.
