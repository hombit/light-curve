# `light-curve`
## Irregular time series processing tools for Rust and Python

The project is aimed to build high-performance tools for light curve analysis suitable to process alert and archival data of current [ZTF](https://ztf.caltech.edu) and future [Vera Rubin Observatory LSST](https://lsst.org) photometric surveys.

The most of the code base is written on Rust and split into several crates.

### Structure

- `light-curve`**WIP** A crate which will be an umbrella for other Rust crates

- `light-curve-common` ![docs.rs badge](https://docs.rs/light-curve-common/badge.svg) Common tools for other crates

- `light-curve-dmdt` ![docs.rs badge](https://docs.rs/light-curve-dmdt/badge.svg) [dm-dt](https://arxiv.org/abs/1709.06257) mapper crate and executable

- `light-curve-feature` ![docs.rs badge](https://docs.rs/light-curve-feature/badge.svg) A collection of features to be extracted from light curves
- `light-curve-interpol`**WIP** Light curve interpolation tools. Currently it includes linear interpolation only
- `light-curve-python` [![PyPI version](https://badge.fury.io/py/light-curve-python.svg)](https://badge.fury.io/py/light-curve-python) Python bindings to `light-curve-dmdt` and `light-curve-feature`. Note that in the future releases `light-curve-python`Python package will be renamed to `light-curve`.
- `.ci` Continuous integration related stuff, currently it is just a custom Docker [maturin](https://github.com/pyo3/maturin) image used to publish x86-64 Linux Python packages via Github Actions
- `.github` GitHub related utils, such as dependabot configuration and Actions workflows

### Citation

If you found this project useful for your research please cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract)

```bibtex
@ARTICLE{2021MNRAS.502.5147M,
       author = {{Malanchev}, K.~L. and {Pruzhinskaya}, M.~V. and {Korolev}, V.~S. and {Aleo}, P.~D. and {Kornilov}, M.~V. and {Ishida}, E.~E.~O. and {Krushinsky}, V.~V. and {Mondon}, F. and {Sreejith}, S. and {Volnova}, A.~A. and {Belinski}, A.~A. and {Dodin}, A.~V. and {Tatarnikov}, A.~M. and {Zheltoukhov}, S.~G. and {(The SNAD Team)}},
        title = "{Anomaly detection in the Zwicky Transient Facility DR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, astronomical data bases: miscellaneous, stars: variables: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {502},
       number = {4},
        pages = {5147-5175},
          doi = {10.1093/mnras/stab316},
archivePrefix = {arXiv},
       eprint = {2012.01419},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
