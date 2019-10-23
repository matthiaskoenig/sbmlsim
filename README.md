
[![Build Status](https://travis-ci.org/matthiaskoenig/sbmlsim.svg?branch=develop)](https://travis-ci.org/matthiaskoenig/sbmlsim)
[![Documentation Status](https://readthedocs.org/projects/sbmlsim/badge/?version=latest)](https://sbmlsim.readthedocs.io/en/latest/)
[![License (LGPL version 3)](https://img.shields.io/badge/license-LGPLv3.0-blue.svg?style=flat-square)](http://opensource.org/licenses/LGPL-3.0)

<h1>sbmlsim: SBML simulation made easy</h1>
<b><a href="https://orcid.org/0000-0003-1725-179X" title="https://orcid.org/0000-0003-1725-179X"><img src="./docs/static/images/orcid.png" height="15"/></a> Matthias König</b>

`sbmlsim`: SBML simulation made easy

For documentation and examples see https://sbmlsim.readthedocs.io

### License
* Source Code: [LGPLv3](http://opensource.org/licenses/LGPL-3.0)
* Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

### Funding
Matthias König is supported by the Federal Ministry of Education and Research (BMBF, Germany)
within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054).

## Installation
`sbmlsim` currently works only python3.6.

HDF5 support is required which can be installed via
```
sudo apt-get install -y libhdf5-serial-dev
```

Best install `sbmlsim` in a virtual environment via pip
```bash
# checkout code
git clone https://github.com/matthiaskoenig/sbmlsim.git

# virtualenv
cd sbmlsim
mkvirtualenv sbmlsim --python=python3.6
(sbmlsim) pip install -e .
```
The notebook support can be installed via
```bash
(sbmlsim) pip install jupyter jupyterlab
(sbmlsim) python -m ipykernel install --user --name sbmlsim
```
Notebooks are available in
- `docs/notebooks` (notebooks used in documentation on https://sbmlsim.readthedocs.io/en/latest/)
- `notebooks`

and can be started from within `jupyterlab`
```
(sbmlsim) jupyter lab
```

&copy; 2019 Matthias König
