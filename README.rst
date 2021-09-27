sbmlsim: SBML simulation made easy
==================================

.. image:: https://github.com/matthiaskoenig/sbmlsim/workflows/CI-CD/badge.svg
   :target: https://github.com/matthiaskoenig/sbmlsim/workflows/CI-CD
   :alt: GitHub Actions CI/CD Status

.. image:: https://img.shields.io/pypi/v/sbmlsim.svg
   :target: https://pypi.org/project/sbmlsim/
   :alt: Current PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/sbmlsim.svg
   :target: https://pypi.org/project/sbmlsim/
   :alt: Supported Python Versions

.. image:: https://img.shields.io/pypi/l/sbmlsim.svg
   :target: http://opensource.org/licenses/LGPL-3.0
   :alt: GNU Lesser General Public License 3

.. image:: https://codecov.io/gh/matthiaskoenig/sbmlsim/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/matthiaskoenig/sbmlsim
   :alt: Codecov

.. image:: https://readthedocs.org/projects/sbmlsim/badge/?version=latest
   :target: https://sbmlsim.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3597770.svg
   :target: https://doi.org/10.5281/zenodo.3597770
   :alt: Zenodo DOI

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Black


sbmlsim is a collection of python utilities to simplify simulations with
`SBML <http://www.sbml.org>`__ models implemented on top of
`roadrunner <http://libroadrunner.org/>`__. Source code is available from
`https://github.com/matthiaskoenig/sbmlsim <https://github.com/matthiaskoenig/sbmlsim>`__.

Features include among others

-  simulation experiments
-  simulation reports
-  parameter fitting

The documentation is available on `https://sbmlsim.readthedocs.io <https://sbmlsim.readthedocs.io>`__.
If you have any questions or issues please `open an issue <https://github.com/matthiaskoenig/sbmlsim/issues>`__.


How to cite
===========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3597770.svg
   :target: https://doi.org/10.5281/zenodo.3597770
   :alt: Zenodo DOI

Contributing
============

Contributions are always welcome! Please read the `contributing guidelines
<https://github.com/matthiaskoenig/sbmlsim/blob/develop/.github/CONTRIBUTING.rst>`__ to
get started.

License
=======

* Source Code: `LGPLv3 <http://opensource.org/licenses/LGPL-3.0>`__
* Documentation: `CC BY-SA 4.0 <http://creativecommons.org/licenses/by-sa/4.0/>`__

The sbmlsim source is released under both the GPL and LGPL licenses version 2 or
later. You may choose which license you choose to use the software under.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License or the GNU Lesser General Public
License as published by the Free Software Foundation, either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

Funding
=======
Matthias König is supported by the Federal Ministry of Education and Research (BMBF, Germany)
within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054) 
and by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 
"`QuaLiPerF <https://qualiperf.de>`__ (Quantifying Liver Perfusion-Function Relationship in Complex Resection - 
A Systems Medicine Approach)" by grant number 436883643. Matthias König has received funding from the EOSCsecretariat.eu which has received funding 
from the European Union's Horizon Programme call H2020-INFRAEOSC-05-2018-2019, grant Agreement number 831644.


Installation
============
``sbmlsim`` is available from `pypi <https://pypi.python.org/pypi/sbmlsim>`__ and
can be installed via::

    pip install sbmlsim

Requirements
------------

HDF5 support is required which can be installed on linux via::

    sudo apt-get install -y libhdf5-serial-dev

Develop version
---------------
The latest develop version can be installed via::

    pip install git+https://github.com/matthiaskoenig/sbmlsim.git@develop

Or via cloning the repository and installing via::

    git clone https://github.com/matthiaskoenig/sbmlsim.git
    cd sbmlsim
    pip install -e .

To install for development use::

    pip install -e .[development]
    
© 2019-2020 Matthias König
