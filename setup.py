#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
sbmlutils pip package
"""
import io
import re
import os
from setuptools import find_packages
from setuptools import setup

setup_kwargs = {}


def read(*names, **kwargs):
    """ Read file info in correct encoding. """
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


# version from file
verstrline = read('sbmlsim/_version.py')
mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", verstrline, re.M)
if mo:
    verstr = mo.group(1)
    setup_kwargs['version'] = verstr
else:
    raise RuntimeError("Unable to find version string")

# description from markdown
long_description = read('README.rst')
setup_kwargs['long_description'] = long_description

setup(
    name='sbmlsim',
    description='SBML simulation made easy',
    url='https://github.com/matthiaskoenig/sbmlutils',
    author='Matthias König',
    author_email='konigmatt@googlemail.com',
    license='LGPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='SBML, simulation',
    packages=find_packages(),
    # package_dir={'': ''},
    package_data={
      '': ['../requirements.txt'],
    },
    include_package_data=True,
    zip_safe=False,
    # List run-time dependencies here.  These will be installed by pip when
    install_requires=[
        "pip>=19.1.1",
        "numpy>=1.17.2",
        "scipy>=1.2.1",
        "matplotlib>=3.1.1",
        "pandas>=0.25.1",
        "tables>0.3.6",
        "python-libsbml-experimental>=5.18.1",
        "libroadrunner>=1.5.4",
        "psutil",
        "ray",
        "pytest>=4.3.1",
        "pytest-cov>=2.6.1",
    ],
    extras_require={},
    **setup_kwargs)
