#!/bin/bash
###############################################################
# Build script for tellurium documentation from rst files and 
# python docstrings in the tellurium package
#
# execute this script in the docs folder i.e., after
# 	cd tellurium/docs
#
# Usage: 
#	./make_docs.sh 2>&1 | tee ./make_docs.log
#
# The documentation is written in docs/_build
###############################################################
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

date
echo "--------------------------------------"
echo "remove old documentation"
echo "--------------------------------------"
rm -rf _built
rm -rf _templates
rm -rf _notebooks

echo "--------------------------------------"
echo "convert notebooks to rst"
echo "--------------------------------------"
NBDIR=$DIR/notebooks

# convert the notebooks to rst after running headlessly
# if errors should abort, remove the --allow-errors option
jupyter nbconvert --ExecutePreprocessor.timeout=600 --to=rst --allow-errors --execute $NBDIR/*.ipynb

echo "--------------------------------------"
echo "postprocessing notebooks rst"
echo "--------------------------------------"
# remove the following lines from the documentation
sed -i '/%load_ext autoreload/d' $NBDIR/*.rst
sed -i '/%autoreload 2/d' $NBDIR/*.rst

# change the image locations (FIXME)
sed -i -- 's/.. image:: /.. image:: notebooks\/docs\//g' ./*.rst


echo "--------------------------------------"
echo "create html docs"
echo "--------------------------------------"
cd $DIR
# make html
sphinx-build -b html . _build/html

# open documentation
firefox _build/html/index.html

