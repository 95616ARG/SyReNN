#!/bin/bash

if [ -z $INSTALLDIR ]; then
    INSTALLDIR=$PWD/$(find **/** -name "installdir" | head -n 1)
fi
export LD_LIBRARY_PATH=$INSTALLDIR/lib/:$INSTALLDIR/lib64/
# The GRPC Bazel rules use the old pip_import syntax which pollutes the
# PYTHONPATH with dependencies that don't align with this version of Python.
# This command strips all dependencies loaded with pip_import, leaving those
# loaded from requirements.txt as long as:
# 1. The current directory path does not include ``pypi''
# 2. None of the actual requirements include ``pypi'' in their name.
export PYTHONPATH=$(\
  echo $PYTHONPATH | \
  sed -e "s/:[^:]*pypi[^:]*//g" \
      -e "s/[^:]*pypi[^:]*://g")
$INSTALLDIR/bin/python3.7 $@
