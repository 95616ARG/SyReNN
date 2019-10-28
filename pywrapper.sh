#!/bin/bash

if [ -z $INSTALLDIR ]; then
    INSTALLDIR=$PWD/$(find **/** -name "installdir" | head -n 1)
fi
export LD_LIBRARY_PATH=$INSTALLDIR/lib/:$INSTALLDIR/lib64/
$INSTALLDIR/bin/python3.7 $@
