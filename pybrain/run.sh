#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib64/atlas/:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/u/lisa/local/install_package/arac/src/python
echo "Using the OS version of atlas (~2-3 slower ?) !!"

python mlp.py
