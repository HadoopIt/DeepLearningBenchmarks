#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib64/atlas/:$LD_LIBRARY_PATH
echo "Using the OS version of atlas (~2-3 slower ?) !!"
THEANO_FLAGS=device=cpu,floatX=float64 python mlp_OS_atlas.py
THEANO_FLAGS=device=cpu,floatX=float32 python mlp_OS_atlas.py
THEANO_FLAGS=device=gpu0,floatX=float32 python mlp_OS_atlas.py
