#!/bin/bash

export LD_LIBRARY_PATH=/u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t:/u/pascanur/arac:/u/pascanur/gtest/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/u/pascanur/arac/src/python

python mlp.py
