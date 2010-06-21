#!/bin/bash

cat /proc/cpuinfo |grep "model name"|uniq > ${HOSTNAME}_config.conf
free >> ${HOSTNAME}_config.conf
uname -a >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy version:",numpy.__version__' >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy config";numpy.__config__.show()' >>  ${HOSTNAME}_config.conf
python -c 'import scipy;print "Scipy version:",scipy.__version__' >>  ${HOSTNAME}_config.conf
python -c 'import scipy;print "Scipy config";scipy.__config__.show()' >>  ${HOSTNAME}_config.conf
python -V 2>>  ${HOSTNAME}_config.conf


TEST_CPU=1
TEST_GPU=0
COND=${1:-"CPU"}

if [ "$COND" == "CPU/GPU" ] ; then
    TEST_CPU=1
    TEST_GPU=1
fi

if [ "$COND" == "GPU" ] ; then
    TEST_CPU=0
    TEST_GPU=1
fi

# FOR MAGGIE I INSTALLED MKL SO DO LIKE THIS:
# LD_LIBRARY_PATH to include     /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t 
# LIBRARY_PATH to include        /u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t
# THEANO_FLAGS="device=cpu,floatX=float64,blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def -lpthread" python mlp.py
#MKL32='linker=c|py_nogc,device=cpu,floatX=float32,blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def'
#MKL64='linker=c|py_nogc,device=cpu,floatX=float64,blas.ldflags=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_def'
GPU32='linker=c|py_nogc,device=gpu0,floatX=float32'
NUMPYBLAS32='linker=c|py_nogc,device=cpu,floatX=float32,blas.ldflags='
NUMPYBLAS64='linker=c|py_nogc,device=cpu,floatX=float64,blas.ldflags='

#THEANO_FLAGS="$MKL32" python mlp.py
#THEANO_FLAGS="$MKL64" python mlp.py

#THEANO_FLAGS="$MKL32" python convnet.py
#THEANO_FLAGS="$MKL64" python convnet.py
#THEANO_FLAGS="$GPU32" python convnet.py


#THEANO_FLAGS="$MKL32" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_cpu32_b1.bmark
#THEANO_FLAGS="$MKL32" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_cpu32_b60.bmark

#THEANO_FLAGS="$MKL64" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_cpu64_b1.bmark
#THEANO_FLAGS="$MKL64" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_cpu64_b60.bmark

#THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_gpu32_b1.bmark
#THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_gpu32_b60.bmark


if [ $TEST_CPU == 1  ] ; then

THEANO_FLAGS="$NUMPYBLAS32" python mlp.py
THEANO_FLAGS="$NUMPYBLAS64" python mlp.py

THEANO_FLAGS="$NUMPYBLAS32" python convnet.py
THEANO_FLAGS="$NUMPYBLAS64" python convnet.py

THEANO_FLAGS="$NUMPYBLAS32" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_cpu32_numpyblas_b1.bmark
THEANO_FLAGS="$NUMPYBLAS32" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_cpu32_numpyblas_b60.bmark

THEANO_FLAGS="$NUMPYBLAS64" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_cpu64_numpyblas_b1.bmark
THEANO_FLAGS="$NUMPYBLAS64" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_cpu64_numpyblas_b60.bmark

fi
## GPU TEST:

if [ $TEST_GPU == 1  ] ; then
  THEANO_FLAGS="$GPU32" python mlp.py

  THEANO_FLAGS="$GPU32" python convnet.py

  THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_cpu32_numpyblas_b1.bmark
  THEANO_FLAGS="$GPU32" python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_cpu32_numpyblas_b60.bmark
fi
