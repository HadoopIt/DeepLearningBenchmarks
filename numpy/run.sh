#!/bin/sh

#To use the numpy/scipy with mkl at lisa you must do:
#export PYTHONPATH=~bastienf/repos/numpy_mkl/lib64/python2.5/site-packages/
#export LD_LIBRARY_PATH=~bastienf/repos/numpy_mkl/lib64/python2.5/site-packages/:/u/bergstrj/pub/intel/mkl/10.2.4.032/lib/em64t

cat /proc/cpuinfo |grep "model name"|uniq > ${HOSTNAME}_config.conf
free >> ${HOSTNAME}_config.conf
uname -a >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy version:",numpy.__version__' >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy config";numpy.__config__.show()' >>  ${HOSTNAME}_config.conf
python -V 2>>  ${HOSTNAME}_config.conf

python mlp.py 784 500 10 1 1000 > ${HOSTNAME}_mlp_1.bmark
python mlp.py 784 500 10 60 100 > ${HOSTNAME}_mlp_60.bmark

python logreg.py 784 10 1 1000 > ${HOSTNAME}_lr_784_1.bmark
python logreg.py 784 10 60 100 > ${HOSTNAME}_lr_784_60.bmark
python logreg.py 32  10 1 1000 > ${HOSTNAME}_lr_32_1.bmark
python logreg.py 32  10 60 100 > ${HOSTNAME}_lr_32_60.bmark
python rbm.py 1024 1024 1 100 > ${HOSTNAME}_rbm_1.bmark
python rbm.py 1024 1024 60 20 > ${HOSTNAME}_rbm_60.bmark
