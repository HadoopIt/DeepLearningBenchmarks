#!/bin/sh

cat /proc/cpuinfo |grep "model name"|uniq > ${HOSTNAME}_config.conf
free >> ${HOSTNAME}_config.conf
uname -a >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy version:",numpy.__version__' >>  ${HOSTNAME}_config.conf
python -c 'import numpy;print "Numpy config";numpy.__config__.show()' >>  ${HOSTNAME}_config.conf
python -c 'import scipy;print "Scipy version:",scipy.__version__' >>  ${HOSTNAME}_config.conf
python -c 'import scipy;print "Scipy config";scipy.__config__.show()' >>  ${HOSTNAME}_config.conf
python -V 2>>  ${HOSTNAME}_config.conf


python convnet.py