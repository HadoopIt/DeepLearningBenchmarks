#!/bin/sh

cat /proc/cpuinfo |grep "model name"|uniq > ${HOSTNAME}_config.conf
free >> ${HOSTNAME}_config.conf
uname -a >>  ${HOSTNAME}_config.conf
nvidia-smi -a >> ${HOSTNAME}_config.conf
matlab -nodisplay -nojvm -r get_info >> ${HOSTNAME}_config.conf


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


if [ $TEST_CPU == 1  ] ; then
    matlab -nodisplay -nojvm -r "num_mlp('$HOSTNAME')"
fi

if [ $TEST_GPU == 1 ] ; then
    matlab -nodisplay -nojvm -r "gpu_num_mlp('$HOSTNAME')"
fi

