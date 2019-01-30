#!/bin/bash

gpucount=8
num_node=2
num_servers=${num_node}
root_dir="/path/to/simpledet"
sync_dir="/tmp/simpledet_sync"

hostfile=hostfile.txt
conffile=faster_r50v2c4_c5_256roi_1x
singularity_image=simpledet.img

export DMLC_INTERFACE=eth0
python -u ../../launcher/tools/launch.py \
    -n ${num_node} \
    --num-servers ${num_servers} \
    --sync-dst-dir ${sync_dir} \
    --launcher ssh -H ${hostfile} \
    scripts/dist_worker.sh ${root_dir} ${singularity_image} ${conffile} \
    2>&1 | tee -a ${root_dir}/log/${conffile}.log
