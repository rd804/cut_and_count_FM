#!/bin/bash

# create array from 0 to 179 in intervals of 32
batch_size=16
submit_arrays=($(seq 0 $batch_size 35))

for i in ${submit_arrays[@]}; do
    start=$i
    end=$((i+batch_size-1))
    echo "Submitting job for task_id: ${start} to ${end}"
    salloc --qos interactive --time=4:00:00 --nodes=4 --account=m4539 --gres=gpu:4 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=4 bash window_scan_flow_matching_extended.sh ${start} ${end}
done
# salloc --qos interactive --time=4:00:00 --nodes=4 --account=m4539 --gres=gpu:4 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=4 bash script.sh 0
# salloc --qos interactive --time=4:00:00 --nodes=4 --account=m4539 --gres=gpu:4 --constraint=gpu --image rd804/ranode_llf:latest --ntasks-per-node=4 bash script.sh 64

