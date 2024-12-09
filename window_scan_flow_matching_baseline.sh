#!/bin/bash


#############################################################
#############################################################

# non lin embedding experiments
# nsig=2000 1000
# x_train=data SR CR
# data= baseline and baseline with delta_r

# get all nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TMOUT=180000
# node1=$(echo $nodes | cut -d ' ' -f 1)
# node2=$(echo $nodes | cut -d ' ' -f 2)
# node3=$(echo $nodes | cut -d ' ' -f 3)
# len of nodes
len=$(echo $nodes | wc -w)
echo "length of nodes: ${len}"


tasks_per_node=$SLURM_NTASKS_PER_NODE
total_tasks=$SLURM_NTASKS

echo "Total tasks: ${total_tasks}"
echo "Tasks per node: ${tasks_per_node}"

epochs=2000
hidden_dim=256
frequencies=20
batch_size=4096
num_blocks=4
#frequencies=3


local_task_counter=0
node_counter=0
task_counter=0

# n_sig_list=(1000)
# for window in ${window_index[@]}; do
#     for n_sig in ${n_sig_list[@]}; do
#         for x_train in ${x_train_list[@]}; do

#             wandb_job_type=window_${window}
#             data_dir=data/extended1
#             data_name=base_nsig_${n_sig}
            
        
#             if [ $local_task_counter -eq $tasks_per_node ]; then
#                 node_counter=$((node_counter+1))
#                 local_task_counter=1
#             else
#                 local_task_counter=$((local_task_counter+1))
#             fi

#             if [ $task_counter -eq $total_tasks ]; then
#                 echo "Waiting for tasks to finish"
#                 wait
#                 task_counter=1
#                 node_counter=0
#             else
#                 task_counter=$((task_counter+1))
#             fi
#             node=$(echo $nodes | cut -d ' ' -f $((node_counter+1)))
#             echo "Node counter: ${node_counter}"
#             echo "Node: ${node}"
#             echo "Task: ${local_task_counter}"
#             echo "doing baseline task: ${n_sig} ${x_train} on node: ${node}"

#             srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/window_scan_flow_matching.py --n_sig=${n_sig} \
#                 --epochs=${epochs} --batch_size=${batch_size} \
#                 --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${data_name}_${x_train} \
#                 --frequencies=${frequencies} --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
#                 --num_blocks=${num_blocks} --wandb --device=cuda:0 --baseline --window_index=${window} \
#                 --x_train=${x_train} &>./results/${data_name}_${x_train}_${window}.out &


#         done
#     done
# done

# echo "waiting for tasks to be finished"
# wait
# echo "All tasks are finished"
# exit
# echo "Exiting compute nodes"


#############################################################
#############################################################

task_start=$1
task_end=$2
#task_end=$((task_start+31))
config=fm_cut_and_count_config.txt
task_array=($(seq ${task_start} ${task_end}))
x_train=CR

# for window in ${window_index[@]}; do
#     for n_sig in ${n_sig_list[@]}; do
for i in ${task_array[@]}; do


    echo "task_id: ${i}"
    n_sig=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $config)
    try=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $3}' $config)
    window=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $4}' $config)

    wandb_group=window_scan_ext1_${n_sig}
    wandb_job_type=window_${window}
    data_dir=data/extended1
    wandb_run_name=try_${try}
    #data_name=dR_nsig_${n_sig}
    

    if [ $local_task_counter -eq $tasks_per_node ]; then
        node_counter=$((node_counter+1))
        local_task_counter=1
    else
        local_task_counter=$((local_task_counter+1))
    fi

    if [ $task_counter -eq $total_tasks ]; then
        echo "Waiting for tasks to finish"
        wait
        task_counter=1
        node_counter=0
    else
        task_counter=$((task_counter+1))
    fi
    node=$(echo $nodes | cut -d ' ' -f $((node_counter+1)))
    echo "Node counter: ${node_counter}"
    echo "Node: ${node}"
    echo "Task: ${local_task_counter}"
    echo "doing baseline task: ${n_sig} ${x_train} on node: ${node}"

    srun --nodelist=${node} -n 1 -N 1 --exact --gpus-per-task=1 shifter python -u scripts/window_scan_flow_matching.py --n_sig=${n_sig} \
        --epochs=${epochs} --batch_size=${batch_size} \
        --data_dir=${data_dir} --wandb_group=${wandb_group} --wandb_run_name=${wandb_run_name} \
        --frequencies=${frequencies} --hidden_dim=${hidden_dim} --wandb_job_type=${wandb_job_type} \
        --num_blocks=${num_blocks} --wandb --device=cuda:0 --window_index=${window} \
        --x_train=${x_train} &>./results/${wandb_group}_${wandb_job_type}_${wandb_run_name}.out &

    if [ $i -eq $task_end ]; then
        echo "Waiting for final set of tasks to finish"
        wait
        exit
    fi

done



