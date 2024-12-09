#!/bin/bash

output_file="fm_cut_and_count_config.txt"


n_sig=(0)

datasets=(herwig_extended1 herwig_extended2 herwig_extended3)
blocks=(4 4 12)

len=${#datasets[@]}
#frequencies=(20 20 20)
#epochs=(5000 5000 5000)

windows=(1 2 3 4 5 6 7 8 9)
tries=(0)

echo "len: $len"

printf "%-15s %-15s %-15s %-15s %-15s %-15s\n" "ArrayTaskID" "n_sig" "tries" "windows" "dataset" "blocks" > "$output_file"

i=0
for ((k=0; k<$len; k++)); do
    for n in ${n_sig[@]}; do
        for w in ${windows[@]}; do
            for t in ${tries[@]}; do
                printf "%-15s %-15s %-15s %-15s %-15s %-15s\n" $i $n $t $w ${datasets[$k]} ${blocks[$k]} >> "$output_file"
                i=$((i+1))
                #done
            done    
        done
    done    
done
# for f in ${freq[@]}; do
#     for h in ${hidd2[@]}; do
#         for b in ${blocks2[@]}; do
#             #printf "%-15s %-15s %-15s\n" $h $b $f >> "$output_file"
#             printf "%-15s %-15s %-15s %-15s %-15s %-15s\n" $i $h $b $f $batch_size $epochs >> "$output_file"
#             i=$((i+1))
#         done
#     done
# done
# for f in ${freq[@]}; do
#     for h in ${hidd2[@]}; do
#         for b in ${blocks2[@]}; do
        
#     done
# done