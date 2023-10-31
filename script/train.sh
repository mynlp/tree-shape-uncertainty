#!/bin/bash

##############
# Parameters #
##############

dataset_dir=./data/preprocessed/
train_dir=./results/train/
log_dir=./log/train/


use_gpu=True
#use_gpu=False
n_parallel=8
gpu_ids="0 1 2 3 4 5 6 7"

force_update=False
#force_update=True

morphism_random_seeds_small=$(seq 0 2)
morphism_random_seeds_large=$(seq 0 1)

archi_l="diora prpn urnng"
train_seeds=$(seq 0 14)


#########
# Train #
#########

small_dataset_names="
ptb_len10_nopunct
ktb_len10_nopunct
"
large_dataset_names="
ptb_len40_nopunct
ktb_len40_nopunct
"

# Add small.
dataset_key_l=""
for dataset_name in ${small_dataset_names}; do
        for morphism_seed in ${morphism_random_seeds_small}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
        done
done

# Add large.
for dataset_name in ${large_dataset_names}; do
        for morphism_seed in ${morphism_random_seeds_large}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
        done
done

# Note that all the datasets (specified by dataset_kay) are assumed to be placed in the same directory dataset_dir.

python -m src.train_many --dataset_key_l ${dataset_key_l} --archi_l ${archi_l} --train_random_seeds ${train_seeds} --n_parallel ${n_parallel} --use_gpu ${use_gpu} --dataset_dir ${dataset_dir} --train_dir ${train_dir} --log_dir ${log_dir} --force_update ${force_update} --debug False --gpu_ids ${gpu_ids} --only_eval False