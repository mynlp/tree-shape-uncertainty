#!/bin/bash

##############
# Parameters #
##############

dataset_dir=./data/preprocessed

###########
# Analyze #
###########

small_dataset_names="
ptb_len10_nopunct
ktb_len10_nopunct
"
large_dataset_names="
ptb_len40_nopunct
ktb_len40_nopunct
"

# Add small.
dataset_dirs=""
for dataset_name in ${small_dataset_names}; do
        for morphism_seed in ${morphism_random_seeds_small}; do
                dataset_dirs="${dataset_dirs} ${dataset_dir}/${dataset_name}_seed-${morphism_seed}"
        done
done

# Add large.
for dataset_name in ${large_dataset_names}; do
        for morphism_seed in ${morphism_random_seeds_large}; do
                dataset_dirs="${dataset_dirs} ${dataset_dir}/${dataset_name}_seed-${morphism_seed}"
        done
done

python -m src.dataset_stats --dataset_dirs ${dataset_dirs}