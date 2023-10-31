#!/bin/bash

##############
# Parameters #
##############

# Where the repository https://github.com/i-lijun/UnsupConstParseEval/tree/master is saved.
# This should not have '/' at the end.
unsupconstparseeval_dir=$1

path_to_raw_ptb=$2
path_to_raw_ktb=$3

morphism_random_seeds_small=$(seq 0 2)
morphism_random_seeds_large=$(seq 0 1)

urnng_batch_size=16

dataset_dir=./data/preprocessed

#######################
# Preprocess Treebank #
#######################

# The preprocessed files will be in ./data/cleaned_datasets/
# Both nopunct and puct versions are generated, but only nopunct is used in the following steps.
python ${unsupconstparseeval_dir}/preprocess.py --path_to_raw_ptb ${path_to_raw_ptb} --path_to_raw_ktb ${path_to_raw_ktb}

# Modify the file names: dev -> dev.tree
# Ad hoc
for dataset in english/ptb/ptb_len japanese/ktb/ktb_len; do
    for len in 10 40; do
        for split in train dev test; do
            # Only use nopunct
            mv ./data/cleaned_datasets/${dataset}${len}_nopunct/${split} ./data/cleaned_datasets/${dataset}${len}_nopunct/${split}.tree
        done
    done
done

##########################
# Generate Automorphisms #
##########################

small_dataset_dirs="
./data/cleaned_datasets/english/ptb/ptb_len10_nopunct
./data/cleaned_datasets/japanese/ktb/ktb_len10_nopunct
"
small_dataset_names="
ptb_len10_nopunct
ktb_len10_nopunct
"

large_dataset_dirs="
./data/cleaned_datasets/english/ptb/ptb_len40_nopunct
./data/cleaned_datasets/japanese/ktb/ktb_len40_nopunct
"
large_dataset_names="
ptb_len40_nopunct
ktb_len40_nopunct
"

python -m src.gen_automorphism --dataset_dirs ${small_dataset_dirs} --random_seeds ${morphism_random_seeds_small}
python -m src.gen_automorphism --dataset_dirs ${large_dataset_dirs} --random_seeds ${morphism_random_seeds_large}

###########################
# Generate Unbiased Texts #
###########################

python -m src.gen_sym_corpus --dataset_dirs ${small_dataset_dirs} --dataset_names ${small_dataset_names} --random_seeds ${morphism_random_seeds_small} --output_dir ${dataset_dir}
python -m src.gen_sym_corpus --dataset_dirs ${large_dataset_dirs} --dataset_names ${large_dataset_names} --random_seeds ${morphism_random_seeds_large} --output_dir ${dataset_dir}

#########################################
# Generate Preprocessed Files for URNNG #
#########################################

# Ad hoc.
dataset_key_dirs=$(find ${dataset_dir} -maxdepth 1 -mindepth 1)
python -m src.preprocess_for_urnng --dataset_key_dirs ${dataset_key_dirs} --batch_size ${urnng_batch_size}