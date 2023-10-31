#!/bin/bash

##############
# Parameters #
##############
save_filepath=./results/plot/branch_gold_trees.png

#####################
# Analyze Gold Data #
#####################

# Parallelization
num_proc=8
#num_proc=1

# Plot for train split.
python -m src.analyze_treebank --corpus_files ./data/cleaned_datasets/english/ptb/ptb_len10_nopunct/train.tree ./data/cleaned_datasets/japanese/ktb/ktb_len10_nopunct/train.tree ./data/cleaned_datasets/english/ptb/ptb_len40_nopunct/train.tree ./data/cleaned_datasets/japanese/ktb/ktb_len40_nopunct/train.tree --corpus_names PTB10 KTB10 PTB40 KTB40 --fill True --fontsize 12 --save_filepath ${save_filepath} --num_proc ${num_proc}