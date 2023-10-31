#!/bin/bash

##############
# Parameters #
##############

save_filepath_long=./results/plot/all_model_plot_long.png
save_filepath_short=./results/plot/all_model_plot_short.png

train_dir=./results/train/

num_proc=20
#num_proc=1

dataset_names_small="ptb_len10_nopunct ktb_len10_nopunct"
dataset_names_large="ptb_len40_nopunct ktb_len40_nopunct"

dataset_bases_small="PTB10 KTB10"
dataset_bases_large="PTB40 KTB40"

morphism_seeds_small=$(seq 0 2)
morphism_seeds_large=$(seq 0 1)

row_num_datasets="1 1 1 1"
row_num_morphisms="3 3 2 2"

full_train_seeds=$(seq 0 14)
num_seed_small=15
num_seed_large=15

archi_l="diora prpn urnng"
archi_name_l="DIORA PRPN URNNG"

# These are the hyperparameters for diora, prpn, and urnng. They can be obtained by looking at the directory names in the results.
hparams_key_l="
419c277467be02256956a299c1b7bfb3e72df4722f57fcc6f4ff8240eb6ff9f7
8bc262d649755807fb1d911bd3a2c16a6a852b99925bf375f7ad12b4aed26fcf
f6c98f232e4f87c4391876a5bcaa64662fa4d92fbd60b33f89457b9a9845503d
"

maxlen=10

#################
# Plot for Long #
#################

dataset_key_l=""
dataset_train_seed_l=""

# Add for small.
for dataset_name in ${dataset_names_small}; do
        for morphism_seed in ${morphism_seeds_small}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
                dataset_train_seed_l="${dataset_train_seed_l} ${num_seed_small}"
        done
done

# Add for large.
for dataset_name in ${dataset_names_large}; do
        for morphism_seed in ${morphism_seeds_large}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
                dataset_train_seed_l="${dataset_train_seed_l} ${num_seed_large}"
        done
done

# Set the yticks.
dataset_name_l=""
# Add for small.
for dataset_base in ${dataset_bases_small}; do
        for morphism_seed in ${morphism_seeds_small}; do
                dataset_name_l="${dataset_name_l}"' $D^{\mathrm{'${dataset_base}'}}_{\phi_'${morphism_seed}'}$'
        done
done

# Add for large.
for dataset_base in ${dataset_bases_large}; do
        for morphism_seed in ${morphism_seeds_large}; do
                dataset_name_l="${dataset_name_l}"' $D^{\mathrm{'${dataset_base}'}}_{\phi_'${morphism_seed}'}$'
        done
done

# This will plot for the test split.
# raw_tick option is necessary for using latex style formulae.
# maxlen is not set, thus not restricting the length.
python -m src.analyze_models --archi_l ${archi_l} --dataset_key_l ${dataset_key_l} --hparams_key_l ${hparams_key_l} --dataset_train_seed_l ${dataset_train_seed_l} --full_train_random_seed_l ${full_train_seeds} --archi_name_l ${archi_name_l} --dataset_name_l ${dataset_name_l} --save_filepath ${save_filepath_long} --train_dir ${train_dir} --row_num_datasets ${row_num_datasets} --row_num_morphisms ${row_num_morphisms} --raw_tick True --num_proc ${num_proc} --alpha 0.5 --markersize 7.5


##################
# Plot for short #
##################

# maxlen is set, filtering the inputs by length.
python -m src.analyze_models --archi_l ${archi_l} --dataset_key_l ${dataset_key_l} --hparams_key_l ${hparams_key_l} --dataset_train_seed_l ${dataset_train_seed_l} --full_train_random_seed_l ${full_train_seeds} --archi_name_l ${archi_name_l} --dataset_name_l ${dataset_name_l} --save_filepath ${save_filepath_short} --train_dir ${train_dir} --row_num_datasets ${row_num_datasets} --row_num_morphisms ${row_num_morphisms} --raw_tick True --num_proc ${num_proc} --alpha 0.5 --markersize 7.5 --maxlen ${maxlen}