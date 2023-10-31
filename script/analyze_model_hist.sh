#!/bin/bash

##############
# Parameters #
##############

train_dir=./results/train
save_dir=./results/plot/hist_plot

splits="train dev test"

num_proc=20
#num_proc=1

dataset_names_small="ptb_len10_nopunct ktb_len10_nopunct"
dataset_names_large="ptb_len40_nopunct ktb_len40_nopunct"

dataset_bases_small="PTB10 KTB10"
dataset_bases_large="PTB40 KTB40"

morphism_seeds_small=$(seq 0 2)
morphism_seeds_large=$(seq 0 1)

diora_hparams_key=419c277467be02256956a299c1b7bfb3e72df4722f57fcc6f4ff8240eb6ff9f7
prpn_hparams_key=8bc262d649755807fb1d911bd3a2c16a6a852b99925bf375f7ad12b4aed26fcf
urnng_hparams_key=f6c98f232e4f87c4391876a5bcaa64662fa4d92fbd60b33f89457b9a9845503d

train_seeds=$(seq 0 14)

# ptb10(m0, 1, 2) train, dev, test, ktb10(m0, 1, 2) train, dev, test, ptb40(m0, 1) train, dev, test, ktb40(m0, 1) train, dev, test
row_num_datasets="3 3 3 3 3 3 2 2 2 2 2 2"

# Add for small.
dataset_key_l=""
dataset_train_seeds=""
for dataset_name in ${dataset_names_small}; do
        for morphism_seed in ${morphism_seeds_small}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
                dataset_train_seeds="${dataset_train_seeds} 15"
        done
done

# Add for large.
for dataset_name in ${dataset_names_large}; do
        for morphism_seed in ${morphism_seeds_large}; do
                dataset_key_l="${dataset_key_l} ${dataset_name}_seed-${morphism_seed}"
                dataset_train_seeds="${dataset_train_seeds} 15"
        done
done

# Set row names.
row_names=""
# Add for small.
for dataset_base in ${dataset_bases_small}; do
        for split in ${splits}; do
                row_names="${row_names}"' $D^{\mathrm{'${dataset_base}'}}_{*}-'${split}'$'

        done
done

# Add for large.
for dataset_base in ${dataset_bases_large}; do
        for split in ${splits}; do
                row_names="${row_names}"' $D^{\mathrm{'${dataset_base}'}}_{*}-'${split}'$'

        done
done

# DIORA
function plot_diora(){
        echo start for DIORA: $(date)
        singularity exec --pwd ${CURRENT_DIR} ${SING_IMG} python -m src.analyze_model_hist --train_dir ${train_dir} --archi_l diora --archi_name_l DIORA --hparams_key_l ${diora_hparams_key} --full_train_random_seed_l ${train_seeds} --dataset_key_l ${dataset_key_l} --dataset_name_l ${dataset_key_l} --dataset_train_seed_l ${dataset_train_seeds} --splits ${splits} --row_names ${row_names} --row_num_datasets ${row_num_datasets} --num_proc ${num_proc} --fill False --save_filepath ${save_dir}/hist-diora.png --colors magenta
        echo finish for DIORA: $(date)
}

# PRPN
function plot_prpn(){
        echo start for PRPN: $(date)
        singularity exec --pwd ${CURRENT_DIR} ${SING_IMG} python -m src.analyze_model_hist --train_dir ${train_dir} --archi_l prpn --archi_name_l PRPN --hparams_key_l ${prpn_hparams_key} --full_train_random_seed_l ${train_seeds} --dataset_key_l ${dataset_key_l} --dataset_name_l ${dataset_key_l} --dataset_train_seed_l ${dataset_train_seeds} --splits ${splits} --row_names ${row_names} --row_num_datasets ${row_num_datasets} --num_proc ${num_proc} --fill False --save_filepath ${save_dir}/hist-prpn.png --colors tab:brown
        echo finish for PRPN: $(date)
}

# URNNG
function plot_urnng(){
        echo start for URNNG: $(date)
        singularity exec --pwd ${CURRENT_DIR} ${SING_IMG} python -m src.analyze_model_hist --train_dir ${train_dir} --archi_l urnng --archi_name_l URNNG --hparams_key_l ${urnng_hparams_key} --full_train_random_seed_l ${train_seeds} --dataset_key_l ${dataset_key_l} --dataset_name_l ${dataset_key_l} --dataset_train_seed_l ${dataset_train_seeds} --splits ${splits} --row_names ${row_names} --row_num_datasets ${row_num_datasets} --num_proc ${num_proc} --fill False --save_filepath ${save_dir}/hist-urnng.png --colors tab:green
        echo finish for URNNG: $(date)
}

# Parallel plot.
# Prallelization may require huge resources.
plot_diora &
plot_prpn &
plot_urnng &
wait