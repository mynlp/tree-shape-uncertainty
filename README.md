# Tree-shape Uncertainty for Analyzing the Inherent Branching Bias of Unsupervised Parsing Models

This repository provides the experimental codes for the CONLL2023 paper [Tree-shape Uncertainty for Analyzing the Inherent Branching Bias of Unsupervised Parsing Models](URL_to_appear).

# Environment

For example, the conda environment can be created as follows:

```
conda create -n env python=3.10
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy
conda install matplotlib
conda install pyyaml
conda install tqdm
conda install nltk
conda install h5py
```

Detailed version information is available in `requirements.txt`.

# Experiments

## Resources 

- [English Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42)
  - Note that PTB requires LDC account for installation
- [Japanese Keyaki Treebank](https://github.com/ajb129/KeyakiTreebank)

## Data Preprocess

1. Preprocess the installed treebanks using [UnsupConstParseEval](https://github.com/i-lijun/UnsupConstParseEval/tree/master)
2. Generate random automorphisms: `src/gen_automorphism.py`
3. Generate texts that do not contain potential branching bias using the automohrphisms: `src/gen_sym_corpus.py`
4. Generate preprocessed files for URNNG: `src/preprocess_for_urnng.py`

The whole preprocess can be done by executing `script/preprocess.sh`.
We generate 10 datasets by this script.

## Training and Parse

For each dataset generated in the previous step:
1. Train models with the train and dev split: `src/train_many.py`
2. Run the trained models and obtain predicted parses for the train, dev and test splits: `src/train_many.py`

Detailed setups of the trianing and evaluation parts are described in `script/train.sh` and `script/parse.sh`, respectively.
Note that training all models would take very long time (especially URNNG). Appropriate parallelization is required (by default, `script/train.sh` assumes 8 gpus are available).

## Analysis

1. Calculate basic statistics of the generated datasets: `src/dataset_stats.py`
2. Plot histograms of branching directions of gold trees: `src/analyze_treebank.py`
3. Plot the averaged branching directions of the model prediction: `src/analyze_models.py`
4. Plot histograms of branching directions of model predictions: `src/analyze_model_hist.py`

Detailed setups of these analyses are provided in `script/dataset_stats.sh`, `script/analyze_treebank.sh`, `script/analyze_models.sh`, and `script/analyze_model_hist.sh`.

# References

## Data Preprocess

Our data preprocess procedure is mostly based on the script of [UnsupConstParseEval](https://github.com/i-lijun/UnsupConstParseEval/tree/master).

## Models

For the models, we slightly modify the codes distributed by the authors to unify the training emvironment.
This repository includes the modified codes:

- [DIORA](https://github.com/iesl/diora): `src/diora`
  - Originally distributed with Apache License 2.0
- [PRPN](https://github.com/yikangshen/PRPN): `src/prpn`
  - Originally distributed with MIT License
- [URNNG](https://github.com/harvardnlp/urnng): `src/urnng`
  - Originally distributed with MIT License

# License

MIT