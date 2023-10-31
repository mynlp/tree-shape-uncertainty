"""Defines the names of files, keys"""

from pathlib import Path

############
## Logger ##
############


def main_logger_name() -> str:
    return "myLogger"


def get_log_file(
    log_dir: Path,
    dataset_key: str,
    archi: str,
    hparams_key: str,
    seed: int,
    eval_split: str,
) -> Path:
    return log_dir.joinpath(
        dataset_key, archi, hparams_key, f"seed{seed}-{eval_split}.log"
    )


##############
## Datasets ##
##############


def train_tree(dataset_dir: Path):
    return Path.joinpath(dataset_dir, "train.tree")


def dev_tree(dataset_dir: Path):
    return Path.joinpath(dataset_dir, "dev.tree")


def test_tree(dataset_dir: Path):
    return Path.joinpath(dataset_dir, "test.tree")


def automorphism_filename(dataset_dir: Path, seed: int):
    return Path.joinpath(dataset_dir, f"seed-{seed}_automorphism.json")


def dataset_key(dataset_name: str, seed: int):
    return f"{dataset_name}_seed-{seed}"


def get_dataset_key_dir(dataset_dir: Path, dataset_key: str):
    return Path.joinpath(dataset_dir, dataset_key)


# The prefix of preprocessed files for urnng.
def urnng_output_file(dataset_key_dir: Path, batch_size: int):
    return Path.joinpath(dataset_key_dir, f"urnng_bsz-{batch_size}")


##############
## Training ##
##############


def hparam_train_work_dir(
    train_dir: Path, dataset_key: str, archi: str, hparams_key: str, seed: int
) -> Path:
    return train_dir.joinpath(dataset_key, archi, hparams_key, str(seed))


def hparam_out_file(
    train_dir: Path,
    dataset_key: str,
    archi: str,
    hparams_key: str,
    seed: int,
    split_to_parse: str,
) -> Path:
    work_dir: Path = hparam_train_work_dir(
        train_dir=train_dir,
        dataset_key=dataset_key,
        archi=archi,
        hparams_key=hparams_key,
        seed=seed,
    )

    return work_dir.joinpath(f"output-{split_to_parse}.jsonl")
