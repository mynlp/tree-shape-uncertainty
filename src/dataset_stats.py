import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nltk

from . import myutils, naming
from .mylogger import main_logger

logger = main_logger.getChild(__name__)


def get_stats(dataset_dir: Path):
    logger.info(f"Start analyzing the statistics of {str(dataset_dir)}")

    train_vocab: set[str] = set()

    with naming.train_tree(dataset_dir).open(mode="r") as f:
        num_data_train: int = 0
        for l in f:
            leaves: list[str] = nltk.Tree.fromstring(l).leaves()
            train_vocab |= set(leaves)

            num_data_train += 1

        logger.info(f"Train size: {num_data_train}")

    with naming.dev_tree(dataset_dir).open(mode="r") as f:
        num_data_dev: int = 0
        for l in f:
            num_data_dev += 1
        logger.info(f"Dev size: {num_data_dev}")

    with naming.test_tree(dataset_dir).open(mode="r") as f:
        num_data_test: int = 0
        for l in f:
            num_data_test += 1
        logger.info(f"Test size: {num_data_test}")

    logger.info(f"Train vocab size: {len(train_vocab)}")


@dataclass
class Args:
    dataset_dirs: list[str]


if __name__ == "__main__":
    # Set args.
    parser: argparse.ArgumentParser = myutils.gen_parser(Args, required=True)
    args: Args = Args(**myutils.parse_and_filter(parser, sys.argv[1:]))

    for dataset_dir in args.dataset_dirs:
        get_stats(Path(dataset_dir))
