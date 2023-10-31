import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import nltk

from . import myutils, naming
from .mylogger import main_logger

logger = main_logger.getChild(__name__)


def extract_vocab(dataset_dir: Path) -> set[str]:
    """Extract vocabulary from train, dev, and test data.

    Note that the words from test data must be extracted to ensure the vocab mapping is an automorphism.
    """
    vocab: set[str] = set()

    with naming.train_tree(dataset_dir).open(mode="r") as f:
        for l in f:
            leaves: list[str] = nltk.Tree.fromstring(l).leaves()
            vocab |= set(leaves)

    with naming.dev_tree(dataset_dir).open(mode="r") as f:
        for l in f:
            leaves: list[str] = nltk.Tree.fromstring(l).leaves()
            vocab |= set(leaves)

    with naming.test_tree(dataset_dir).open(mode="r") as f:
        for l in f:
            leaves: list[str] = nltk.Tree.fromstring(l).leaves()
            vocab |= set(leaves)

    return vocab


def gen_vocab_automorphism(vocab: set[str], seed: int) -> dict[str, str]:
    """Randomly generate order 2 automorphism."""

    # Set random seed.
    myutils.set_random_seed(seed)

    # Randomly shuffle.
    vocab_list: list[str] = random.sample(list(vocab), len(vocab))

    # Split the vocabulary into two parts.
    automorphism: dict[str, str] = {}

    if len(vocab_list) % 2 != 0:
        # If the vocabulary size is an odd number, just take one and map it to itself.
        # Since vocab_list is shuffled, the one is taken randomly.
        w: str = vocab_list.pop()
        automorphism[w] = w

    vocab_left: list[str] = vocab_list[: int(len(vocab_list) / 2)]
    vocab_right: list[str] = vocab_list[int(len(vocab_list) / 2) :]
    assert len(vocab_left) == len(vocab_right)

    for x, y in zip(vocab_left, vocab_right):
        # Make an order 2 mapping.
        automorphism[x] = y
        automorphism[y] = x

    # Check if the generated automorphism is really order 2.
    for w in vocab:
        assert automorphism[automorphism[w]] == w

    return automorphism


@dataclass
class Args:
    dataset_dirs: list[str]
    random_seeds: list[int]


if __name__ == "__main__":
    # Set args.
    parser: argparse.ArgumentParser = myutils.gen_parser(Args, required=True)
    args: Args = Args(**myutils.parse_and_filter(parser, sys.argv[1:]))

    for dirname in args.dataset_dirs:
        logger.info(f"Extracting vocabulary from {dirname}")
        vocab: set[str] = extract_vocab(dataset_dir=Path(dirname))

        for seed in args.random_seeds:
            logger.info(f"Generating random automorphism with seed {seed}")
            automorphism: dict[str, str] = gen_vocab_automorphism(
                vocab=vocab, seed=seed
            )

            # Save the automorphism as json.
            myutils.to_json(
                automorphism,
                filepath=naming.automorphism_filename(
                    dataset_dir=Path(dirname), seed=seed
                ),
            )
