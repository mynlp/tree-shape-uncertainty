import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nltk

from . import myutils, naming
from .mylogger import main_logger

logger = main_logger.getChild(__name__)


def flip_tree(t):
    """Recursively flip a given tree."""
    if not isinstance(t, nltk.Tree):
        return
    else:
        for c in t:
            flip_tree(c)
        t.reverse()


def get_sym_texts(f, vocab_automorphism: dict[str, str]) -> list[str]:
    """Given opened file stream, get symmetric corpus by Z \cup \phi(Z^-1)."""
    sym_corpus: list[str] = []

    for l in f:
        # First, add the original text.
        sym_corpus.append(l.rstrip())

        # Then, map the words and flip.
        t = nltk.Tree.fromstring(l)

        leaves: list[str] = t.leaves()
        for i in range(len(leaves)):
            # Obtain original word.
            wi: str = leaves[i]

            # Apply automorphism.
            tp = t.leaf_treeposition(i)
            t[tp] = vocab_automorphism[wi]

        flip_tree(t)
        # Ad hoc technique to print without any '\n'
        converted_line: str = t.pformat(margin=sys.maxsize)

        sym_corpus.append(converted_line)

    return sym_corpus


def gen_symmetric_corpus(
    original_dataset_dir: Path, automorphism_filepath: Path, output_dataset_dir: Path
):
    """Create symmetric corpus by Z \cup \phi(Z^-1)."""
    vocab_automorphism: dict[str, str] = myutils.from_json(automorphism_filepath)

    with naming.train_tree(dataset_dir=original_dataset_dir).open(mode="r") as f:
        sym_train: list[str] = get_sym_texts(f, vocab_automorphism)

        with naming.train_tree(dataset_dir=output_dataset_dir).open(mode="w") as g:
            g.write("\n".join(sym_train))

    with naming.dev_tree(dataset_dir=original_dataset_dir).open(mode="r") as f:
        sym_dev: list[str] = get_sym_texts(f, vocab_automorphism)

        with naming.dev_tree(dataset_dir=output_dataset_dir).open(mode="w") as g:
            g.write("\n".join(sym_dev))

    with naming.test_tree(dataset_dir=original_dataset_dir).open(mode="r") as f:
        sym_test: list[str] = get_sym_texts(f, vocab_automorphism)

        with naming.test_tree(dataset_dir=output_dataset_dir).open(mode="w") as g:
            g.write("\n".join(sym_test))


@dataclass
class Args:
    dataset_dirs: list[str]
    dataset_names: list[str]
    random_seeds: list[int]


@dataclass
class OptionalArgs:
    output_dir: str = "./data/preprocessed"


if __name__ == "__main__":
    # Set args.
    parser: argparse.ArgumentParser = myutils.gen_parser(Args, required=True)
    args: Args = Args(**myutils.parse_and_filter(parser, sys.argv[1:]))

    opt_parser: argparse.ArgumentParser = myutils.gen_parser(
        OptionalArgs, required=False
    )
    opt_args: OptionalArgs = OptionalArgs(
        **myutils.parse_and_filter(opt_parser, sys.argv[1:])
    )

    assert len(args.dataset_dirs) == len(args.dataset_names)

    for dirname, dataset_name in zip(args.dataset_dirs, args.dataset_names):
        original_dataset_dir: Path = Path(dirname)

        for seed in args.random_seeds:
            dataset_key: str = naming.dataset_key(dataset_name=dataset_name, seed=seed)
            logger.info(f"Generating symmetric corpus for {dataset_key}")

            automorphism_filepath: Path = naming.automorphism_filename(
                dataset_dir=original_dataset_dir, seed=seed
            )
            output_dataset_dir: Path = naming.get_dataset_key_dir(
                dataset_dir=Path(opt_args.output_dir), dataset_key=dataset_key
            )
            output_dataset_dir.mkdir(parents=True, exist_ok=True)

            gen_symmetric_corpus(
                original_dataset_dir=original_dataset_dir,
                automorphism_filepath=automorphism_filepath,
                output_dataset_dir=output_dataset_dir,
            )
