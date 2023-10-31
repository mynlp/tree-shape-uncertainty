import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import myutils, naming
from .mylogger import main_logger

logger = main_logger.getChild(__name__)


@dataclass
class Args:
    dataset_key_dirs: list[str]
    batch_size: int


if __name__ == "__main__":
    # Set args.
    parser: argparse.ArgumentParser = myutils.gen_parser(Args, required=True)
    args: Args = Args(**myutils.parse_and_filter(parser, sys.argv[1:]))

    for dataset_key_dirpath in args.dataset_key_dirs:
        logger.info(f"Preprocess for {dataset_key_dirpath}")

        dataset_key_dir: Path = Path(dataset_key_dirpath)

        train_file: Path = naming.train_tree(dataset_dir=dataset_key_dir)
        dev_file: Path = naming.dev_tree(dataset_dir=dataset_key_dir)
        test_file: Path = naming.test_tree(dataset_dir=dataset_key_dir)

        urnng_outputfile = naming.urnng_output_file(
            dataset_key_dir=dataset_key_dir, batch_size=args.batch_size
        )

        preprocess_command_str: str = " ".join(
            [
                f"python preprocess.py",
                f"--trainfile {train_file.resolve()}",
                f"--valfile {dev_file.resolve()}",
                f"--testfile {test_file.resolve()}",
                f"--outputfile {urnng_outputfile.resolve()}",
                f"--batchsize {args.batch_size}",
                # ad hoc.
                f"--seqlength 200",
            ]
        )

        command_str: str = f"cd src/urnng; {preprocess_command_str}"

        try:
            p = subprocess.run(
                command_str,
                shell=True,
                stderr=subprocess.STDOUT,
            )
            p.check_returncode()
        except Exception as e:
            print(e, file=sys.stderr)
