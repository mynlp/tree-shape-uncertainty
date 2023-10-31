import argparse
import hashlib
import json
import pathlib
import random
from typing import Any, Generator, Iterable

import numpy as np
import torch
import yaml


def set_random_seed(random_seed: int):
    """Set the random seeds to the given value."""
    random.seed(random_seed)

    torch.manual_seed(random_seed)

    np.random.seed(random_seed)


def to_yaml(data: Any, filepath: pathlib.Path) -> None:
    with filepath.open(mode="w") as f:
        yaml.dump(data, f)


def from_yaml(filepath: pathlib.Path) -> Any:
    with filepath.open(mode="r") as f:
        return yaml.safe_load(f)


def to_json(data: Any, filepath: pathlib.Path) -> None:
    with filepath.open(mode="w") as f:
        # json.dump(data, f, indent="\t")
        json.dump(data, f)


def from_json(filepath: pathlib.Path) -> Any:
    with filepath.open(mode="r") as f:
        return json.load(f)


def batch_iterable(i: Iterable[Any], n: int) -> Generator[list[Any], None, None]:
    """Generates lists of n elements taken from given iterable.

    If the remaining elements are less than n, all the remaining elements are returned.

    Args:
        i:
            Given iterable.
        n:
            Number of elements to take each time.
    """
    count = 0
    batch: list[Any] = []
    for x in i:
        batch.append(x)
        count += 1
        if count == n:
            count = 0
            tmp = batch
            batch = []
            yield tmp
    if count != 0:
        yield batch


def get_hashed_key(s: str) -> str:
    """Use sha 256 to encode given string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def str2bool(s: str) -> bool:
    match s:
        case "True":
            return True
        case "False":
            return False
        case _:
            raise Exception("Invalid boolean string: {}".format(s))


def gen_parser(cls: type, required: bool) -> argparse.ArgumentParser:
    """Generates a field parser based on annotations.

    All the args will have same optionality. Boolean field will be converted to explicit argument like "--x True --y False".
    """
    value_types = [int, str, float]
    list_types = [list[t] for t in value_types]

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for key, val_type in cls.__annotations__.items():
        if val_type in value_types:
            parser.add_argument("--{}".format(key), type=val_type, required=required)
        elif val_type == bool:
            parser.add_argument("--{}".format(key), type=str2bool, required=required)

        elif val_type in list_types:
            parser.add_argument(
                "--{}".format(key),
                type=val_type.__args__[0],
                nargs="*",
                required=required,
            )

        elif val_type == list[bool]:
            parser.add_argument(
                "--{}".format(key),
                type=str2bool,
                nargs="*",
                required=required,
            )

        else:
            raise NotImplementedError(
                "gen_parser is not implemented for value type: {}".format(val_type)
            )

    return parser


def to_args(obj: Any) -> list[str]:
    """Convert the class fields into args.

    Boolean field will be converted to explicit argument like "--x True --y False".
    """

    value_types = [int, str, float, bool]

    def check_value_type_list(val: Any) -> bool:
        if not isinstance(val, list):
            return False

        for v in val:
            if type(v) not in value_types:
                return False
        return True

    args: list[str] = []
    for key, val in vars(obj).items():
        if type(val) in value_types:
            args += ["--{}".format(key), str(val)]
        elif check_value_type_list(val):
            args += ["--{}".format(key)]
            args += [str(v) for v in val]

        else:
            raise NotImplementedError(
                "to_args is not implemented for this value type: '{}' =  {}".format(
                    key, val
                )
            )

    return args


def parse_and_filter(
    parser: argparse.ArgumentParser, argv: list[str]
) -> dict[str, Any]:
    """Parse given arguments and convert it to dictionary.

    Arguments that have None values will be filtered.
    """
    return {
        k: v for k, v in vars(parser.parse_known_args(argv)[0]).items() if v is not None
    }
