import argparse
import itertools
import json
import multiprocessing
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from . import myutils, naming
from .branching_measure import (
    DeepListTree,
    SpanTree,
    get_num_leaves_deeplisttree,
    nary_relative_corrected_colles_index,
    nary_relative_equal_weights_corrected_colles_index,
    nary_relative_rogers_j_index,
    rename_deeplisttree,
    spantree2deeplisttree,
)
from .mylogger import main_logger

# from tqdm import tqdm


logger = main_logger.getChild(__name__)

# Set colors and markers to use.
colors: Iterable[str] = itertools.cycle(
    [
        "magenta",
        "tab:brown",
        "tab:green",
        "gold",
        "tab:pink",
        "tab:purple",
        "tab:blue",
        "tab:orange",
        "tab:gray",
        "tab:red",
        "tab:gray",
    ]
)

markers: Iterable[str] = itertools.cycle(["o", "v", "^", "s", "d"])


def get_key2color(keys: list[str], given_colors: list[str] = []) -> dict[str, str]:
    # Use the specified colors if given.
    if len(given_colors) > 0:
        colors: Iterable[str] = itertools.cycle(given_colors)
    else:
        colors: Iterable[str] = itertools.cycle(
            [
                "magenta",
                "tab:brown",
                "tab:green",
                "gold",
                "tab:pink",
                "tab:purple",
                "tab:blue",
                "tab:orange",
                "tab:gray",
                "tab:red",
                "tab:gray",
            ]
        )

    key2color: dict[str, str] = {}
    for key in keys:
        key2color[key] = next(colors)
    return key2color


def get_key2marker(keys: list[str]) -> dict[str, str]:
    key2marker: dict[str, str] = {}
    for key in keys:
        key2marker[key] = next(markers)
    return key2marker


# archi_name -> dataset_name -> train_random_seed -> list[DeepListTree]
MODEL2TREES: TypeAlias = dict[str, dict[str, list[list[DeepListTree]]]]


def get_model_trees(
    archi: str, result_filepath: Path, maxlen: int
) -> list[DeepListTree]:
    trees: list[DeepListTree] = []

    # The results are in jsonl format.
    logger.info(f"Loading trees for {archi} from {str(result_filepath)}")
    with result_filepath.open(mode="r") as f:
        for line in f:
            match archi:
                case "diora":
                    diora_loaded: dict[str, int | DeepListTree] = json.loads(line)
                    dlt: DeepListTree = rename_deeplisttree(diora_loaded["tree"])

                case "prpn":
                    prpn_loaded: DeepListTree = json.loads(line)
                    dlt: DeepListTree = rename_deeplisttree(prpn_loaded)

                case "urnng":
                    urnng_loaded: list[list[int]] = json.loads(line)

                    # Since tuples are expressed as lists in json format, convert the lists into tuples.
                    st: SpanTree = list(map(tuple, urnng_loaded))

                    dlt: DeepListTree = spantree2deeplisttree(st)
                case _:
                    raise Exception(f"No such archi: {archi}")

            # Exclude trivial trees.
            num_leaves: int = get_num_leaves_deeplisttree(dlt)
            if num_leaves > 2:
                # Filter data by length.
                # If maxlen < 0, then data are not filtered.
                if (maxlen > 0) and (num_leaves > maxlen):
                    continue

                trees.append(dlt)

    logger.info(f"Loaded {len(trees)} trees.")
    return trees


def get_model2trees(
    train_dir: Path,
    archi_l: list[str],
    dataset_key_l: list[str],
    hparams_key_l: list[str],
    dataset_train_seed_l: list[int],
    full_train_random_seed_l: list[int],
    archi_name_l: list[str],
    dataset_name_l: list[str],
    maxlen: int,
    split_to_parse: str,
) -> MODEL2TREES:
    """Retrieve the results.

    Args:
        train_dir:
            Working directory for model training.
        dataset_key_l:
            List of strings specifying the dataset.
        archi_l:
            List of strings specifying the architecture to train.
        hparams_key_l:
            List of strings specifying the hyperparameters. The hyperparameters must align with archi_l.
        dataset_train_seed_l:
            The number of train seeds for each dataset_key. The seeds are assumed to be full_train_random_seed_l[:dataset_train_seed_l[i]] for i-th dataset.
        full_train_random_seed_l:
            Full list of random seeds used for training.
        archi_name_l:
            List of strings representing the architecture (and hyperparameters). This is used as the legends. The names must align with archi_l and hparams_key_l.
        dataset_name_l:
            List of strings representing the train datasets. This is used as the xtick labels.
        maxlen:
            Evaluation is only done for sequences equal or shorter than maxlen. If maxlen < 0, then sequences are not filtered (except trivial ones).
    """
    assert len(archi_l) == len(archi_name_l)
    assert len(archi_l) == len(hparams_key_l)
    assert len(dataset_key_l) == len(dataset_name_l)
    assert len(dataset_key_l) == len(dataset_train_seed_l)

    model2trees: MODEL2TREES = {}

    for archi, hparams_key, archi_name in zip(archi_l, hparams_key_l, archi_name_l):
        if archi_name not in model2trees:
            model2trees[archi_name] = {}

        for i, (dataset_key, dataset_name) in enumerate(
            zip(dataset_key_l, dataset_name_l)
        ):
            if dataset_name not in model2trees:
                model2trees[archi_name][dataset_name] = []

            # Retrieve the results for each seed.
            # Assume that for i-th dataset, train seeds are full_train_random_seed_l[: dataset_train_seed_l[i]]
            for seed in full_train_random_seed_l[: dataset_train_seed_l[i]]:
                result_file: Path = naming.hparam_out_file(
                    train_dir=train_dir,
                    dataset_key=dataset_key,
                    archi=archi,
                    hparams_key=hparams_key,
                    seed=seed,
                    split_to_parse=split_to_parse,
                )
                # TODO:
                seed_res: list[DeepListTree] = get_model_trees(
                    archi=archi, result_filepath=result_file, maxlen=maxlen
                )

                model2trees[archi_name][dataset_name].append(seed_res)

    return model2trees


def nrow_plot_line(
    model2trees: MODEL2TREES,
    archi_name_l: list[str],
    dataset_name_l: list[str],
    row_construct: list[tuple[int, int]],
    save_filepath: Path,
    figtitle: str,
    figsize: tuple[float, float],
    alpha: float,
    capsize: float,
    rotate_ytick: bool,
    raw_label: bool,
    fontsize: float,
    legend_fontsize: float,
    legend_out: bool,
    markersize: float,
    raw_tick: bool,
    nospace: bool,
    num_proc: int,
    manual_adjust: bool,
    adjust_left: float,
    adjust_right: float,
    adjust_top: float,
    adjust_bottom: float,
    adjust_wspace: float,
    adjust_hspace: float,
):
    """
    row_construct:
        i-th tuple (n, m) indicates that i-th row consists of n-datasets with m-variations. the m-variations are jointed by line for each dataset. (E.g., the variation is the morphisms.)
    """

    logger.info(f"Start plotting for: {archi_name_l}, {dataset_name_l}")

    # Set fontsize and adjust subplots.
    plt.rcParams["font.size"] = fontsize
    # plt.rcParams["figure.subplot.left"] = adjust_left
    # plt.rcParams["figure.subplot.right"] = adjust_right
    # plt.rcParams["figure.subplot.top"] = adjust_top
    # plt.rcParams["figure.subplot.bottom"] = adjust_bottom
    # plt.rcParams["figure.subplot.wspace"] = adjust_wspace
    # plt.rcParams["figure.subplot.hspace"] = adjust_hspace

    if nospace:
        plt.rcParams["savefig.pad_inches"] = 0

    fig, axes = plt.subplots(
        len(row_construct), 3, figsize=figsize, sharex=True, sharey="row"
    )
    if len(row_construct) == 1:
        axes = [axes]

    if manual_adjust:
        fig.subplots_adjust(
            left=adjust_left,
            right=adjust_right,
            top=adjust_top,
            bottom=adjust_bottom,
            wspace=adjust_wspace,
            hspace=adjust_hspace,
        )

    if raw_tick:
        dataset_name_l = [r"{}".format(dn) for dn in dataset_name_l]

    # Set labels
    axes[-1][0].set_xlabel(xlabel=r"$\mathrm{CC}^{\pm}$")
    axes[-1][1].set_xlabel(xlabel=r"$\mathrm{EWC}^{\pm}$")
    axes[-1][2].set_xlabel(xlabel=r"$\mathrm{RJ}^{\pm}$")

    for ax in axes[-1]:
        ax.set_xlim(left=-1.05, right=1.05)

    num_drawn_datasets: int = 0

    archi2color: dict[str, str] = get_key2color(archi_name_l)
    archi2marker: dict[str, str] = get_key2marker(archi_name_l)

    branching_measures = [
        nary_relative_corrected_colles_index,
        nary_relative_equal_weights_corrected_colles_index,
        nary_relative_rogers_j_index,
    ]

    # Plot.
    logger.info(f"Use {num_proc} process!!!")
    pool = multiprocessing.Pool(processes=num_proc)

    for i, (n_datsets, m_random_seeds) in enumerate(row_construct):
        for ax in axes[i]:
            ax.axvline(x=0.0, color="lightgrey", linestyle="dashed")

        # Retrive the datasets for this row.
        dataset_names_row: list[str] = dataset_name_l[
            num_drawn_datasets : num_drawn_datasets + (n_datsets * m_random_seeds)
        ]

        # Set yticks.
        ytick_values = list(range(len(dataset_names_row)))
        axes[i][0].set_yticks(ytick_values)

        # Reverse dataset_name_l so that the datasets are listed from the top.
        dataset_names_row_inv = dataset_names_row[::-1]

        if rotate_ytick:
            axes[i][0].set_yticklabels(dataset_names_row_inv, rotation=45, ha="right")
        else:
            axes[i][0].set_yticklabels(dataset_names_row_inv)

        for archi_name in archi_name_l:
            for ax, b in zip(axes[i], branching_measures):
                for j in range(n_datsets):
                    dataset_variations: list[str] = dataset_names_row_inv[
                        m_random_seeds * j : m_random_seeds * (j + 1)
                    ]
                    tmp_ytick_values = ytick_values[
                        m_random_seeds * j : m_random_seeds * (j + 1)
                    ]

                    # Calculate branching values:
                    means: list[float] = []
                    stderrs: list[float] = []

                    for dataset_name in dataset_variations:
                        # The results for different random seeds.
                        res_seeds: list[float] = []

                        # First, calculate the mean for each model.
                        for model_result in model2trees[archi_name][dataset_name]:
                            # branching_values: list[float] = [
                            #    b(dlt) for dlt in model_result
                            # ]

                            # Parallel processing.
                            branching_values: list[float] = pool.map(b, model_result)

                            # Calculate the mean branching value for the output trees.
                            seed_mean: float = np.mean(branching_values)

                            res_seeds.append(seed_mean)

                        mean: float = np.mean(res_seeds)
                        stderr: float = np.std(res_seeds) / np.sqrt(len(res_seeds))

                        means.append(mean)
                        stderrs.append(stderr)

                    ax.errorbar(
                        x=means,
                        y=tmp_ytick_values,
                        xerr=stderrs,
                        marker=archi2marker[archi_name],
                        markersize=markersize,
                        color=archi2color[archi_name],
                        capsize=capsize,
                        alpha=alpha,
                        label=archi_name if i == 0 and j == 0 else None,
                    )
        num_drawn_datasets += n_datsets * m_random_seeds

    if figtitle != "":
        if raw_label:
            fig.suptitle(r"{}".format(figtitle))
        else:
            fig.suptitle(figtitle)

    # Set the legend.
    if legend_out:
        axes[0][1].legend(
            bbox_to_anchor=(0.5, 1.01),
            loc="lower center",
            ncol=len(archi_name_l),
            shadow=True,
            fontsize=legend_fontsize,
        )
    else:
        axes[0][1].legend(shadow=True, fontsize=legend_fontsize)

    # Save the plot.
    if not manual_adjust:
        fig.tight_layout()
    fig.savefig(str(save_filepath))

    logger.info(f"Finish plotting for: {archi_name_l}, {dataset_name_l}")


@dataclass
class Args:
    archi_l: list[str]
    dataset_key_l: list[str]
    hparams_key_l: list[str]
    dataset_train_seed_l: list[int]
    full_train_random_seed_l: list[int]

    archi_name_l: list[str]
    dataset_name_l: list[str]


@dataclass
class OptionalArgs:
    save_filepath: str = "./results/plot/plot.png"

    train_dir: str = "./results/train/"

    split_to_parse: str = "test"

    figtitle: str = ""
    figsize_x: float = 12
    figsize_y: float = 6
    alpha: float = 0.8
    fill: bool = True
    capsize: float = 10.0
    rotate_ytick: bool = False
    raw_label: bool = True
    fontsize: float = 12
    num_bins: int = 100
    nospace: bool = True
    legend_fontsize: float = 10
    legend_out: bool = False

    colors: list[str] = field(default_factory=list)

    row_num_datasets: list[int] = field(default_factory=list)
    row_num_morphisms: list[int] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)

    raw_tick: bool = False
    markersize: float = 5.0
    num_proc: int = 1

    maxlen: int = -1

    manual_adjust: bool = False
    adjust_left: float = 0.125
    adjust_right: float = 0.9
    adjust_top: float = 0.9
    adjust_bottom: float = 0.1
    adjust_wspace: float = 0.1
    adjust_hspace: float = 0.1


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

    save_filepath: Path = Path(opt_args.save_filepath)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Start loading the results!!!!")
    logger.info(
        "No length limit!!"
        if opt_args.maxlen < 0
        else f"Limit test sequences by maxlen: {opt_args.maxlen}"
    )
    model2trees: MODEL2TREES = get_model2trees(
        train_dir=Path(opt_args.train_dir),
        archi_l=args.archi_l,
        dataset_key_l=args.dataset_key_l,
        hparams_key_l=args.hparams_key_l,
        dataset_train_seed_l=args.dataset_train_seed_l,
        full_train_random_seed_l=args.full_train_random_seed_l,
        archi_name_l=args.archi_name_l,
        dataset_name_l=args.dataset_name_l,
        maxlen=opt_args.maxlen,
        split_to_parse=opt_args.split_to_parse,
    )
    logger.info("Finish loading the results!!!!")

    assert len(opt_args.row_num_datasets) == len(opt_args.row_num_morphisms)

    nrow_plot_line(
        model2trees=model2trees,
        archi_name_l=args.archi_name_l,
        dataset_name_l=args.dataset_name_l,
        save_filepath=save_filepath,
        row_construct=list(zip(opt_args.row_num_datasets, opt_args.row_num_morphisms)),
        figtitle=opt_args.figtitle,
        figsize=(opt_args.figsize_x, opt_args.figsize_y),
        alpha=opt_args.alpha,
        capsize=opt_args.capsize,
        rotate_ytick=opt_args.rotate_ytick,
        raw_label=opt_args.raw_label,
        fontsize=opt_args.fontsize,
        nospace=opt_args.nospace,
        legend_fontsize=opt_args.legend_fontsize,
        legend_out=opt_args.legend_out,
        raw_tick=opt_args.raw_tick,
        markersize=opt_args.markersize,
        num_proc=opt_args.num_proc,
        manual_adjust=opt_args.manual_adjust,
        adjust_left=opt_args.adjust_left,
        adjust_right=opt_args.adjust_right,
        adjust_top=opt_args.adjust_top,
        adjust_bottom=opt_args.adjust_bottom,
        adjust_wspace=opt_args.adjust_wspace,
        adjust_hspace=opt_args.adjust_hspace,
    )
