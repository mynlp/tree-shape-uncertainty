import argparse
import itertools
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nltk
import numpy as np

from . import myutils
from .branching_measure import (
    DeepListTree,
    nary_relative_corrected_colles_index,
    nary_relative_equal_weights_corrected_colles_index,
    nary_relative_rogers_j_index,
    nltktree2deeplisttree,
)
from .mylogger import main_logger

logger = main_logger.getChild(__name__)

# Set colors and markers to use.
colors: Iterable[str] = itertools.cycle(
    [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:cyan",
        "tab:gray",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]
)


def get_corpus2trees(
    corpus_files: list[str], corpus_names: list[str]
) -> dict[str, list[DeepListTree]]:
    """Obtain unlabeled trees from PTB-style corpora."""

    assert len(corpus_files) == len(corpus_names)

    corpus2trees: dict[str, list[DeepListTree]] = {}

    for corpus_file, corpus_name in zip(corpus_files, corpus_names):
        if corpus_name not in corpus2trees:
            corpus2trees[corpus_name] = []

        with Path(corpus_file).open(mode="r") as f:
            for l in f:
                # Convert nltk trees.
                t = nltk.Tree.fromstring(l)

                # Only consider sequences longer than 2.
                if len(t.leaves()) > 2:
                    tree: DeepListTree = nltktree2deeplisttree(t)

                    corpus2trees[corpus_name].append(tree)

    return corpus2trees


def nrow_histogram_plot(
    corpus2trees: dict[str, list[DeepListTree]],
    corpus_names: list[str],
    save_filepath: Path,
    nrow: int,
    figtitle: str,
    figsize: tuple[float, float],
    alpha: float,
    raw_label: bool,
    fontsize: float,
    legend_fontsize: float,
    legend_out: bool,
    num_bins: int,
    fill: bool,
    nospace: bool,
    num_proc: int,
):
    logger.info(f"Start plotting for {corpus2trees.keys()}")

    # Set fontsize.
    plt.rcParams["font.size"] = fontsize

    if nospace:
        plt.rcParams["savefig.pad_inches"] = 0

    fig, axes = plt.subplots(nrow, 3, figsize=figsize, sharex=True, sharey=True)

    if nrow == 1:
        axes = [axes]

    axes[-1][0].set_xlabel(xlabel=r"$\mathrm{CC}^{\pm}$")
    axes[-1][1].set_xlabel(xlabel=r"$\mathrm{EWC}^{\pm}$")
    axes[-1][2].set_xlabel(xlabel=r"$\mathrm{RJ}^{\pm}$")

    for i in range(nrow):
        axes[i][0].set_ylabel(ylabel="proportion")

    logger.info(f"Use {num_proc} process!!!")
    pool = multiprocessing.Pool(processes=num_proc)

    for i in range(nrow):
        for ax in axes[i]:
            ax.axvline(x=0.0, color="lightgrey", linestyle="dashed")

        branching_measures = [
            nary_relative_corrected_colles_index,
            nary_relative_equal_weights_corrected_colles_index,
            nary_relative_rogers_j_index,
        ]

        for corpus_name in corpus_names[
            int(len(corpus_names) / nrow * i) : int(len(corpus_names) / nrow * (i + 1))
        ]:
            # Use the same color for each corpus.
            color: str = next(colors)

            for ax, b in zip(axes[i], branching_measures):
                # Calculate the counts for each bin.
                # values: list[float] = [b(t) for t in corpus2trees[corpus_name]]

                # Parallel execution.
                values: list[float] = pool.map(b, corpus2trees[corpus_name])

                counts, bins = np.histogram(values, bins=num_bins, range=(-1, 1))

                # Normalize the counts.
                normalized_counts = counts / sum(counts)

                # Plot histogram.
                # Set the weights so that the sum of the heights of the bars is 1.
                ax.hist(
                    x=bins[:-1],
                    bins=num_bins,
                    histtype="step" if not fill else "stepfilled",
                    color=color,
                    alpha=alpha,
                    weights=normalized_counts,
                    label=corpus_name,
                )

                # Plot the mean
                mean = np.mean(values)
                ax.axvline(x=mean, color=color, linestyle="dotted")

        # Set the legend.
        if legend_out:
            axes[i][1].legend(
                bbox_to_anchor=(0.5, 1.01),
                loc="lower center",
                ncol=len(corpus2trees.keys()),
                shadow=True,
                fontsize=legend_fontsize,
            )
        else:
            axes[i][1].legend(
                bbox_to_anchor=(0.01, 0.99),
                loc="upper left",
                shadow=True,
                fontsize=legend_fontsize,
            )

    if figtitle != "":
        if raw_label:
            fig.suptitle(r"{}".format(figtitle))
        else:
            fig.suptitle(figtitle)

    # Save the plot.
    fig.tight_layout()
    fig.savefig(str(save_filepath))

    logger.info(f"Finish plotting for: {corpus_names}")


@dataclass
class Args:
    corpus_files: list[str]
    corpus_names: list[str]


@dataclass
class OptionalArgs:
    save_filepath: str = "./results/plot/plot.png"

    figtitle: str = ""
    figsize_x: float = 12
    figsize_y: float = 6
    alpha: float = 0.5
    raw_label: bool = True
    fontsize: float = 12
    nospace: bool = True
    legend_fontsize: float = 10
    legend_out: bool = False
    markersize: float = 5.0
    num_bins: int = 100
    nrow: int = 2
    fill: bool = True
    num_proc: int = 1


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

    corpus2trees: dict[str, list[DeepListTree]] = get_corpus2trees(
        corpus_files=args.corpus_files, corpus_names=args.corpus_names
    )

    nrow_histogram_plot(
        corpus2trees=corpus2trees,
        corpus_names=args.corpus_names,
        save_filepath=save_filepath,
        nrow=opt_args.nrow,
        figtitle=opt_args.figtitle,
        figsize=(opt_args.figsize_x, opt_args.figsize_y),
        alpha=opt_args.alpha,
        raw_label=opt_args.raw_label,
        fontsize=opt_args.fontsize,
        legend_fontsize=opt_args.legend_fontsize,
        legend_out=opt_args.legend_out,
        num_bins=opt_args.num_bins,
        fill=opt_args.fill,
        nospace=opt_args.nospace,
        num_proc=opt_args.num_proc,
    )
