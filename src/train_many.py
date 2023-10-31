"""Train models with given configuration."""
import argparse
import datetime
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import current_thread
from typing import Generator, TypeAlias

from . import myutils, naming
from .mylogger import main_logger

logger = main_logger.getChild(__name__)

# Have to set this environment variable.
# Probably related to the following issue:
# https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"] = "GNU"


@dataclass
class DIORAHparams:
    """The model-related hyperparameters of DIORA.

    These are the default values in the original source code.
    """

    max_epoch: int = 5
    batch_size: int = 10
    hidden_dim: int = 10
    lr: float = 4e-3
    k_neg: int = 3
    freq_dist_power: float = 0.75
    margin: float = 1.0


@dataclass
class PRPNHparams:
    """The model-related hyperparameters of PRPN.

    These are the default values in the original source code.
    """

    epochs: int = 100
    batch_size: int = 64
    emsize: int = 200
    nhid: int = 400
    nlayers: int = 2
    nslots: int = 15
    nlookback: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-6
    clip: float = 1.0
    dropout: float = 0.2
    idropout: float = 0.2
    rdropout: float = 0.0
    tied: bool = True
    hard: bool = True
    res: int = 0
    resolution: float = 0.1


@dataclass
class URNNGHparams:
    """The model-related hyperparameters of URNNG.

    These are the default values in the original source code.
    """

    batch_size: int = 16
    num_epochs: int = 18
    min_epochs: int = 8
    train_q_epochs: int = 2
    w_dim: int = 650
    h_dim: int = 650
    q_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.5
    samples: int = 8
    lr: float = 1.0
    q_lr: float = 0.0001
    action_lr: float = 0.1
    decay: float = 0.5
    kl_warmup: int = 2
    max_grad_norm: float = 5.0
    q_max_grad_norm: float = 1.0


HPARAMS: TypeAlias = DIORAHparams | PRPNHparams | URNNGHparams


def get_thread_id() -> int:
    thread = current_thread()

    # Ad-hoc way to obrain thread index.
    # Thread by default has a name of "ThreadPoolExecuter-%d-%d", where the first number is the thread number in the process, and the scond is the worker thread number within the pool.

    thread_id: int = int(thread.name.split("-")[-1])

    return thread_id


def thread_log(message: str):
    """Print message to stderr."""
    print(
        "{}: Thread {}: {}".format(datetime.datetime.now(), get_thread_id(), message),
        file=sys.stderr,
    )


def thread_run_subprocess(command_str: str, log_file: Path):
    try:
        with log_file.open(mode="w") as f:
            p = subprocess.run(
                command_str,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            p.check_returncode()
    except Exception as e:
        print(
            "Exception raised in thread {}: {}".format(get_thread_id(), e),
            file=sys.stderr,
        )


# Only try the defaults first.
def gen_diora_configs(debug: bool) -> Generator[DIORAHparams, None, None]:
    if not debug:
        # max_epoch is not specified in the paper.
        # In the paper (batch_size, hidden_dim) = (128, 400), (64, 800)
        # and lr = {2, 4, 8, 10}*10^-4
        # Due to memory constraint, we must use hdim=400 and bsz=32
        # We reduce the maximum epochs due to resource problem.
        yield DIORAHparams(
            max_epoch=75, batch_size=32, hidden_dim=400, lr=10e-4, k_neg=100
        )
    else:
        yield DIORAHparams(max_epoch=2, batch_size=2, hidden_dim=10)


def gen_prpn_configs(debug: bool) -> Generator[PRPNHparams, None, None]:
    if not debug:
        # Since the hyperparameters for uncupervised parsing are not reported in the paper, we use the default parameters in their implemenatation.
        # We reduce the maximum epochs due to resource problem.
        yield PRPNHparams(epochs=75)
        # The following parameters are for word-level PTB language modeling.
        # yield PRPNHparams(
        #    epochs=75,
        #    batch_size=64,
        #    lr=0.003,
        #    emsize=800,
        #    nhid=1200,
        #    nlayers=2,
        #    nlookback=5,
        #    dropout=0.5,
        #    idropout=0.7,
        #    rdropout=0.5,
        #    nslots=15,
        #    res=0,
        # )
    else:
        yield PRPNHparams(
            epochs=2,
            batch_size=2,
            emsize=10,
            nhid=10,
            nlayers=1,
            res=0,
        )


def gen_urnng_configs(debug: bool) -> Generator[URNNGHparams, None, None]:
    if not debug:
        # The hyperparameters reported in the paper.
        yield URNNGHparams(
            batch_size=16,
            num_epochs=18,
            min_epochs=8,
            train_q_epochs=2,
            w_dim=650,
            h_dim=650,
            q_dim=256,
            num_layers=1,
            dropout=0.5,
            samples=8,
            lr=1.0,
            q_lr=0.0001,
            action_lr=0.1,
            decay=0.5,
            kl_warmup=2,
            max_grad_norm=5.0,
            q_max_grad_norm=1.0,
        )
    else:
        yield URNNGHparams(
            batch_size=2,
            num_epochs=2,
            min_epochs=1,
            w_dim=10,
            h_dim=10,
            q_dim=10,
            num_layers=1,
            dropout=0.5,
        )


def train_many(
    dataset_dir: Path,
    dataset_key_l: list[str],
    train_dir: Path,
    log_dir: Path,
    archi_l: list[str],
    train_random_seeds: list[int],
    split_to_parse: str,
    n_parallel: int,
    use_gpu: bool,
    gpu_ids: list[int],
    only_eval: bool,
    force_update: bool,
    debug: bool,
):
    """Train models in parallel.

    Args:
        dataset_dir:
            Directory where all the datasets are placed.
        train_dataset_key_l:
            List of strings specifying the dataset used for training.
        test_dataset_key:
            String specifying the dataset used for testing.
        train_dir:
            Working directory for model training.
        log_dir:
            Logging directory where the log files will be saved.
        archi:
            String specifying the architecture to train.
        train_random_seeds:
            List of random seeds used for training.
        split_to_parse:
            The dataset split to parse. This value should be one of "train", "dev", and "test".
        n_parallel:
            Number of parallel processes to run for tuning.
        use_gpu:
            If True, then it is assumed that there are n_parallel gpus available. Each of the parallel process is assigned one gpu. If the number of available gpus n_gpus is smaller than n_parallel, then (n_gpus) process will use gpus and (n_parallel - n_gpus) proces will be cpu-only.
        only_eval:
            Skip training and use trained model for evaluation.
        force_update:
            If True, then the training will be done ignoring previously saved results.
    """

    # Check gpu counts.
    if use_gpu and (n_parallel > len(gpu_ids)):
        logger.warning(
            f"!!!!n_parallel={n_parallel} but only {len(gpu_ids)} gpus given. Using {n_parallel - len(gpu_ids)} cpus instead.!!!!"
        )

    # Set gen_config:
    def gen_configs() -> Generator[tuple[str, HPARAMS], None, None]:
        for archi in archi_l:
            match archi:
                case "diora":
                    tmp_gen_configs = gen_diora_configs
                case "prpn":
                    tmp_gen_configs = gen_prpn_configs
                case "urnng":
                    tmp_gen_configs = gen_urnng_configs
                case _:
                    raise Exception(f"No such architecture: {archi}")
            for config in tmp_gen_configs(debug=debug):
                yield archi, config

    # Define the train execution for the models.
    def execute_train(args: tuple[tuple[str, HPARAMS], str, int]):
        (archi, hparams), dataset_key, seed = args
        thread_log(
            f"For archi: {archi}, seed: {seed}, hparams: {hparams}, dataset_key: {dataset_key}, eval split: {split_to_parse}"
        )

        hparams_key: str = myutils.get_hashed_key(str(hparams))

        out_file: Path = naming.hparam_out_file(
            train_dir=train_dir,
            dataset_key=dataset_key,
            archi=archi,
            hparams_key=hparams_key,
            seed=seed,
            split_to_parse=split_to_parse,
        )

        dataset_key_dir: Path = naming.get_dataset_key_dir(
            dataset_dir=dataset_dir, dataset_key=dataset_key
        )
        train_file: Path = naming.train_tree(dataset_dir=dataset_key_dir)
        dev_file: Path = naming.dev_tree(dataset_dir=dataset_key_dir)
        test_file: Path = naming.test_tree(dataset_dir=dataset_key_dir)

        match split_to_parse:
            case "train":
                file_to_parse: Path = train_file
            case "dev":
                file_to_parse: Path = dev_file
            case "test":
                file_to_parse: Path = test_file
            case _:
                raise Exception("Invalid split to parse")

        # Check if the train results already exists.
        # If the results already exists training will be skipped.
        if force_update or (not out_file.exists()):
            # Make the hparams working directory just in case.
            work_dir: Path = out_file.parent
            work_dir.mkdir(parents=True, exist_ok=True)

            # Get gpu id.
            gpu_id = None
            thread_id: int = get_thread_id()
            if use_gpu and (thread_id < len(gpu_ids)):
                gpu_id = gpu_ids[thread_id]

            # Set gpu option.
            gpu_option: list[str] = (
                ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)]
                if gpu_id is not None
                else ['CUDA_VISIBLE_DEVICES=""']
            )

            # Generate the command for training.
            # Match the hparams class.
            if isinstance(hparams, DIORAHparams):
                gpu_flag: list[str] = ["--cuda"] if gpu_id is not None else []

                # Give the abolute paths.
                train_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python pytorch/diora/scripts/train.py",
                        f"--experiment_path {work_dir.resolve()}",
                        f"--train_path {train_file.resolve()}",
                        f"--validation_path {dev_file.resolve()}",
                        f"--test_path {test_file.resolve()}",
                        f"--data_type ptb",
                        f"--emb onehot",
                        f"--seed {seed}",
                        f"--hidden_dim {hparams.hidden_dim}",
                        f"--batch_size {hparams.batch_size}",
                        f"--max_epoch {hparams.max_epoch}",
                        f"--lr {hparams.lr}",
                        f"--k_neg {hparams.k_neg}",
                        f"--freq_dist_power {hparams.freq_dist_power}",
                        f"--margin {hparams.margin}",
                    ]
                    + gpu_flag
                )

                model_file: Path = work_dir.joinpath("model.pt")

                test_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python pytorch/diora/scripts/parse.py",
                        f"--experiment_path {work_dir.resolve()}",
                        f"--train_path {train_file.resolve()}",
                        f"--validation_path {dev_file.resolve()}",
                        f"--test_path {file_to_parse.resolve()}",
                        f"--output_file {out_file.resolve()}",
                        f"--data_type ptb",
                        f"--emb onehot",
                        f"--load_model_path {model_file.resolve()}",
                        # f"--seed {seed}",
                        f"--hidden_dim {hparams.hidden_dim}",
                        f"--batch_size {hparams.batch_size}",
                        f"--max_epoch {hparams.max_epoch}",
                        # f"--lr {hparams.lr}",
                        f"--k_neg {hparams.k_neg}",
                        f"--freq_dist_power {hparams.freq_dist_power}",
                        f"--margin {hparams.margin}",
                    ]
                    + gpu_flag
                )

                if not only_eval:
                    thread_log("Train from scratch!!!!")
                    command_str: str = (
                        f"cd src/diora; {train_command_str}; {test_command_str}"
                    )
                else:
                    thread_log("Skip training and use an already-trained model!!!!")
                    # Use already-trained model and skip training.
                    command_str: str = f"cd src/diora; {test_command_str}"

            elif isinstance(hparams, PRPNHparams):
                model_file: Path = work_dir.joinpath("model.pickle")

                gpu_flag: list[str] = ["--cuda"] if gpu_id is not None else []

                # Give the abolute paths.
                train_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python main_UP.py",
                        f"--train_file {train_file.resolve()}",
                        f"--dev_file {dev_file.resolve()}",
                        f"--test_file {test_file.resolve()}",
                        f"--save {model_file.resolve()}",
                        f"--seed {seed}",
                        f"--tied" if hparams.tied else "",
                        f"--hard" if hparams.hard else "",
                        f"--epochs {hparams.epochs}",
                        f"--batch_size {hparams.batch_size}",
                        f"--emsize {hparams.emsize}",
                        f"--nhid {hparams.nhid}",
                        f"--nlayers {hparams.nlayers}",
                        f"--lr {hparams.lr}",
                        f"--weight_decay {hparams.weight_decay}",
                        f"--clip {hparams.clip}",
                        f"--dropout {hparams.dropout}",
                        f"--idropout {hparams.idropout}",
                        f"--rdropout {hparams.rdropout}",
                        f"--nslots {hparams.nslots}",
                        f"--nlookback {hparams.nlookback}",
                        f"--resolution {hparams.resolution}",
                    ]
                    + gpu_flag
                )

                test_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python my_parse.py",
                        f"--train_file {train_file.resolve()}",
                        f"--dev_file {dev_file.resolve()}",
                        f"--test_file {test_file.resolve()}",
                        f"--input_file {file_to_parse.resolve()}",
                        f"--output_file {out_file.resolve()}",
                        f"--checkpoint {model_file.resolve()}",
                    ]
                    + gpu_flag
                )

                if not only_eval:
                    thread_log("Train from scratch!!!!")
                    command_str: str = (
                        f"cd src/prpn; {train_command_str}; {test_command_str}"
                    )
                else:
                    thread_log("Skip training and use an already-trained model!!!!")
                    # Use already-trained model and skip training.
                    command_str: str = f"cd src/prpn; {test_command_str}"

            elif isinstance(hparams, URNNGHparams):
                # Use the preprocessed file for urnng.
                urnng_output_file: Path = naming.urnng_output_file(
                    dataset_key_dir=dataset_key_dir, batch_size=hparams.batch_size
                )

                model_file: Path = work_dir.joinpath("model.pickle")

                gpu_flag: list[str] = ["--gpu 0"] if gpu_id is not None else []

                # Give the abolute paths.
                train_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python train.py",
                        f"--output_file {urnng_output_file.resolve()}",
                        f"--save_path {model_file.resolve()}",
                        f"--seed {seed}",
                        f"--num_epochs {hparams.num_epochs}",
                        f"--min_epochs {hparams.min_epochs}",
                        f"--train_q_epochs {hparams.train_q_epochs}",
                        f"--w_dim {hparams.w_dim}",
                        f"--h_dim {hparams.h_dim}",
                        f"--q_dim {hparams.q_dim}",
                        f"--num_layers {hparams.num_layers}",
                        f"--dropout {hparams.dropout}",
                        f"--samples {hparams.samples}",
                        f"--lr {hparams.lr}",
                        f"--q_lr {hparams.q_lr}",
                        f"--action_lr {hparams.action_lr}",
                        f"--decay {hparams.decay}",
                        f"--kl_warmup {hparams.kl_warmup}",
                        f"--max_grad_norm {hparams.max_grad_norm}",
                        f"--q_max_grad_norm {hparams.q_max_grad_norm}",
                    ]
                    + gpu_flag
                )

                test_command_str: str = " ".join(
                    gpu_option
                    + [
                        f"python parse.py",
                        f"--data_file {file_to_parse.resolve()}",
                        f"--model_file {model_file.resolve()}",
                        f"--out_file {out_file.resolve()}",
                    ]
                    + gpu_flag
                )

                if not only_eval:
                    thread_log("Train from scratch!!!!")
                    command_str: str = (
                        f"cd src/urnng; {train_command_str}; {test_command_str}"
                    )
                else:
                    thread_log("Skip training and use an already-trained model!!!!")
                    # Use already-trained model and skip training.
                    command_str: str = f"cd src/urnng; {test_command_str}"

            else:
                raise Exception(f"No such Hparams: {hparams}")

            thread_log(
                f"GPU id: {gpu_id}: Start training: {archi}, {seed}, {str(hparams)}, {dataset_key}, {split_to_parse}"
            )

            log_file: Path = naming.get_log_file(
                log_dir=log_dir,
                dataset_key=dataset_key,
                archi=archi,
                hparams_key=hparams_key,
                seed=seed,
                eval_split=split_to_parse,
            )
            log_file.parent.mkdir(parents=True, exist_ok=True)

            thread_run_subprocess(
                command_str=command_str,
                log_file=log_file,
            )

            thread_log(
                f"GPU id: {gpu_id}: End training: {archi}, {seed}, {str(hparams)}, {dataset_key}, {split_to_parse}"
            )
        else:
            thread_log(
                f"Skip training: {archi}, {seed}, {str(hparams)}, {dataset_key}, {split_to_parse}"
            )

    logger.info("Start training!!!")

    with ThreadPoolExecutor(max_workers=n_parallel) as executer:
        # Pass the seed and hparams.
        executer.map(
            execute_train,
            itertools.product(gen_configs(), dataset_key_l, train_random_seeds),
        )

    logger.info("Finish training!!!")


@dataclass
class Args:
    dataset_key_l: list[str]

    archi_l: list[str]
    train_random_seeds: list[int]

    n_parallel: int
    use_gpu: bool


@dataclass
class OptionalArgs:
    dataset_dir: str = "./data/dataset"
    train_dir: str = "./results/train/"
    log_dir: str = "./log/train"

    split_to_parse: str = "test"

    gpu_ids: list[int] = field(default_factory=list)

    only_eval: bool = False
    force_update: bool = False
    debug: bool = True


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

    log_dir: Path = Path(opt_args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_dir: Path = Path(opt_args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    train_many(
        dataset_dir=Path(opt_args.dataset_dir),
        dataset_key_l=args.dataset_key_l,
        train_dir=train_dir,
        log_dir=log_dir,
        archi_l=args.archi_l,
        train_random_seeds=args.train_random_seeds,
        split_to_parse=opt_args.split_to_parse,
        n_parallel=args.n_parallel,
        use_gpu=args.use_gpu,
        gpu_ids=opt_args.gpu_ids,
        only_eval=opt_args.only_eval,
        force_update=opt_args.force_update,
        debug=opt_args.debug,
    )
