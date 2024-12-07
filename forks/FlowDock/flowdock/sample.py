import os

import hydra
import lovely_tensors as lt
import pandas as pd
import rootutils
import torch
from beartype.typing import Any, Dict, List, Tuple
from lightning import LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

lt.monkey_patch()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from flowdock import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from flowdock import register_custom_omegaconf_resolvers, resolve_omegaconf_variable
from flowdock.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from flowdock.utils.data_utils import (
    create_full_pdb_with_zero_coordinates,
    create_temp_ligand_frag_files,
)

log = RankedLogger(__name__, rank_zero_only=True)

AVAILABLE_SAMPLING_TASKS = ["batched_structure_sampling"]


class SamplingDataset(torch.utils.data.Dataset):
    """Dataset for sampling."""

    def __init__(self, cfg: DictConfig):
        """Initializes the SamplingDataset."""
        if cfg.sampling_task == "batched_structure_sampling":
            if cfg.csv_path is not None:
                # handle variable CSV inputs
                df_rows = []
                self.df = pd.read_csv(cfg.csv_path)
                for _, row in self.df.iterrows():
                    sample_id = row.id
                    input_receptor = row.input_receptor
                    input_ligand = row.input_ligand
                    input_template = row.input_template
                    assert input_receptor is not None, "Receptor path is required for sampling."
                    if input_ligand is not None:
                        if input_ligand.endswith(".sdf"):
                            ligand_paths = create_temp_ligand_frag_files(input_ligand)
                        else:
                            ligand_paths = list(input_ligand.split("|"))
                    else:
                        ligand_paths = None  # handle `null` ligand input
                    if not input_receptor.endswith(".pdb"):
                        log.warning(
                            "Assuming the provided receptor input is a protein sequence. Creating a dummy PDB file."
                        )
                        create_full_pdb_with_zero_coordinates(
                            input_receptor, os.path.join(cfg.out_path, f"input_{sample_id}.pdb")
                        )
                        input_receptor = os.path.join(cfg.out_path, f"input_{sample_id}.pdb")
                    df_row = {
                        "sample_id": sample_id,
                        "rec_path": input_receptor,
                        "lig_paths": ligand_paths,
                    }
                    if input_template is not None:
                        df_row["input_template"] = input_template
                    df_rows.append(df_row)
                self.df = pd.DataFrame(df_rows)
            else:
                sample_id = cfg.sample_id
                input_receptor = cfg.input_receptor
                input_ligand = cfg.input_ligand
                if input_ligand is not None:
                    if input_ligand.endswith(".sdf"):
                        ligand_paths = create_temp_ligand_frag_files(input_ligand)
                    else:
                        ligand_paths = list(input_ligand.split("|"))
                else:
                    ligand_paths = None  # handle `null` ligand input
                if not input_receptor.endswith(".pdb"):
                    log.warning(
                        "Assuming the provided receptor input is a protein sequence. Creating a dummy PDB file."
                    )
                    create_full_pdb_with_zero_coordinates(
                        input_receptor, os.path.join(cfg.out_path, "input.pdb")
                    )
                    input_receptor = os.path.join(cfg.out_path, "input.pdb")
                self.df = pd.DataFrame(
                    [
                        {
                            "sample_id": sample_id,
                            "rec_path": input_receptor,
                            "lig_paths": ligand_paths,
                        }
                    ]
                )
                if cfg.input_template is not None:
                    self.df["input_template"] = cfg.input_template
        else:
            raise NotImplementedError(f"Sampling task {cfg.sampling_task} is not implemented.")

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Returns the input receptor and input ligand."""
        return self.df.iloc[idx].to_dict()


@task_wrapper
def sample(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Samples using given checkpoint on a datamodule predictset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path, "Please provide a checkpoint path with which to sample!"
    assert os.path.exists(cfg.ckpt_path), f"Checkpoint path {cfg.ckpt_path} does not exist!"
    assert (
        cfg.sampling_task in AVAILABLE_SAMPLING_TASKS
    ), f"Sampling task {cfg.sampling_task} is not one of the following available tasks: {AVAILABLE_SAMPLING_TASKS}."
    assert (cfg.input_receptor is not None and cfg.input_ligand is not None) or (
        cfg.csv_path is not None and os.path.exists(cfg.csv_path)
    ), "Please provide either an input receptor and ligand or a CSV file with receptor and ligand sequences/filepaths."

    log.info(
        f"Setting `float32_matmul_precision` to {cfg.model.cfg.task.float32_matmul_precision}."
    )
    torch.set_float32_matmul_precision(precision=cfg.model.cfg.task.float32_matmul_precision)

    # Establish model input arguments
    with open_dict(cfg):
        # NOTE: Structure trajectories will not be visualized when performing auxiliary estimation only
        cfg.model.cfg.prior_type = cfg.prior_type
        cfg.model.cfg.task.detect_covalent = cfg.detect_covalent
        cfg.model.cfg.task.use_template = cfg.use_template
        cfg.model.cfg.task.csv_path = cfg.csv_path
        cfg.model.cfg.task.input_receptor = cfg.input_receptor
        cfg.model.cfg.task.input_ligand = cfg.input_ligand
        cfg.model.cfg.task.input_template = cfg.input_template
        cfg.model.cfg.task.visualize_generated_samples = (
            cfg.visualize_sample_trajectories and not cfg.auxiliary_estimation_only
        )
        cfg.model.cfg.task.auxiliary_estimation_only = cfg.auxiliary_estimation_only
    if cfg.latent_model is not None:
        with open_dict(cfg):
            cfg.model.cfg.latent_model = cfg.latent_model
    with open_dict(cfg):
        if cfg.start_time == "auto":
            cfg.start_time = 1.0
        else:
            cfg.start_time = float(cfg.start_time)

    log.info("Converting sampling inputs into a <SamplingDataset>")
    dataloaders: List[DataLoader] = [
        DataLoader(
            SamplingDataset(cfg),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    ]

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.hparams.cfg.update(cfg)  # update model config with the sampling config

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    plugins = None
    if "_target_" in cfg.environment:
        log.info(f"Instantiating environment <{cfg.environment._target_}>")
        plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.environment)

    strategy = getattr(cfg.trainer, "strategy", None)
    if "_target_" in cfg.strategy:
        log.info(f"Instantiating strategy <{cfg.strategy._target_}>")
        strategy: Strategy = hydra.utils.instantiate(cfg.strategy)
        if (
            "mixed_precision" in strategy.__dict__
            and getattr(strategy, "mixed_precision", None) is not None
        ):
            strategy.mixed_precision.param_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.param_dtype)
                if getattr(cfg.strategy.mixed_precision, "param_dtype", None) is not None
                else None
            )
            strategy.mixed_precision.reduce_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.reduce_dtype)
                if getattr(cfg.strategy.mixed_precision, "reduce_dtype", None) is not None
                else None
            )
            strategy.mixed_precision.buffer_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.buffer_dtype)
                if getattr(cfg.strategy.mixed_precision, "buffer_dtype", None) is not None
                else None
            )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = (
        hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
            strategy=strategy,
        )
        if strategy is not None
        else hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
        )
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for sampling.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    sample(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
