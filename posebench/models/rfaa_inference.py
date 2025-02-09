# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import re
import subprocess  # nosec
import traceback

import hydra
import rootutils
from omegaconf import DictConfig, open_dict
from rdkit import Chem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.models.ensemble_generation import (
    create_rfaa_bash_script,
    dynamically_build_rfaa_input_config,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def protein_file_sort_key(filename: str) -> int:
    """Sort key for protein filenames."""
    match = re.search(r"chain_(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def ligand_file_sort_key(filename: str) -> int:
    """Sort key for ligand filenames."""
    match = re.search(r"ligand_(.+?)_(\d+)\.sdf", filename)
    if match:
        return int(match.group(2))
    return 0


def run_rfaa_inference_directly(cfg: DictConfig):
    """Run inference using a trained RoseTTAFold-All-Atom model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    assert (
        cfg.inference_config_name is not None
    ), "`inference_config_name` must be provided in the config file."
    assert (
        cfg.inference_dir_name is not None
    ), "`inference_dir_name` must be provided in the config file."
    os.makedirs(os.path.join(cfg.output_dir, cfg.inference_dir_name), exist_ok=True)
    try:
        subprocess.run(
            [
                cfg.python_exec_path,
                "-m",
                "rf2aa.run_inference",
                "--config-name",
                cfg.inference_config_name,
                f"loader_params.MAXCYCLE={cfg.max_cycles}",
                f"output_path={os.path.join(cfg.output_dir, cfg.inference_dir_name)}",
            ],
            check=True,
            cwd=cfg.rfaa_exec_dir,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg.cuda_device_index)},
        )  # nosec
    except Exception as e:
        raise e
    logger.info(
        f"RoseTTAFold-All-Atom inference for config input file `{cfg.inference_config_name}` complete."
    )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="rfaa_inference.yaml",
)
def main(cfg: DictConfig):
    """Create SLURM job submission scripts for inference with RoseTTAFold-All-
    Atom.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    with open_dict(cfg):
        if cfg.pocket_only_baseline:
            cfg.input_dir = cfg.input_dir.replace(cfg.dataset, f"{cfg.dataset}_pocket_only")
            cfg.output_dir = cfg.output_dir.replace(cfg.dataset, f"{cfg.dataset}_pocket_only")

        if cfg.max_num_inputs:
            cfg.output_dir = cfg.output_dir.replace(
                cfg.dataset, f"{cfg.dataset}_first_{cfg.max_num_inputs}"
            )

    if cfg.run_inference_directly:
        num_dir_items_found = 0
        for item in os.listdir(cfg.input_dir):
            item_path = os.path.join(cfg.input_dir, item)
            if os.path.isdir(item_path):
                num_dir_items_found += 1
                if cfg.max_num_inputs and num_dir_items_found > cfg.max_num_inputs:
                    logger.info(
                        f"Maximum number of input directories reached ({cfg.max_num_inputs}). Exiting inference loop."
                    )
                    break
                if (
                    cfg.skip_existing
                    and os.path.exists(os.path.join(cfg.output_dir, item, f"{item}.pdb"))
                    and not os.path.exists(os.path.join(cfg.output_dir, item, "error_log.txt"))
                ):
                    logger.info(
                        f"Skipping inference for `{item}` as output directory already exists."
                    )
                    continue
                fasta_filepaths = sorted(
                    list(glob.glob(os.path.join(item_path, "*.fasta"))),
                    key=protein_file_sort_key,
                )
                sdf_filepaths = sorted(
                    list(glob.glob(os.path.join(item_path, "*.sdf"))), key=ligand_file_sort_key
                )
                smiles_strings = [
                    Chem.MolToSmiles(Chem.MolFromMolFile(sdf)) for sdf in sdf_filepaths
                ]
                if not fasta_filepaths or not sdf_filepaths or not smiles_strings:
                    logger.error(
                        f"Failed to find all required files for item `{item}`. Skipping..."
                    )
                    continue
                assert not any(
                    smiles is None for smiles in smiles_strings
                ), f"Failed to parse all SMILES strings from ligand files for item `{item}`."
                # dynamically build the RoseTTAFold-All-Atom inference configuration files
                config_filepath = dynamically_build_rfaa_input_config(
                    fasta_filepaths=fasta_filepaths,
                    sdf_filepaths=sdf_filepaths,
                    input_id=item,
                    cfg=cfg,
                    smiles_strings=smiles_strings,
                )
                with open_dict(cfg):
                    cfg.inference_config_name = os.path.basename(config_filepath)
                    cfg.inference_dir_name = item
                try:
                    run_rfaa_inference_directly(cfg)
                    if os.path.isfile(os.path.join(cfg.output_dir, item, "error_log.txt")):
                        os.remove(os.path.join(cfg.output_dir, item, "error_log.txt"))
                except Exception as e:
                    logger.error(
                        f"Failed to run RoseTTAFold-All-Atom inference for item `{item}` due to: {e}. Skipping..."
                    )
                    with open(os.path.join(cfg.output_dir, item, "error_log.txt"), "w") as f:
                        traceback.print_exception(type(e), e, e.__traceback__, file=f)
    else:
        num_dir_items_found = 0
        for item in os.listdir(cfg.input_dir):
            item_path = os.path.join(cfg.input_dir, item)
            if os.path.isdir(item_path):
                num_dir_items_found += 1
                if cfg.max_num_inputs and num_dir_items_found > cfg.max_num_inputs:
                    logger.info(
                        f"Maximum number of input directories reached ({cfg.max_num_inputs}). Exiting inference loop."
                    )
                    break
                fasta_filepaths = sorted(
                    list(glob.glob(os.path.join(item_path, "*.fasta"))), key=protein_file_sort_key
                )
                sdf_filepaths = sorted(
                    list(glob.glob(os.path.join(item_path, "*.sdf"))), key=ligand_file_sort_key
                )
                create_rfaa_bash_script(fasta_filepaths, sdf_filepaths, item, cfg)
        logger.info(f"RoseTTAFold-All-Atom inference scripts written to `{cfg.output_dir}`.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
