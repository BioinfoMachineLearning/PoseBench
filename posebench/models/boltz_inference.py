# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import subprocess  # nosec
import traceback

import hydra
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_boltz_inference(fasta_file: str, cfg: DictConfig):
    """Run inference using a trained Boltz model checkpoint.

    :param fasta_filepath: Path to the input FASTA file.
    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    assert os.path.exists(fasta_file), f"FASTA file `{fasta_file}` not found."
    os.makedirs(cfg.output_dir, exist_ok=True)
    try:
        cmd = [
            "boltz",
            "predict",
            fasta_file,
            "--out_dir",
            cfg.output_dir,
            "--model",
            cfg.model,
        ]
        if cfg.use_potentials:
            cmd.append("--use_potentials")
        subprocess.run(cmd, check=True)  # nosec
    except Exception as e:
        raise e
    logger.info(f"Boltz inference for FASTA file `{fasta_file}` complete.")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="boltz_inference.yaml",
)
def main(cfg: DictConfig):
    """Create SLURM job submission scripts for inference with Boltz.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    assert cfg.model in [
        "boltz1",
        "boltz2",
    ], f"Invalid model `{cfg.model}` specified. Must be one of (`boltz1`, `boltz2`)."

    with open_dict(cfg):
        if cfg.pocket_only_baseline:
            cfg.input_dir = cfg.input_dir.replace(cfg.dataset, f"{cfg.dataset}_pocket_only")
            cfg.output_dir = cfg.output_dir.replace(cfg.dataset, f"{cfg.dataset}_pocket_only")

        if cfg.max_num_inputs:
            cfg.output_dir = cfg.output_dir.replace(
                cfg.dataset, f"{cfg.dataset}_first_{cfg.max_num_inputs}"
            )

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
                and os.path.exists(
                    os.path.join(
                        cfg.output_dir,
                        f"boltz_results_{item}",
                        "predictions",
                        item,
                        f"{item}_model_0.cif",
                    )
                )
                and not os.path.exists(
                    os.path.join(cfg.output_dir, f"boltz_results_{item}", "error_log.txt")
                )
            ):
                logger.info(f"Skipping inference for `{item}` as output directory already exists.")
                continue
            fasta_filepaths = list(glob.glob(os.path.join(item_path, "*.fasta")))
            if not fasta_filepaths:
                logger.error(f"Failed to find all required files for item `{item}`. Skipping...")
                continue
            fasta_filepath = fasta_filepaths[0]
            try:
                run_boltz_inference(
                    fasta_file=fasta_filepath,
                    cfg=cfg,
                )
                if os.path.isfile(
                    os.path.join(cfg.output_dir, f"boltz_results_{item}", item, "error_log.txt")
                ):
                    os.remove(
                        os.path.join(
                            cfg.output_dir, f"boltz_results_{item}", item, "error_log.txt"
                        )
                    )
            except Exception as e:
                logger.error(
                    f"Failed to run Boltz inference for item `{item}` due to: {e}. Skipping..."
                )
                with open(
                    os.path.join(cfg.output_dir, f"boltz_results_{item}", "error_log.txt"), "w"
                ) as f:
                    traceback.print_exception(type(e), e, e.__traceback__, file=f)

    logger.info("Boltz inference complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
