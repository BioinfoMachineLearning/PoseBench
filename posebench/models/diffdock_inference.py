# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
import subprocess  # nosec

import hydra
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="diffdock_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained DiffDock model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_csv_path = (
        cfg.input_csv_path.replace(".csv", f"_first_{cfg.max_num_inputs}.csv")
        if cfg.max_num_inputs
        else cfg.input_csv_path
    )

    with open_dict(cfg):
        if cfg.v1_baseline:
            cfg.diffdock_exec_dir = cfg.diffdock_exec_dir.replace("DiffDock", "DiffDockv1")
            cfg.input_csv_path = cfg.input_csv_path.replace("DiffDock", "DiffDockv1")
            cfg.model_dir = cfg.model_dir.replace(
                "forks/DiffDock/workdir/v1.1/score_model",
                "forks/DiffDockv1/workdir/paper_score_model",
            )
            cfg.confidence_model_dir = cfg.confidence_model_dir.replace(
                "forks/DiffDock/workdir/v1.1/confidence_model",
                "forks/DiffDockv1/workdir/paper_confidence_model",
            )
            cfg.output_dir = cfg.output_dir.replace("DiffDock", "DiffDockv1")
            cfg.actual_steps = 18
            cfg.no_final_step_noise = True

        if cfg.pocket_only_baseline:
            input_csv_path = input_csv_path.replace(
                f"diffdock_{cfg.dataset}", f"diffdock_pocket_only_{cfg.dataset}"
            )
            cfg.output_dir = cfg.output_dir.replace(
                f"diffdock_{cfg.dataset}", f"diffdock_pocket_only_{cfg.dataset}"
            )

        if cfg.max_num_inputs:
            cfg.output_dir = cfg.output_dir.replace(
                f"_{cfg.dataset}", f"_{cfg.dataset}_first_{cfg.max_num_inputs}"
            )

    assert os.path.exists(input_csv_path), f"Input CSV file `{input_csv_path}` not found."
    try:
        cmd = [
            cfg.python_exec_path,
            os.path.join(cfg.diffdock_exec_dir, "inference.py"),
            "--protein_ligand_csv",
            input_csv_path,
            "--out_dir",
            cfg.output_dir,
            "--inference_steps",
            str(cfg.inference_steps),
            "--samples_per_complex",
            str(cfg.samples_per_complex),
            "--batch_size",
            str(cfg.batch_size),
            "--actual_steps",
            str(cfg.actual_steps),
            "--no_final_step_noise" if cfg.no_final_step_noise else "",
            "--cuda_device_index",
            str(cfg.cuda_device_index),
            "--model_dir",
            cfg.model_dir,
            "--confidence_model_dir",
            cfg.confidence_model_dir,
        ]
        if cfg.skip_existing:
            cmd.append("--skip_existing")
        if not cfg.v1_baseline:
            cmd.extend(["--config", cfg.inference_config_path])
        subprocess.run(cmd, check=True)  # nosec
    except Exception as e:
        raise e
    logger.info(f"DiffDock inference for CSV input file `{input_csv_path}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
