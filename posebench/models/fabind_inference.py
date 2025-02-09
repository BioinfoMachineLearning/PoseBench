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
    config_name="fabind_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained FABind model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_csv_path = (
        cfg.input_csv_path.replace(".csv", f"_first_{cfg.max_num_inputs}.csv")
        if cfg.max_num_inputs
        else cfg.input_csv_path
    )

    with open_dict(cfg):
        if cfg.pocket_only_baseline:
            cfg.save_mols_dir = cfg.save_mols_dir.replace(
                f"fabind_{cfg.dataset}", f"fabind_pocket_only_{cfg.dataset}"
            )
            cfg.save_pt_dir = cfg.save_pt_dir.replace(
                f"fabind_{cfg.dataset}", f"fabind_pocket_only_{cfg.dataset}"
            )
            cfg.output_dir = cfg.output_dir.replace(
                f"fabind_{cfg.dataset}", f"fabind_pocket_only_{cfg.dataset}"
            )

        if cfg.max_num_inputs:
            cfg.output_dir = cfg.output_dir.replace(
                f"_{cfg.dataset}", f"_first_{cfg.max_num_inputs}_{cfg.dataset}"
            )

    assert os.path.exists(input_csv_path), f"Input CSV file `{input_csv_path}` not found."
    try:
        subprocess.run(
            [
                cfg.python_exec_path,
                os.path.join(cfg.fabind_exec_dir, "inference_preprocess_mol_confs.py"),
                "--index_csv",
                input_csv_path,
                "--save_mols_dir",
                cfg.save_mols_dir,
                "--num_threads",
                str(cfg.num_threads),
            ],
            check=True,
        )  # nosec
    except Exception as e:
        raise e
    logger.info(f"FABind molecule preprocessing for CSV input file `{input_csv_path}` complete.")

    try:
        subprocess.run(
            [
                cfg.python_exec_path,
                os.path.join(cfg.fabind_exec_dir, "inference_preprocess_protein.py"),
                "--pdb_file_dir",
                (
                    f"{cfg.input_data_dir}_bs_cropped"
                    if cfg.pocket_only_baseline
                    else cfg.input_data_dir
                ),
                "--save_pt_dir",
                cfg.save_pt_dir,
                "--cuda_device_index",
                str(cfg.cuda_device_index),
            ],
            check=True,
        )  # nosec
    except Exception as e:
        raise e
    logger.info(f"FABind protein preprocessing for CSV input file `{input_csv_path}` complete.")

    try:
        subprocess.run(
            [
                cfg.python_exec_path,
                os.path.join(cfg.fabind_exec_dir, "fabind_inference.py"),
                "--ckpt",
                cfg.ckpt_path,
                "--batch_size",
                "4",
                "--seed",
                "128",
                "--test-gumbel-soft",
                "--redocking",
                "--post-optim",
                "--write-mol-to-file",
                "--sdf-output-path-post-optim",
                cfg.output_dir,
                "--index-csv",
                input_csv_path,
                "--preprocess-dir",
                cfg.save_pt_dir,
                "--cuda_device_index",
                str(cfg.cuda_device_index),
            ],
            check=True,
        )  # nosec
    except Exception as e:
        raise e
    logger.info(f"FABind inference for CSV input file `{input_csv_path}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
