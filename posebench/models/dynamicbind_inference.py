# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import subprocess  # nosec
import uuid
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.utils import find_ligand_files, find_protein_files

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="dynamicbind_inference.yaml",
)
def main(cfg: DictConfig):
    """Run inference using a trained DynamicBind model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    pocket_suffix = "_bs_cropped" if cfg.pocket_only_baseline else ""
    pocket_only_suffix = "_pocket_only" if cfg.pocket_only_baseline else ""
    max_num_inputs_suffix = f"_first_{cfg.max_num_inputs}" if cfg.max_num_inputs else ""

    os.environ["MKL_THREADING_LAYER"] = "GNU"  # address MKL threading issue
    protein_filepaths = find_protein_files(Path(cfg.input_data_dir + pocket_suffix))
    ligand_filepaths = [
        ligand_filepath
        for ligand_filepath in find_ligand_files(Path(cfg.input_ligand_csv_dir), extension="csv")
        if any(
            ligand_filepath.stem.split("_")[0] in protein_filepath.stem
            for protein_filepath in protein_filepaths
        )
    ]
    if len(protein_filepaths) > len(ligand_filepaths):
        protein_filepaths = [
            protein_filepath
            for protein_filepath in protein_filepaths
            if (
                cfg.dataset == "dockgen"
                and any(
                    "_".join(ligand_filepath.stem.split("_")[:4]) in protein_filepath.stem
                    for ligand_filepath in ligand_filepaths
                )
            )
            or (
                cfg.dataset != "dockgen"
                and any(
                    ligand_filepath.stem.split("_")[0] in protein_filepath.stem
                    for ligand_filepath in ligand_filepaths
                )
            )
        ]
    for ligand_filepath in ligand_filepaths:
        if len(protein_filepaths) < len(ligand_filepaths):
            protein_filepaths.append(
                next(
                    protein_filepath
                    for protein_filepath in protein_filepaths
                    if (
                        "_".join(cfg.dataset == "dockgen" and ligand_filepath.stem.split("_")[:4])
                        in protein_filepath.stem
                    )
                    or (
                        cfg.dataset != "dockgen"
                        and ligand_filepath.stem.split("_")[0] in protein_filepath.stem
                    )
                )
            )
    assert len(protein_filepaths) == len(
        ligand_filepaths
    ), f"Number of protein ({len(protein_filepaths)}) and ligand ({len(ligand_filepaths)}) files must be equal."
    protein_filepaths = sorted(protein_filepaths)
    ligand_filepaths = sorted(ligand_filepaths)
    if cfg.max_num_inputs and protein_filepaths and ligand_filepaths:
        protein_filepaths = protein_filepaths[: cfg.max_num_inputs]
        ligand_filepaths = ligand_filepaths[: cfg.max_num_inputs]
        assert (
            len(protein_filepaths) > 0 and len(ligand_filepaths) > 0
        ), "No input files found after subsetting with `max_num_inputs`."
    for protein_filepath, ligand_filepath in zip(protein_filepaths, ligand_filepaths):
        assert (
            protein_filepath.stem.split("_")[0] == ligand_filepath.stem.split("_")[0]
        ), "Protein and ligand files must have the same ID."
        if cfg.dataset == "dockgen":
            assert "_".join(protein_filepath.stem.split("_")[:4]) == "_".join(
                ligand_filepath.stem.split("_")[:4]
            ), "Protein and ligand files must have the same ID."
        ligand_output_filepaths = list(
            glob.glob(
                os.path.join(
                    cfg.dynamicbind_exec_dir,
                    "inference",
                    "outputs",
                    "results",
                    f"{cfg.dataset}{pocket_only_suffix}{max_num_inputs_suffix}_{ligand_filepath.stem}_{cfg.repeat_index}",
                    "index0_idx_0",
                    "rank1_ligand*.sdf",
                )
            )
        )
        if cfg.skip_existing and ligand_output_filepaths:
            logger.info(
                f"Skipping inference for completed protein `{protein_filepath}` and ligand `{ligand_filepath}`."
            )
            continue
        unique_cache_id = uuid.uuid4()
        unique_cache_path = (
            str(cfg.cache_path)
            + f"_{cfg.dataset}{pocket_only_suffix}{max_num_inputs_suffix}_{ligand_filepath.stem}_{cfg.repeat_index}_{unique_cache_id}"
        )
        try:
            subprocess.run(
                [
                    cfg.python_exec_path,
                    os.path.join(cfg.dynamicbind_exec_dir, "run_single_protein_inference.py"),
                    protein_filepath,
                    ligand_filepath,
                    "--samples_per_complex",
                    str(cfg.samples_per_complex),
                    "--savings_per_complex",
                    str(cfg.savings_per_complex),
                    "--inference_steps",
                    str(cfg.inference_steps),
                    "--batch_size",
                    str(cfg.batch_size),
                    "--cache_path",
                    unique_cache_path,
                    "--header",
                    str(cfg.header)
                    + f"{pocket_only_suffix}{max_num_inputs_suffix}_{ligand_filepath.stem}"
                    + f"_{cfg.repeat_index}",
                    "--device",
                    str(cfg.cuda_device_index),
                    "--python",
                    str(cfg.python_exec_path),
                    "--relax_python",
                    str(cfg.python_exec_path),
                    "--results",
                    str(os.path.join(cfg.dynamicbind_exec_dir, "inference", "outputs", "results")),
                    "--no_relax",  # NOTE: must be set to `True` within `PoseBench` since method-native relaxation is not supported
                    "--paper",  # NOTE: must be set to `True` within `PoseBench` since only the paper weights are available
                ],
                check=True,
            )  # nosec
        except Exception as e:
            logger.error(
                f"Error occurred while running DynamicBind inference for protein `{protein_filepath}` and ligand `{ligand_filepath}`: {e}. Skipping..."
            )
            continue
        logger.info(
            f"DynamicBind inference for protein `{protein_filepath}` and `{ligand_filepath}` complete."
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
