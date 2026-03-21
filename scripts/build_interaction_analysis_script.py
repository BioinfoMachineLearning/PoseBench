# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from typing import List, Literal

import hydra
import rootutils
from beartype import beartype
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.models.ensemble_generation import insert_hpc_headers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


INTERACTION_ANALYSIS_METHODS = Literal[
    "vina_p2rank",
    "diffdock",
    "dynamicbind",
    "neuralplexer",
    "rfaa",
    "chai-lab_ss",
    "chai-lab",
    "boltz_ss",
    "boltz",
    "alphafold3_ss",
    "alphafold3",
]
INTERACTION_ANALYSIS_DATASETS = Literal[
    "astex_diverse",
    "casp15",
    "dockgen",
    "posebusters_benchmark",
]

DATASET_TO_NOTEBOOK = {
    "astex_diverse": "notebooks/astex_method_interaction_analysis_plotting_slurm.py",
    "casp15": "notebooks/casp15_method_interaction_analysis_plotting_slurm.py",
    "dockgen": "notebooks/dockgen_method_interaction_analysis_plotting_slurm.py",
    "posebusters_benchmark": "notebooks/posebusters_method_interaction_analysis_plotting_slurm.py",
}


@beartype
def build_interaction_analysis_script(
    method: INTERACTION_ANALYSIS_METHODS,
    dataset: INTERACTION_ANALYSIS_DATASETS,
    output_script_dir: str,
    repeat_index: int = 1,
):
    """Build a SLURM script to preprocess one method's interactions for one
    dataset."""
    if repeat_index != 1:
        raise ValueError("Only repeat_index=1 is supported for interaction preprocessing.")

    os.makedirs(output_script_dir, exist_ok=True)
    output_script = os.path.join(
        output_script_dir,
        f"{method}_{dataset}_hpc_interaction_analysis_{repeat_index}.sh",
    )

    with open(output_script, "w") as f:
        f.write("#!/bin/bash -l\n\n")
        f.write(insert_hpc_headers(method="diffdock"))
        f.write(
            "# Store model weights in a larger storage location\n"
            + 'export TORCH_HOME="/pscratch/sd/a/$USER/torch_cache"\n'
            + 'export HF_HOME="/pscratch/sd/a/$USER/hf_cache"\n\n'
            + 'mkdir -p "$TORCH_HOME"\n'
            + 'mkdir -p "$HF_HOME"\n\n'
        )
        f.write("# Preprocess method interaction H5 files only\n")
        f.write(
            "srun --kill-on-bad-exit=1 shifter "
            f"python3 {DATASET_TO_NOTEBOOK[dataset]} --method {method} --exit-after-preprocessing\n\n"
        )
        f.write(f"echo 'Interaction preprocessing for {method} on {dataset} completed.'\n")

    os.chmod(output_script, 0o755)  # nosec
    logger.info(f"Script {output_script} created successfully.")


@beartype
def build_interaction_analysis_scripts(
    methods_to_sweep: List[INTERACTION_ANALYSIS_METHODS],
    datasets_to_sweep: List[INTERACTION_ANALYSIS_DATASETS],
    output_script_dir: str,
    repeat_index: int = 1,
):
    """Build interaction preprocessing scripts for a method-dataset sweep."""
    for method in methods_to_sweep:
        for dataset in datasets_to_sweep:
            build_interaction_analysis_script(
                method=method,
                dataset=dataset,
                output_script_dir=output_script_dir,
                repeat_index=repeat_index,
            )


@hydra.main(
    version_base="1.3",
    config_path="../configs/scripts",
    config_name="build_interaction_analysis_script.yaml",
)
def main(cfg: DictConfig):
    """Build interaction analysis scripts according to user arguments."""
    if cfg.sweep:
        build_interaction_analysis_scripts(
            methods_to_sweep=list(cfg.methods_to_sweep),
            datasets_to_sweep=list(cfg.datasets_to_sweep),
            output_script_dir=cfg.output_script_dir,
            repeat_index=cfg.repeat_index,
        )
    else:
        build_interaction_analysis_script(
            method=cfg.method,
            dataset=cfg.dataset,
            output_script_dir=cfg.output_script_dir,
            repeat_index=cfg.repeat_index,
        )


if __name__ == "__main__":
    main()
