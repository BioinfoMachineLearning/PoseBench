# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
import subprocess  # nosec
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from posebench import utils`)
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

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_usalign_metrics(
    pred_pdb_filepath: str,
    reference_pdb_filepath: str,
    usalign_exec_path: str,
    flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates US-align structural metrics between predicted and reference
    macromolecular structures.

    :param pred_pdb_filepath: Filepath to predicted macromolecular
        structure in PDB format.
    :param reference_pdb_filepath: Filepath to reference macromolecular
        structure in PDB format.
    :param usalign_exec_path: Path to US-align executable.
    :param flags: Command-line flags to pass to US-align, optional.
    :return: Dictionary containing macromolecular US-align structural
        metrics and metadata.
    """
    # run US-align with subprocess and capture output
    cmd = [usalign_exec_path, pred_pdb_filepath, reference_pdb_filepath]
    if flags is not None:
        cmd += flags
    output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)  # nosec

    # parse US-align output to extract structural metrics
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Name of Structure_1:"):
            metrics["Name of Structure_1"] = line.split(": ", 1)[1]
        elif line.startswith("Name of Structure_2:"):
            metrics["Name of Structure_2"] = line.split(": ", 1)[1]
        elif line.startswith("Length of Structure_1:"):
            metrics["Length of Structure_1"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Length of Structure_2:"):
            metrics["Length of Structure_2"] = int(line.split(": ")[1].split()[0])
        elif line.startswith("Aligned length="):
            aligned_length = line.split("=")[1].split(",")[0]
            rmsd = line.split("=")[2].split(",")[0]
            seq_id = line.split("=")[4]
            metrics["Aligned length"] = int(aligned_length.strip())
            metrics["RMSD"] = float(rmsd.strip())
            metrics["Seq_ID"] = float(seq_id.strip())
        elif line.startswith("TM-score="):
            if "normalized by length of Structure_1" in line:
                metrics["TM-score_1"] = float(line.split("=")[1].split()[0])
            elif "normalized by length of Structure_2" in line:
                metrics["TM-score_2"] = float(line.split("=")[1].split()[0])

    return metrics


def plot_dataset_rmsd(
    dataset: str,
    dataset_name: str,
    pred_pdb_dir: str,
    ref_pdb_dir: str,
    output_dir: str,
    usalign_exec_path: str,
    usalign_flags: Optional[List[str]] = [
        "-mol",
        "prot",  # NOTE: align only protein chains
        "-ter",
        "0",  # NOTE: biological unit alignment
        "-split",
        "0",  # NOTE: treat a whole file as a single chain
        "-se",  # NOTE: calculate TMscore without alignment
    ],
    filtered_ids_to_keep_file: Optional[str] = None,
    filtered_ids_to_skip: Optional[Set[str]] = None,
    is_casp_dataset: bool = False,
    public_plots: bool = True,
    accurate_rmsd_threshold: float = 4.0,
    accurate_tm_score_threshold: float = 0.7,
):
    """Plot the RMSD between predicted and reference protein structures in a
    given dataset.

    :param dataset: Informal name of the dataset.
    :param dataset_name: Formal name of the dataset.
    :param pred_pdb_dir: Directory containing predicted protein
        structures in PDB format.
    :param ref_pdb_dir: Directory containing reference protein
        structures in PDB format.
    :param output_dir: Directory to save the plots.
    :param usalign_exec_path: Path to the US-align executable.
    :param usalign_flags: Command-line flags to pass to US-align.
    :param filtered_ids_to_keep_file: File containing IDs of sequences
        to keep.
    :param filtered_ids_to_skip: Set of IDs of sequences to skip.
    :param is_casp_dataset: Whether the dataset is a CASP dataset.
    :param public_plots: Whether to save the public versions of the
        plots.
    :param accurate_rmsd_threshold: RMSD threshold for accurate
        predictions.
    :param accurate_tm_score_threshold: TM-score threshold for accurate
        predictions.
    """

    # Filter out sequences that are not in the filtered_ids_file

    filtered_ids_to_keep = None

    if filtered_ids_to_keep_file:
        with open(filtered_ids_to_keep_file) as f:
            filtered_ids_to_keep = set(f.read().splitlines())

    # Collect the RMSD values for each predicted protein structure in the dataset

    dataset_rows = []

    dataset_suffix = " (Public)" if is_casp_dataset and public_plots else ""

    for pred_pdb_file in tqdm(
        os.listdir(pred_pdb_dir),
        desc=f"Plotting RMSD for {dataset_name}{dataset_suffix}",
    ):
        pdb_id = os.path.splitext(os.path.basename(pred_pdb_file))[0].split("_holo")[0]

        if filtered_ids_to_keep is not None and pdb_id not in filtered_ids_to_keep:
            logging.info(f"Skipping {pdb_id} as it is not in the filtered IDs to keep file")
            continue

        if filtered_ids_to_skip is not None and pdb_id in filtered_ids_to_skip:
            logging.info(f"Skipping {pdb_id} as it is in the filtered IDs to skip")
            continue

        pred_pdb_filepath = os.path.join(pred_pdb_dir, pred_pdb_file)
        if is_casp_dataset:
            ref_pdb_filepath = os.path.join(ref_pdb_dir, f"{pdb_id}_lig.pdb")
            if not os.path.exists(ref_pdb_filepath):
                logging.info(f"CASP reference PDB file not found: {ref_pdb_filepath}")
                continue
        else:
            ref_pdb_filepath = os.path.join(ref_pdb_dir, pdb_id, f"{pdb_id}_protein.pdb")

        assert os.path.exists(
            pred_pdb_filepath
        ), f"Predicted PDB file not found: {pred_pdb_filepath}"
        if not os.path.exists(ref_pdb_filepath) and os.path.exists(
            ref_pdb_filepath.replace(".pdb", "_processed.pdb")
        ):
            # Handle DockGen's protein file naming convention
            ref_pdb_filepath = ref_pdb_filepath.replace(".pdb", "_processed.pdb")
        assert os.path.exists(
            ref_pdb_filepath
        ), f"Reference PDB file not found: {ref_pdb_filepath}"

        tm_score_metrics = calculate_usalign_metrics(
            pred_pdb_filepath, ref_pdb_filepath, usalign_exec_path, flags=usalign_flags
        )

        dataset_rows.append(
            {
                "pdb_id": pdb_id,
                "pred_pdb_filepath": pred_pdb_filepath,
                "ref_pdb_filepath": ref_pdb_filepath,
                "RMSD": tm_score_metrics["RMSD"],
                "TM-score": tm_score_metrics["TM-score_2"],
            }
        )

    dataset_df = pd.DataFrame(dataset_rows)

    # Plot the RMSD values

    accurate_predictions_percent = (
        dataset_df[
            (dataset_df["RMSD"] < accurate_rmsd_threshold)
            & (dataset_df["TM-score"] > accurate_tm_score_threshold)
        ].shape[0]
        / dataset_df.shape[0]
    )
    logging.info(
        f"For the {dataset_name}{dataset_suffix} dataset, {accurate_predictions_percent * 100:.2f}% of the predictions have RMSD < {accurate_rmsd_threshold} and TM-score > {accurate_tm_score_threshold}."
    )

    plot_dir = Path(output_dir) / ("public_plots" if is_casp_dataset and public_plots else "plots")
    plot_dir.mkdir(exist_ok=True)

    plt.clf()
    sns.histplot(dataset_df["TM-score"])
    plt.title("Apo-To-Holo Protein TM-score")
    plt.savefig(plot_dir / f"{dataset}_a2h_TM-score_hist.png")

    plt.clf()
    sns.histplot(dataset_df["RMSD"])
    plt.title("Apo-To-Holo Protein RMSD")
    plt.savefig(plot_dir / f"{dataset}_a2h_RMSD_hist.png")


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="plot_dataset_rmsd.yaml",
)
def main(cfg: DictConfig):
    """Main function to plot RMSD values for predicted protein structures of
    different datasets."""

    # NOTE: If USalign is not already available locally, follow the following steps to install it:
    # Install US-align to align macromolecular structures
    # cd $MY_PROGRAMS_DIR  # download US-align to your choice of directory (e.g., `~/Programs/`)
    # git clone https://github.com/pylelab/USalign.git && cd USalign/ && git checkout 97325d3aad852f8a4407649f25e697bbaa17e186
    # g++ -static -O3 -ffast-math -lm -o USalign USalign.cpp
    # NOTE: Make sure to update the `usalign_exec_path` value in `configs/data/components/plot_dataset_rmsd.yaml` to reflect where you have placed the US-align executable on your machine.

    plot_dataset_rmsd(
        "astex_diverse",
        "Astex Diverse Set",
        os.path.join(
            cfg.data_dir,
            "astex_diverse_set",
            "astex_diverse_holo_aligned_predicted_structures",
        ),
        os.path.join(cfg.data_dir, "astex_diverse_set"),
        os.path.join(
            cfg.data_dir,
            "astex_diverse_set",
        ),
        usalign_exec_path=cfg.usalign_exec_path,
    )

    plot_dataset_rmsd(
        "posebusters_benchmark",
        "PoseBusters Benchmark Set",
        os.path.join(
            cfg.data_dir,
            "posebusters_benchmark_set",
            "posebusters_benchmark_holo_aligned_predicted_structures",
        ),
        os.path.join(cfg.data_dir, "posebusters_benchmark_set"),
        os.path.join(
            cfg.data_dir,
            "posebusters_benchmark_set",
        ),
        usalign_exec_path=cfg.usalign_exec_path,
        filtered_ids_to_keep_file=os.path.join(cfg.data_dir, "posebusters_pdb_ccd_ids.txt"),
    )

    plot_dataset_rmsd(
        "dockgen",
        "DockGen Set",
        os.path.join(cfg.data_dir, "dockgen_set", "dockgen_holo_aligned_predicted_structures"),
        os.path.join(cfg.data_dir, "dockgen_set"),
        os.path.join(
            cfg.data_dir,
            "dockgen_set",
        ),
        usalign_exec_path=cfg.usalign_exec_path,
        filtered_ids_to_keep_file=os.path.join(cfg.data_dir, "dockgen_set", "split_test.txt"),
    )

    plot_dataset_rmsd(
        "casp15",
        "CASP15 Set",
        os.path.join(cfg.data_dir, "casp15_set", "casp15_holo_aligned_predicted_structures"),
        os.path.join(cfg.data_dir, "casp15_set", "targets"),
        os.path.join(
            cfg.data_dir,
            "casp15_set",
        ),
        usalign_exec_path=cfg.usalign_exec_path,
        filtered_ids_to_skip={
            "T1127v2",
            "T1146",
            "T1170",
            "T1181",
            "T1186",
        },  # NOTE: We don't score `T1170` due to CASP internal parsing issues
        is_casp_dataset=True,
        public_plots=True,
    )

    # plot_dataset_rmsd(
    #     "casp15",
    #     "CASP15 Set",
    #     os.path.join(cfg.data_dir, "casp15_set", "casp15_holo_aligned_predicted_structures"),
    #     os.path.join(cfg.data_dir, "casp15_set", "targets"),
    #     os.path.join(
    #         cfg.data_dir,
    #         "casp15_set",
    #     ),
    #     usalign_exec_path=cfg.usalign_exec_path,
    #     filtered_ids_to_skip={
    #         "T1170",
    #     },  # NOTE: We don't score `T1170` due to CASP internal parsing issues
    #     is_casp_dataset=True,
    #     public_plots=False,
    # )


if __name__ == "__main__":
    main()
