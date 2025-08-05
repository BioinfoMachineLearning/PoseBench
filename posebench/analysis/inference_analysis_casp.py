# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import shutil
import subprocess  # nosec
import tempfile
from pathlib import Path
from typing import Tuple

import hydra
import pandas as pd
import rootutils
from beartype.typing import List
from omegaconf import DictConfig, open_dict
from posebusters import PoseBusters

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers, resolve_method_title
from posebench.utils.data_utils import renumber_pdb_df_residues

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CASP_BUST_TEST_COLUMNS = [
    # chemical validity and consistency #
    "mol_pred_loaded",
    "mol_cond_loaded",
    "sanitization",
    # "all_atoms_connected",  # NOTE: for CASP15 targets, we ignore this test since it is not applicable to complexes containing ions
    # intramolecular validity #
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    # intermolecular validity #
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]
All_CASP15_SINGLE_LIGAND_TARGETS = [
    "T1146",
    "T1152",
    "T1158v1",
    "T1158v2",
    "T1158v3",
    "T1186",
]
PUBLIC_CASP15_SINGLE_LIGAND_TARGETS = [
    "T1152",
    "T1158v1",
    "T1158v2",
    "T1158v3",
]
All_CASP15_MULTI_LIGAND_TARGETS = [
    "H1135",
    "H1171v1",
    "H1171v2",
    "H1172v1",
    "H1172v2",
    "H1172v3",
    "H1172v4",
    "T1124",
    "T1127v2",
    "T1158v4",
    "T1181",
    "T1187",
    "T1188",
]
PUBLIC_CASP15_MULTI_LIGAND_TARGETS = [
    "H1135",
    "H1171v1",
    "H1171v2",
    "H1172v1",
    "H1172v2",
    "H1172v3",
    "H1172v4",
    "T1124",
    "T1158v4",
    "T1187",
    "T1188",
]
NUM_SCOREABLE_CASP15_TARGETS = len(All_CASP15_SINGLE_LIGAND_TARGETS) + len(
    All_CASP15_MULTI_LIGAND_TARGETS
)


def create_casp_input_dirs(cfg: DictConfig, config: str) -> Tuple[str, List[str]]:
    """Create the input directories for the CASP ligand scoring pipeline and
    return the resulting (temporary) parent directory as a `Path`.

    :param cfg: Configuration dictionary from the hydra YAML file.
    :param config: The configuration suffix to append to the output directory.
    :return: The path to the temporary parent directory as a `Path` as well as
        a list of available prediction targets for "tolerant methods".
    """
    target_ids = []
    temp_dir_path = Path(
        tempfile.mkdtemp(
            suffix=f"_{cfg.method}_{cfg.vina_binding_site_method}_{cfg.dataset}{config}"
        )
    )
    (temp_dir_path / "predictions").mkdir(parents=True, exist_ok=True)
    (temp_dir_path / "targets").mkdir(parents=True, exist_ok=True)
    target_dir_path = Path(cfg.dataset_dir) / "targets"
    for pred_dir in os.listdir(cfg.predictions_dir):
        if not config == "_relaxed" and pred_dir.endswith("_relaxed"):
            continue
        pred_dir_path = Path(cfg.predictions_dir) / (pred_dir + config)
        if pred_dir_path.is_dir():
            for item in os.listdir(pred_dir_path):
                if not item.endswith((".csv", ".pdb", ".sdf")):
                    (temp_dir_path / "predictions" / pred_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy(pred_dir_path / item, temp_dir_path / "predictions" / pred_dir)
        target_files = glob.glob(str(target_dir_path / f"{pred_dir}*"))
        for target_file in target_files:
            if target_file.endswith("_lig.pdb"):
                renumber_pdb_df_residues(
                    target_file, str(temp_dir_path / "targets" / Path(target_file).name)
                )
                target_ids.append(Path(target_file).stem.split("_")[0])
            else:
                shutil.copy(target_file, temp_dir_path / "targets")
    return temp_dir_path, target_ids


def create_casp_mol_table(
    input_data_dir: Path,
    targets_to_select: List[str],
    relaxed: bool = False,
    relax_protein: bool = False,
    rank_to_select: int = 1,
) -> pd.DataFrame:
    """Create a table of CASP molecules and their corresponding ligand files.

    :param input_data_dir: Path to the input data directory.
    :param targets_to_select: List of targets to select.
    :param relaxed: Whether to use the relaxed poses.
    :param rank_to_select: The rank of the complex to select.
    :return: Molecule table DataFrame.
    """
    mol_table_rows = []
    for item in os.listdir(input_data_dir):
        data_dir = input_data_dir / item
        if data_dir.is_dir() and item.replace("_relaxed", "") in targets_to_select:
            if relaxed and "_relaxed" not in item or not relaxed and "_relaxed" in item:
                continue
            sdf_data_files = glob.glob(str(data_dir / f"*_rank{rank_to_select}_*.sdf"))
            pdb_data_files = glob.glob(str(data_dir / f"*_rank{rank_to_select}_*.pdb"))
            if relax_protein:
                pdb_data_files = glob.glob(str(data_dir / f"*_rank{rank_to_select}_*_relaxed.pdb"))
            assert (
                len(sdf_data_files) == 1
            ), f"Expected 1 SDF file, but found {len(sdf_data_files)}: {sdf_data_files}."
            assert (
                len(pdb_data_files) == 1
            ), f"Expected 1 PDB file, but found {len(pdb_data_files)}: {pdb_data_files}."
            sdf_data_file = sdf_data_files[0]
            pdb_data_file = pdb_data_files[0]
            mol_table_rows.append(
                {
                    "mol_cond": pdb_data_file,
                    "mol_pred": sdf_data_file,
                    "mol_true": None,
                }
            )
    mol_table = pd.DataFrame(mol_table_rows)
    return mol_table


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="inference_analysis_casp.yaml",
)
def main(cfg: DictConfig):
    """Analyze the inference results of a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    if cfg.v1_baseline:
        with open_dict(cfg):
            cfg.predictions_dir = cfg.predictions_dir.replace(
                f"top_{cfg.method}", f"top_{cfg.method}v1"
            )
    if cfg.no_ilcl:
        with open_dict(cfg):
            cfg.predictions_dir = cfg.predictions_dir.replace(
                "_ensemble_predictions", "_no_ilcl_ensemble_predictions"
            )
    if cfg.method == "vina":
        with open_dict(cfg):
            cfg.predictions_dir = cfg.predictions_dir.replace(
                "vina_", f"vina_{cfg.vina_binding_site_method}_"
            )
    elif cfg.method == "ensemble":
        with open_dict(cfg):
            cfg.predictions_dir = cfg.predictions_dir.replace(
                f"top_{cfg.method}_ensemble_predictions",
                f"top_{cfg.ensemble_ranking_method}_ensemble_predictions",
            )
    for config in ["", "_relaxed"]:
        if config == "_relaxed" and not cfg.score_relaxed_structures:
            continue
        assert os.path.exists(
            cfg.predictions_dir
        ), f"Directory `{cfg.predictions_dir}` does not exist."
        output_dir = cfg.predictions_dir + config
        scoring_results_filepath = Path(output_dir) / "scoring_results.csv"
        bust_results_filepath = Path(output_dir) / "bust_results.csv"

        # differentiate relaxed and unrelaxed protein pose results
        if "relaxed" in config and cfg.relax_protein:
            bust_results_filepath = bust_results_filepath.replace(".csv", "_protein_relaxed.csv")

        os.makedirs(scoring_results_filepath.parent, exist_ok=True)

        # collect analysis results
        if os.path.exists(scoring_results_filepath) and not cfg.force_casp15_rescore:
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} analysis results for inference directory `{output_dir}` already exist at `{scoring_results_filepath}`. Directly analyzing..."
            )
        else:
            temp_dir_path, available_targets = create_casp_input_dirs(cfg, config)

            # run CASP ligand scoring pipeline
            scoring_args = [
                cfg.python_exec_path,
                cfg.scoring_script_path,
                "-d",
                str(temp_dir_path / "targets"),
                "-p",
                str(temp_dir_path / "predictions"),
                "-o",
                output_dir,
                "-v",
                "DEBUG",
            ]
            targets_to_score = cfg.targets
            if cfg.allow_missing_predictions:
                # NOTE: Since e.g., DiffDock-L is notably unstable for the CASP15 multi-ligand
                # targets, we only score the targets for which such a method was able to generate
                # predictions after five retries of its respective inference script.
                targets_to_score = available_targets
                assert (
                    len(targets_to_score) > 0
                ), f"No available targets to score for {cfg.method}."
            if targets_to_score is not None:
                scoring_args.extend(["--targets", *[str(t) for t in targets_to_score]])
            if cfg.fault_tolerant:
                scoring_args.append("--fault-tolerant")
            result = subprocess.run(scoring_args)  # nosec

            if result.returncode == 0:
                logger.info(
                    f"{resolve_method_title(cfg.method)}{config} analysis results for inference directory `{output_dir}` successfully saved to `{scoring_results_filepath}`."
                )
            else:
                raise RuntimeError(
                    f"{resolve_method_title(cfg.method)}{config} analysis for inference directory `{output_dir}` failed with return code: {result.returncode}."
                )
        analysis_results = pd.read_csv(scoring_results_filepath)
        targets_to_select = analysis_results.target.unique().tolist()
        if len(targets_to_select) < NUM_SCOREABLE_CASP15_TARGETS:
            logger.warning(
                f"Number of targets analyzed is {len(targets_to_select)}, not the full {NUM_SCOREABLE_CASP15_TARGETS}."
            )

        # collect bust results
        if os.path.exists(bust_results_filepath) and not cfg.force_pb_rescore:
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` already exist at `{bust_results_filepath}`. Directly analyzing..."
            )
            bust_results = pd.read_csv(bust_results_filepath)
        else:
            mol_table = create_casp_mol_table(
                Path(cfg.predictions_dir),
                targets_to_select,
                relaxed="relaxed" in config,
                relax_protein=cfg.relax_protein,
            )
            assert len(mol_table) == len(
                targets_to_select
            ), f"Number of targets to bust is {len(mol_table)}, not the expected {len(targets_to_select)}."

            # NOTE: we use the `dock` mode here since with each method we implicitly perform cognate (e.g., apo or ab initio) docking,
            # yet for CASP data we do not have access to the ground-truth ligand structures in SDF format
            buster = PoseBusters(config="dock", top_n=None)
            bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)
            bust_results["target"] = targets_to_select

            bust_results.to_csv(bust_results_filepath, index=False)
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` successfully saved to `{bust_results_filepath}`."
            )

        # report results
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} ligands rmsd_≤_2å: {(analysis_results['rmsd'] <= 2.0).mean()}"
        )
        tests_table = bust_results[CASP_BUST_TEST_COLUMNS]
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} complexes pb_valid: {tests_table.all(axis=1).mean()}"
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
