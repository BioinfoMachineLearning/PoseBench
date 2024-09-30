# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
from collections import defaultdict

import hydra
import rootutils
from omegaconf import DictConfig
from rdkit import Chem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.utils.data_utils import combine_molecules

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="tulip_output_extraction.yaml",
)
def main(cfg: DictConfig):
    """Extract proteins and ligands separately from the prediction outputs."""
    os.makedirs(cfg.inference_outputs_dir, exist_ok=True)

    pdb_ids = None
    if cfg.dataset == "posebusters_benchmark" and cfg.posebusters_ccd_ids_filepath is not None:
        assert os.path.exists(
            cfg.posebusters_ccd_ids_filepath
        ), f"Invalid CCD IDs file path for PoseBusters Benchmark: {os.path.exists(cfg.posebusters_ccd_ids_filepath)}."
        with open(cfg.posebusters_ccd_ids_filepath) as f:
            pdb_ids = set(f.read().splitlines())
    elif cfg.dataset == "dockgen" and cfg.dockgen_test_ids_filepath is not None:
        assert os.path.exists(
            cfg.dockgen_test_ids_filepath
        ), f"Invalid test IDs file path for DockGen: {os.path.exists(cfg.dockgen_test_ids_filepath)}."
        with open(cfg.dockgen_test_ids_filepath) as f:
            pdb_ids = {line.replace(" ", "-") for line in f.read().splitlines()}
    elif cfg.dataset not in ["posebusters_benchmark", "astex_diverse", "dockgen", "casp15"]:
        raise ValueError(f"Dataset `{cfg.dataset}` not supported.")

    for target_name in os.listdir(cfg.prediction_outputs_dir):
        if pdb_ids is not None and target_name not in pdb_ids:
            logger.info(
                f"Skipping target {target_name} as it is not in the specified PoseBusters Benchmark or DockGen PDB IDs set."
            )
            continue
        target_dir_path = os.path.join(cfg.prediction_outputs_dir, target_name)
        if os.path.isdir(target_dir_path):
            ligand_ids = sorted(os.listdir(target_dir_path), key=lambda x: int(x))
            min_num_ranks = min(
                len(glob.glob(os.path.join(target_dir_path, ligand_id, "rank_*.sdf")))
                for ligand_id in ligand_ids
            )
            effective_top_n_to_select = min(min_num_ranks, cfg.method_top_n_to_select)
            assert (
                effective_top_n_to_select > 0
            ), f"Effective top-N ligands to select for {target_name} must be non-zero."
            top_n_combined_ligands_per_rank = defaultdict(list)
            for ligand_id in ligand_ids:
                ligand_dir_path = os.path.join(target_dir_path, ligand_id)
                top_n_ligand_sdf_filenames = sorted(
                    os.listdir(ligand_dir_path),
                    key=lambda x: int(os.path.splitext(x)[0].split("_")[1]),
                )[:effective_top_n_to_select]
                for rank_index in range(effective_top_n_to_select):
                    rank_ligand_filepath = os.path.join(
                        ligand_dir_path, top_n_ligand_sdf_filenames[rank_index]
                    )
                    rank_ligand = Chem.MolFromMolFile(rank_ligand_filepath)
                    combined_rank_ligands = top_n_combined_ligands_per_rank[rank_index]
                    if combined_rank_ligands:
                        combined_rank_ligand = combined_rank_ligands[0]
                        new_combined_rank_ligand = combine_molecules(
                            [combined_rank_ligand, rank_ligand]
                        )
                        top_n_combined_ligands_per_rank[rank_index] = [new_combined_rank_ligand]
                    else:
                        top_n_combined_ligands_per_rank[rank_index].append(rank_ligand)
            for rank_index in range(effective_top_n_to_select):
                combined_rank_ligand = top_n_combined_ligands_per_rank[rank_index][0]
                combined_rank_ligand_filepath = os.path.join(
                    cfg.inference_outputs_dir, target_name, f"rank{rank_index + 1}.sdf"
                )
                os.makedirs(os.path.dirname(combined_rank_ligand_filepath), exist_ok=True)
                Chem.MolToMolFile(combined_rank_ligand, combined_rank_ligand_filepath)
            logger.info(f"Finished creating combined ligand SDF file for target {target_name}.")
    logger.info(f"Finished extracting {cfg.dataset} combined ligands for all predictions.")


if __name__ == "__main__":
    main()
