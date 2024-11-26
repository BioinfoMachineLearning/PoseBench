# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.utils.data_utils import (
    extract_protein_and_ligands_with_prody,
    parse_inference_inputs_from_dir,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="chai_output_extraction.yaml",
)
def main(cfg: DictConfig):
    """Extract proteins and ligands separately from the prediction outputs."""
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

    if cfg.pocket_only_baseline:
        with open_dict(cfg):
            cfg.prediction_inputs_dir = cfg.prediction_inputs_dir.replace(
                cfg.dataset, f"{cfg.dataset}_pocket_only"
            )
            cfg.prediction_outputs_dir = cfg.prediction_outputs_dir.replace(
                cfg.dataset, f"{cfg.dataset}_pocket_only"
            )
            cfg.inference_outputs_dir = cfg.inference_outputs_dir.replace(
                f"chai-lab_{cfg.dataset}", f"chai-lab_pocket_only_{cfg.dataset}"
            )

    if cfg.complex_filepath is not None:
        # process single-complex inputs
        assert os.path.exists(
            cfg.complex_filepath
        ), f"Complex PDB file not found: {cfg.complex_filepath}"
        assert (
            cfg.complex_id is not None
        ), "Complex ID must be provided when extracting single complex outputs."
        assert (
            cfg.ligand_smiles is not None
        ), "Ligand SMILES must be provided when extracting single complex outputs."
        assert (
            cfg.output_dir is not None
        ), "Output directory must be provided when extracting single complex outputs."
        intermediate_output_filepath = cfg.complex_filepath
        final_output_filepath = os.path.join(
            cfg.output_dir, cfg.complex_id, os.path.basename(cfg.complex_filepath)
        )
        os.makedirs(os.path.dirname(final_output_filepath), exist_ok=True)
        try:
            extract_protein_and_ligands_with_prody(
                intermediate_output_filepath,
                final_output_filepath.replace(".cif", "_protein.pdb"),
                final_output_filepath.replace(".cif", "_ligand.sdf"),
                sanitize=False,
                add_element_types=True,
                ligand_smiles=cfg.ligand_smiles,
            )
        except Exception as e:
            logger.error(f"Failed to extract protein and ligands for {cfg.complex_id} due to: {e}")
    else:
        # process all complexes in a dataset
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        pdb_id_to_smiles = {pdb_id: smiles for smiles, pdb_id in smiles_and_pdb_id_list}
        for item in os.listdir(cfg.prediction_inputs_dir):
            input_item_path = os.path.join(cfg.prediction_inputs_dir, item)
            output_item_path = os.path.join(cfg.prediction_outputs_dir, item)

            all_scores = dict()
            if os.path.isdir(input_item_path) and os.path.isdir(output_item_path):
                for file in os.listdir(output_item_path):
                    if not file.endswith(".cif"):
                        continue
                    intermediate_output_filepath = os.path.join(output_item_path, file)

                    scores_filepath = os.path.join(
                        output_item_path, file.replace("pred.", "scores.").replace(".cif", ".npz")
                    )
                    scores = dict(np.load(scores_filepath))
                    all_scores[intermediate_output_filepath] = scores["aggregate_score"]

            # rank by aggregate score
            all_ranks = {
                k: rank
                for rank, (k, _) in enumerate(
                    sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                )
            }

            if os.path.isdir(input_item_path) and os.path.isdir(output_item_path):
                for file in os.listdir(output_item_path):
                    if not file.endswith(".cif"):
                        continue
                    intermediate_output_filepath = os.path.join(output_item_path, file)
                    final_output_filepath = os.path.join(cfg.inference_outputs_dir, item, file)
                    os.makedirs(os.path.dirname(final_output_filepath), exist_ok=True)

                    if cfg.dataset in ["posebusters_benchmark", "astex_diverse", "dockgen"]:
                        ligand_smiles = pdb_id_to_smiles[item]
                    else:
                        # NOTE: for the `casp15` dataset, standalone ligand SMILES are not available
                        ligand_smiles = None

                    rank = all_ranks[intermediate_output_filepath]
                    model_idx = os.path.splitext(file)[0].split("_")[-1]

                    try:
                        extract_protein_and_ligands_with_prody(
                            # NOTE: to simplify this implementation, we repurpose model
                            # indices as sample ranks past this point in the codebase
                            intermediate_output_filepath,
                            final_output_filepath.replace(
                                f"model_idx_{model_idx}", f"model_idx_{rank}"
                            ).replace(".cif", "_protein.pdb"),
                            final_output_filepath.replace(
                                f"model_idx_{model_idx}", f"model_idx_{rank}"
                            ).replace(".cif", "_ligand.sdf"),
                            sanitize=False,
                            add_element_types=True,
                            ligand_smiles=ligand_smiles,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to extract protein and ligands for {item} due to: {e}"
                        )

    logger.info(
        f"Finished extracting {cfg.dataset} protein and ligands from all prediction outputs."
    )


if __name__ == "__main__":
    main()
