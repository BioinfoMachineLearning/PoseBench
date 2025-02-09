# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Optional, Tuple
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.models.ensemble_generation import (
    LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE,
)
from posebench.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_input_csv(
    smiles_and_pdb_id_list: Optional[List[Tuple[Any, str]]],
    output_csv_path: str,
    input_receptor_structure_dir: Optional[str],
    input_receptor: Optional[str] = None,
    input_ligand: Optional[Any] = None,
    input_template: Optional[str] = None,
    input_id: Optional[str] = None,
):
    """Write a FlowDock inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
    :param output_csv_path: Path to the output CSV file.
    :param input_receptor_structure_dir: Path to the directory
        containing the protein structure input files.
    :param input_receptor: Optional path to a single input protein
        sequence.
    :param input_ligand: Optional single input ligand SMILES string.
    :param input_template: Path to the optional template protein
        structure.
    :param input_id: Optional input ID.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w") as f:
        f.write("id,input_receptor,input_ligand,input_template\n")
        if input_ligand is not None:
            input_id = (
                (
                    "_".join(os.path.splitext(os.path.basename(input_template))[0].split("_")[:2])
                    if input_template
                    else "ensemble_input_ligand"
                )
                if input_id is None
                else input_id
            )
            # NOTE: a placeholder protein sequence is used when making ligand-only predictions
            if not input_receptor:
                input_receptor = LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
            if not input_template:
                input_template = LIGAND_ONLY_RECEPTOR_PLACEHOLDER_SEQUENCE
            f.write(f"{input_id},{input_receptor},{input_ligand},{input_template}\n")
        else:
            for smiles, pdb_id in smiles_and_pdb_id_list:
                if os.path.isdir(os.path.join(input_receptor_structure_dir, pdb_id)):
                    input_receptor = os.path.join(
                        input_receptor_structure_dir, pdb_id, f"{pdb_id}_ligand.pdb"
                    )
                else:
                    input_receptor = (
                        os.path.join(
                            input_receptor_structure_dir,
                            f"{pdb_id}_holo_aligned_predicted_protein.pdb",
                        )
                        if os.path.exists(
                            os.path.join(
                                input_receptor_structure_dir,
                                f"{pdb_id}_holo_aligned_predicted_protein.pdb",
                            )
                        )
                        else os.path.join(input_receptor_structure_dir, f"{pdb_id}.pdb")
                    )
                if not os.path.exists(input_receptor):
                    logger.warning(f"Skipping input protein which was not found: {input_receptor}")
                    continue
                f.write(f"{pdb_id},{input_receptor},{smiles.replace('.', '|')},{input_receptor}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="flowdock_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare corresponding inference CSV file for the FlowDock
    model.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    # load ID subset if requested
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
            cfg.output_csv_path = cfg.output_csv_path.replace(
                f"flowdock_{cfg.dataset}", f"flowdock_pocket_only_{cfg.dataset}"
            )

    input_receptor_structure_dir = (
        cfg.input_receptor_structure_dir + "_bs_cropped"
        if cfg.pocket_only_baseline
        else cfg.input_receptor_structure_dir
    )
    if cfg.input_receptor is not None and cfg.input_ligand is not None:
        write_input_csv(
            [],
            output_csv_path=cfg.output_csv_path,
            input_receptor_structure_dir=input_receptor_structure_dir,
            input_receptor=cfg.input_receptor,
            input_ligand=cfg.input_ligand,
            input_template=cfg.input_template,
            input_id=cfg.input_id,
        )
    else:
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        write_input_csv(
            smiles_and_pdb_id_list=smiles_and_pdb_id_list,
            output_csv_path=cfg.output_csv_path,
            input_receptor_structure_dir=input_receptor_structure_dir,
            input_receptor=cfg.input_receptor,
            input_ligand=cfg.input_ligand,
            input_template=cfg.input_template,
            input_id=cfg.input_id,
        )

    logger.info(f"FlowDock input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
