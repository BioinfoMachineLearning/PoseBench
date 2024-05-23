# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Optional, Tuple
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers
from src.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_input_csv(
    smiles_and_pdb_id_list: List[Tuple[Any, str]],
    output_csv_path: str,
    input_protein_structure_dir: str,
    protein_filepath: Optional[str] = None,
    ligand_smiles: Optional[Any] = None,
    input_id: Optional[str] = None,
):
    """Write a DiffDock inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a SMILES string and a PDB ID.
    :param output_csv_path: Path to the output CSV file.
    :param input_protein_structure_dir: Path to the directory containing the protein structure
        input files.
    :param protein_filepath: Optional path to the protein structure file.
    :param ligand_smiles: Optional SMILES string of the ligand.
    :param input_id: Optional input ID.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w") as f:
        f.write("complex_name,protein_path,ligand_description,protein_sequence\n")
        if protein_filepath is not None and ligand_smiles is not None:
            input_id = (
                "_".join(os.path.splitext(os.path.basename(protein_filepath))[0].split("_")[:2])
                if input_id is None
                else input_id
            )
            smiles = ligand_smiles[0] if isinstance(ligand_smiles, tuple) else ligand_smiles
            protein_sequence = ligand_smiles[1] if isinstance(ligand_smiles, tuple) else ""
            f.write(f"{input_id},{protein_filepath},{smiles},{protein_sequence}\n")
        else:
            for smiles, pdb_id in smiles_and_pdb_id_list:
                if os.path.isdir(os.path.join(input_protein_structure_dir, pdb_id)):
                    protein_filepath = os.path.join(
                        input_protein_structure_dir, pdb_id, f"{pdb_id}_ligand.pdb"
                    )
                else:
                    protein_filepath = (
                        os.path.join(
                            input_protein_structure_dir,
                            f"{pdb_id}_holo_aligned_esmfold_protein.pdb",
                        )
                        if os.path.exists(
                            os.path.join(
                                input_protein_structure_dir,
                                f"{pdb_id}_holo_aligned_esmfold_protein.pdb",
                            )
                        )
                        else os.path.join(input_protein_structure_dir, f"{pdb_id}.pdb")
                    )
                if not os.path.exists(protein_filepath):
                    logger.warning(
                        f"Skipping protein structure file which was not found: {protein_filepath}"
                    )
                    continue
                protein_sequence = smiles[1] if isinstance(smiles, tuple) else ""
                f.write(f"{pdb_id},{protein_filepath},{smiles},{protein_sequence}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="diffdock_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand complexes and prepare
    corresponding inference CSV file for the DiffDock model.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    if cfg.protein_filepath is not None and cfg.ligand_smiles is not None:
        write_input_csv(
            [],
            cfg.output_csv_path,
            cfg.input_protein_structure_dir,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )
    else:
        ccd_ids_filepath = (
            cfg.posebusters_ccd_ids_filepath if cfg.dataset == "posebusters_benchmark" else None
        )
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir, ccd_ids_filepath=ccd_ids_filepath
        )
        write_input_csv(
            smiles_and_pdb_id_list,
            cfg.output_csv_path,
            cfg.input_protein_structure_dir,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )

    logger.info(f"DiffDock input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
