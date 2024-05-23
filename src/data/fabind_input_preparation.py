# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging

import hydra
import rootutils
from beartype import beartype
from beartype.typing import List, Tuple
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import register_custom_omegaconf_resolvers
from src.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_input_csv(smiles_and_pdb_id_list: List[Tuple[str, str]], output_csv_path: str):
    """Write a FABind inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a SMILES string and a PDB ID.
    :param output_csv_path: Path to the output CSV file.
    """
    with open(output_csv_path, "w") as f:
        f.write("Cleaned_SMILES,pdb_id\n")
        for smiles, pdb_id in smiles_and_pdb_id_list:
            f.write(f"{smiles},{pdb_id}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="fabind_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand complexes and prepare a
    corresponding inference CSV file for the FABind model.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    ccd_ids_filepath = (
        cfg.posebusters_ccd_ids_filepath if cfg.dataset == "posebusters_benchmark" else None
    )
    smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
        cfg.input_data_dir, ccd_ids_filepath=ccd_ids_filepath
    )
    write_input_csv(smiles_and_pdb_id_list, cfg.output_csv_path)

    logger.info(f"FABind input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
