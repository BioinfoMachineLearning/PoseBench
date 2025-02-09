# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from beartype import beartype
from beartype.typing import List, Tuple
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_input_csv(smiles_and_pdb_id_list: List[Tuple[str, str]], output_csv_path: str):
    """Write a FABind inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
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
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare a corresponding inference CSV file for the FABind
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
                f"fabind_{cfg.dataset}", f"fabind_pocket_only_{cfg.dataset}"
            )

    smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
        cfg.input_data_dir,
        pdb_ids=pdb_ids,
    )
    write_input_csv(smiles_and_pdb_id_list, cfg.output_csv_path)

    logger.info(f"FABind input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
