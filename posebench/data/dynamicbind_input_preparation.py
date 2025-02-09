# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import shutil

import hydra
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Optional, Tuple
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_input_csv(
    smiles_and_pdb_id_list: List[Tuple[str, str]],
    output_csv_dir: str,
    protein_filepath: Optional[str] = None,
    ligand_smiles: Optional[Any] = None,
):
    """Write a DynamicBind inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
    :param output_csv_dir: Path to the output CSV directory.
    :param protein_filepath: Optional path to the protein structure
        file.
    :param ligand_smiles: Optional SMILES string of the ligand.
    """
    if protein_filepath is not None and ligand_smiles is not None:
        output_csv_path = os.path.join(
            output_csv_dir,
            os.path.basename(protein_filepath).replace(".pdb", ".csv"),
        )
        for file in glob.glob(output_csv_dir + f"{os.sep}*"):
            try:
                os.remove(file)
            except OSError:
                continue
        with open(output_csv_path, "w") as f:
            f.write("ligand\n")
            f.write(f"{ligand_smiles}\n")
    else:
        for smiles, pdb_id in smiles_and_pdb_id_list:
            output_csv_path = os.path.join(output_csv_dir, f"{pdb_id}.csv")
            with open(output_csv_path, "w") as f:
                f.write("ligand\n")
                f.write(f"{smiles}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="dynamicbind_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare corresponding inference CSV files for the DynamicBind
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
            cfg.output_csv_dir = cfg.output_csv_dir.replace(
                f"dynamicbind_{cfg.dataset}", f"dynamicbind_pocket_only_{cfg.dataset}"
            )

    if cfg.input_protein_data_dir and not os.path.exists(cfg.input_protein_data_dir):
        os.makedirs(cfg.input_protein_data_dir, exist_ok=True)
    if not os.path.exists(cfg.output_csv_dir):
        os.makedirs(cfg.output_csv_dir, exist_ok=True)

    if cfg.protein_filepath is not None and cfg.ligand_smiles is not None:
        if cfg.input_protein_data_dir:
            for file in glob.glob(cfg.input_protein_data_dir + f"{os.sep}*"):
                try:
                    os.remove(file)
                except OSError:
                    continue
            shutil.copy(cfg.protein_filepath, cfg.input_protein_data_dir)
        write_input_csv(
            [],
            cfg.output_csv_dir,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
        )
    else:
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        write_input_csv(
            smiles_and_pdb_id_list,
            cfg.output_csv_dir,
        )

    logger.info(f"DynamicBind input CSV preparation for dataset `{cfg.dataset}` complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
