# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os

import hydra
import rootutils
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig, open_dict
from rdkit import Chem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.utils.data_utils import (
    extract_protein_and_ligands_with_prody,
    parse_inference_inputs_from_dir,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def distinguish_ligand_atoms(input_pdb_file: str, output_pdb_file: str):
    """Mark ligand atoms as heteroatoms.

    :param input_pdb_file: Path to the input PDB file.
    :param output_pdb_file: Path to the output PDB file.
    """
    pdb = PandasPdb().read_pdb(input_pdb_file)
    ligand_atoms = pdb.df["ATOM"][pdb.df["ATOM"]["residue_name"] == "LIG"]

    ligand_indices = ligand_atoms.index
    pdb.df["ATOM"] = pdb.df["ATOM"].drop(ligand_indices)

    ligand_atoms.record_name = "HETATM"
    pdb.df["HETATM"] = ligand_atoms

    pdb.to_pdb(output_pdb_file)


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
                f"rfaa_{cfg.dataset}", f"rfaa_pocket_only_{cfg.dataset}"
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
        distinguish_ligand_atoms(
            intermediate_output_filepath,
            final_output_filepath.replace(".pdb", "_fixed.pdb"),
        )
        extract_protein_and_ligands_with_prody(
            final_output_filepath.replace(".pdb", "_fixed.pdb"),
            final_output_filepath.replace(".pdb", "_protein.pdb"),
            final_output_filepath.replace(".pdb", "_ligand.sdf"),
            sanitize=False,
            add_element_types=True,
            ligand_smiles=cfg.ligand_smiles,
        )
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
            if os.path.isdir(input_item_path) and os.path.isdir(output_item_path):
                for file in os.listdir(output_item_path):
                    if not file.endswith(".pdb"):
                        continue
                    intermediate_output_filepath = os.path.join(output_item_path, file)
                    final_output_filepath = os.path.join(cfg.inference_outputs_dir, item, file)
                    os.makedirs(os.path.dirname(final_output_filepath), exist_ok=True)
                    if cfg.dataset in ["posebusters_benchmark", "astex_diverse", "dockgen"]:
                        ligand_smiles = pdb_id_to_smiles[item].replace("|", ".")
                    else:
                        # NOTE: for the `casp15` dataset, standalone ligand SMILES are not available
                        ligand_smiles = None
                    distinguish_ligand_atoms(
                        intermediate_output_filepath,
                        final_output_filepath.replace(".pdb", "_fixed.pdb"),
                    )
                    extract_protein_and_ligands_with_prody(
                        final_output_filepath.replace(".pdb", "_fixed.pdb"),
                        final_output_filepath.replace(".pdb", "_protein.pdb"),
                        final_output_filepath.replace(".pdb", "_ligand.sdf"),
                        sanitize=False,
                        add_element_types=True,
                        ligand_smiles=ligand_smiles,
                    )
                    os.remove(final_output_filepath.replace(".pdb", "_fixed.pdb"))

    logger.info(
        f"Finished extracting {cfg.dataset} protein and ligands from all prediction outputs."
    )


if __name__ == "__main__":
    main()
