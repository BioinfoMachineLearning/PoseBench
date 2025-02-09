# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os

import hydra
import numpy as np
import rootutils
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig, open_dict
from rdkit import Chem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.utils.data_utils import (
    combine_molecules,
    extract_protein_and_ligands_with_prody,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def insert_unique_ligand_atom_name(input_pdb_file: str, output_pdb_file: str):
    """Add unique identifier to ligand atoms with the same residue ID and chain
    ID.

    :param input_pdb_file: Path to the input PDB file.
    :param output_pdb_file: Path to the output PDB file.
    """
    pdb = PandasPdb().read_pdb(input_pdb_file)
    ligand_atoms = pdb.df["HETATM"]
    ligand_atoms["atom_name"] = ligand_atoms["atom_name"] + np.arange(
        1, len(ligand_atoms) + 1
    ).astype(str)
    pdb.df["HETATM"] = ligand_atoms
    pdb.to_pdb(output_pdb_file)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="rfaa_output_extraction.yaml",
)
def main(cfg: DictConfig):
    """Extract proteins and ligands separately from the prediction outputs."""
    assert cfg.dataset in [
        "posebusters_benchmark",
        "astex_diverse",
        "dockgen",
        "casp15",
    ], "Dataset must be one of 'posebusters_benchmark', 'astex_diverse', 'dockgen', 'casp15'."

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
        insert_unique_ligand_atom_name(
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
        for item in os.listdir(cfg.prediction_inputs_dir):
            input_item_path = os.path.join(cfg.prediction_inputs_dir, item)
            output_item_path = os.path.join(cfg.prediction_outputs_dir, item)
            if os.path.isdir(input_item_path) and os.path.isdir(output_item_path):
                for file in os.listdir(output_item_path):
                    if not file.endswith(".pdb"):
                        continue
                    input_filepath = os.path.join(
                        input_item_path, file.replace(".pdb", "_ligands.sdf")
                    )
                    intermediate_output_filepath = os.path.join(output_item_path, file)
                    final_output_filepath = os.path.join(cfg.inference_outputs_dir, item, file)
                    os.makedirs(os.path.dirname(final_output_filepath), exist_ok=True)
                    if cfg.dataset in ["posebusters_benchmark", "astex_diverse", "dockgen"]:
                        if cfg.dataset == "dockgen":
                            input_filepaths = glob.glob(
                                input_filepath.replace("_ligands.sdf", "*.sdf")
                            )
                            assert len(input_filepaths) > 0, "No DockGen ligand SDF files found."
                            ligand_mols = [
                                Chem.MolFromMolFile(input_file, sanitize=False)
                                for input_file in input_filepaths
                            ]
                            ligand_mol = combine_molecules(ligand_mols)
                            ligand_smiles = Chem.MolToSmiles(ligand_mol)
                        else:
                            assert os.path.exists(
                                input_filepath
                            ), f"Ligand SDF file not found: {input_filepath}"
                            ligand_smiles = Chem.MolToSmiles(Chem.MolFromMolFile(input_filepath))
                    else:
                        # NOTE: for the `casp15` dataset, standalone ligand SMILES are not available
                        ligand_smiles = None
                    insert_unique_ligand_atom_name(
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
                        permute_ligand_smiles=True,
                    )
                    os.remove(final_output_filepath.replace(".pdb", "_fixed.pdb"))

    logger.info(
        f"Finished extracting {cfg.dataset} protein and ligands from all prediction outputs."
    )


if __name__ == "__main__":
    main()
