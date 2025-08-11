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

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import (
    extract_sequences_from_protein_structure_file,
    parse_inference_inputs_from_dir,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_scripts(
    smiles_and_pdb_id_list: List[Tuple[Any, str]],
    input_data_dir: str,
    msa_dir: str | None,
    output_scripts_path: str,
    dataset: str,
    pocket_only_baseline: bool = False,
    protein_filepath: Optional[str] = None,
    ligand_smiles: Optional[Any] = None,
    input_id: Optional[str] = None,
):
    """Write a Boltz inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
    :param input_data_dir: Path to directory of input protein-ligand
        complex subdirectories.
    :param msa_dir: Path to directory of MSA files.
    :param output_scripts_path: Path to directory of output FASTA
        sequence files.
    :param dataset: Dataset name.
    :param pocket_only_baseline: Whether to provide only the protein
        pocket as a baseline experiment.
    :param protein_filepath: Optional path to the protein structure
        file.
    :param ligand_smiles: Optional SMILES string of the ligand.
    :param input_id: Optional input ID.
    """
    if pocket_only_baseline:
        output_scripts_path = output_scripts_path.replace(dataset, f"{dataset}_pocket_only")

    os.makedirs(output_scripts_path, exist_ok=True)
    if protein_filepath is not None and ligand_smiles is not None:
        input_id = (
            "_".join(os.path.splitext(os.path.basename(protein_filepath))[0].split("_")[:2])
            if input_id is None
            else input_id
        )
        # only parse protein chains (e.g., not nucleic acids)
        protein_sequence_list = [
            seq
            for seq in extract_sequences_from_protein_structure_file(protein_filepath)
            if len(seq) > 0
        ]
        output_dir = os.path.join(output_scripts_path, input_id)
        fasta_filepath = os.path.join(output_dir, f"{input_id}.fasta")
        if os.path.exists(fasta_filepath):
            logger.warning(
                f"FASTA file already exists for input ID {input_id}. Skipping writing to file..."
            )
            return
        for chain_index, sequence in enumerate(protein_sequence_list, start=1):
            chain_id = chr(ord("A") + chain_index - 1)
            msa_path = (
                os.path.join(msa_dir, f"{input_id}_chain_{chain_index - 1}.csv")
                if msa_dir is not None
                else None
            )
            msa_suffix = msa_path if msa_path is not None else "empty"
            with open(fasta_filepath, "a") as f:
                f.write(f">{chain_id}|protein|{msa_suffix}\n{sequence}\n")
        # NOTE: in the inference setting, `:` is used to separate ligand SMILES strings
        for chain_index, smiles in enumerate(ligand_smiles.split(":"), start=1):
            chain_id = chr(ord("A") + chain_index - 1)
            with open(fasta_filepath, "a") as f:
                f.write(f">{chain_id}|smiles\n{smiles}\n")
    else:
        for smiles_string, pdb_id in smiles_and_pdb_id_list:
            output_dir = os.path.join(output_scripts_path, pdb_id)
            os.makedirs(output_dir, exist_ok=True)
            casp_dataset_requested = os.path.basename(input_data_dir) == "targets"
            if casp_dataset_requested:
                protein_filepath = os.path.join(input_data_dir, f"{pdb_id}_lig.pdb")
            else:
                if pocket_only_baseline:
                    protein_filepath = os.path.join(
                        input_data_dir,
                        f"{dataset}_holo_aligned_predicted_structures_bs_cropped",
                        f"{pdb_id}_holo_aligned_predicted_protein.pdb",
                    )
                    if not os.path.exists(protein_filepath):
                        logger.warning(
                            f"Protein structure file not found for PDB ID {pdb_id}. Skipping..."
                        )
                        continue
                else:
                    protein_id = pdb_id.split("_")[0] if dataset == "dockgen" else pdb_id
                    protein_file_suffix = "_processed" if dataset == "dockgen" else "_protein"
                    protein_filepath = os.path.join(
                        input_data_dir, pdb_id, f"{protein_id}{protein_file_suffix}.pdb"
                    )
            protein_sequence_list = [
                seq
                for seq in extract_sequences_from_protein_structure_file(protein_filepath)
                if len(seq) > 0
            ]
            ligand_smiles_list = smiles_string.split(".")
            fasta_filepath = os.path.join(output_dir, f"{pdb_id}.fasta")
            if os.path.exists(fasta_filepath):
                logger.warning(
                    f"FASTA file already exists for PDB ID {pdb_id}. Skipping writing to file..."
                )
                continue
            same_seq_chain_mapping = {}
            for chain_index, sequence in enumerate(protein_sequence_list, start=1):
                if sequence not in same_seq_chain_mapping:
                    same_seq_chain_mapping[sequence] = chain_index - 1
                chain_id = chr(ord("A") + chain_index - 1)
                msa_path = (
                    # NOTE: for Boltz, identical protein sequences are mapped to the same (first) MSA chain ID of the same sequence
                    os.path.join(msa_dir, f"{pdb_id}_chain_{same_seq_chain_mapping[sequence]}.csv")
                    if msa_dir is not None
                    else None
                )
                msa_suffix = msa_path if msa_path is not None else "empty"
                with open(fasta_filepath, "a") as f:
                    f.write(f">{chain_id}|protein|{msa_suffix}\n{sequence}\n")
            for chain_index, sequence in enumerate(
                ligand_smiles_list, start=len(protein_sequence_list) + 1
            ):
                chain_id = chr(ord("A") + chain_index - 1)
                with open(fasta_filepath, "a") as f:
                    f.write(f">{chain_id}|smiles\n{sequence}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="boltz_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare corresponding inference CSV file for the Boltz model.

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

    if cfg.protein_filepath is not None and cfg.ligand_smiles is None:
        write_scripts(
            smiles_and_pdb_id_list=[],
            input_data_dir=cfg.input_data_dir,
            msa_dir=cfg.msa_dir,
            output_scripts_path=cfg.output_scripts_path,
            dataset=cfg.dataset,
            pocket_only_baseline=cfg.pocket_only_baseline,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )
    else:
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            input_data_dir=cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        write_scripts(
            smiles_and_pdb_id_list=smiles_and_pdb_id_list,
            input_data_dir=cfg.input_data_dir,
            msa_dir=cfg.msa_dir,
            output_scripts_path=cfg.output_scripts_path,
            dataset=cfg.dataset,
            pocket_only_baseline=cfg.pocket_only_baseline,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )

    logger.info(f"Boltz input files for dataset `{cfg.dataset}` are fully prepared for inference.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
