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
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import (
    create_sdf_file_from_smiles,
    extract_sequences_from_protein_structure_file,
    parse_inference_inputs_from_dir,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@beartype
def write_scripts(
    smiles_and_pdb_id_list: List[Tuple[Any, str]],
    input_data_dir: str,
    output_scripts_path: str,
    dataset: str,
    pocket_only_baseline: bool = False,
    protein_filepath: Optional[str] = None,
    ligand_smiles: Optional[Any] = None,
    input_id: Optional[str] = None,
):
    """Write a RoseTTAFold-All-Atom inference CSV file.

    :param smiles_and_pdb_id_list: A list of tuples each containing a
        SMILES string and a PDB ID.
    :param input_data_dir: Path to directory of input protein-ligand
        complex subdirectories.
    :param output_scripts_path: Path to directory of output FASTA
        sequence and ligand SDF files.
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
        for chain_index, sequence in enumerate(protein_sequence_list, start=1):
            with open(os.path.join(output_dir, f"{input_id}_chain_{chain_index}.fasta"), "w") as f:
                f.write(f">{input_id}|Chain {chain_index}\n{sequence}\n")
        ligand_filepaths = [
            path
            for path in glob.glob(
                os.path.join(os.path.dirname(ligand_smiles), f"{input_id}_*.sdf")
            )
            if not any(
                x in path for x in ["protein", "lig.sdf", "ligand.sdf", "multicom", "start_conf"]
            )
        ]
        for ligand_filepath in ligand_filepaths:
            shutil.copy(ligand_filepath, output_dir)
    else:
        for smiles_string, pdb_id in smiles_and_pdb_id_list:
            output_dir = os.path.join(output_scripts_path, pdb_id)
            os.makedirs(output_dir, exist_ok=True)
            casp_dataset_requested = os.path.basename(input_data_dir) == "targets"
            if casp_dataset_requested:
                protein_filepath = os.path.join(input_data_dir, f"{pdb_id}_lig.pdb")
                ligand_filepaths = [
                    create_sdf_file_from_smiles(
                        smiles, os.path.join(output_dir, f"{pdb_id}_{i}.sdf")
                    )
                    for i, smiles in enumerate(smiles_string.split("."), start=1)
                ]
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
                if dataset == "dockgen":
                    ligand_filepaths = [
                        create_sdf_file_from_smiles(
                            smiles, os.path.join(output_dir, f"{pdb_id}_{i}.sdf")
                        )
                        for i, smiles in enumerate(smiles_string.split("."), start=1)
                    ]
                else:
                    ligand_filepaths = [
                        path
                        for path in glob.glob(
                            os.path.join(input_data_dir, pdb_id, f"{pdb_id}_*.sdf")
                        )
                        if not any(
                            x in path
                            for x in ["protein", "lig.sdf", "ligand.sdf", "multicom", "start_conf"]
                        )
                    ]
            protein_sequence_list = [
                seq
                for seq in extract_sequences_from_protein_structure_file(protein_filepath)
                if len(seq) > 0
            ]
            for chain_index, sequence in enumerate(protein_sequence_list, start=1):
                with open(
                    os.path.join(output_dir, f"{pdb_id}_chain_{chain_index}.fasta"), "w"
                ) as f:
                    f.write(f">{pdb_id}|Chain {chain_index}\n{sequence}\n")
            if not casp_dataset_requested and dataset != "dockgen":
                for ligand_filepath in ligand_filepaths:
                    shutil.copy(ligand_filepath, output_dir)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/data",
    config_name="rfaa_input_preparation.yaml",
)
def main(cfg: DictConfig):
    """Parse a data directory containing subdirectories of protein-ligand
    complexes and prepare corresponding inference CSV file for the RoseTTAFold-
    All-Atom model.

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
            [],
            cfg.input_data_dir,
            cfg.output_scripts_path,
            dataset=cfg.dataset,
            pocket_only_baseline=cfg.pocket_only_baseline,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )
    else:
        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.input_data_dir,
            pdb_ids=pdb_ids,
        )
        write_scripts(
            smiles_and_pdb_id_list,
            cfg.input_data_dir,
            cfg.output_scripts_path,
            dataset=cfg.dataset,
            pocket_only_baseline=cfg.pocket_only_baseline,
            protein_filepath=cfg.protein_filepath,
            ligand_smiles=cfg.ligand_smiles,
            input_id=cfg.input_id,
        )

    logger.info(
        f"RoseTTAFold-All-Atom input files for dataset `{cfg.dataset}` are fully prepared for inference."
    )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
