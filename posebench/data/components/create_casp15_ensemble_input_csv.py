# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import argparse
import logging
import os

import pandas as pd
import rootutils

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.utils.data_utils import extract_sequences_from_protein_structure_file


def create_casp15_ensemble_input_csv(args: argparse.Namespace):
    """Create a CSV file with the protein sequences and ligand SMILES strings
    for the CASP15 dataset.

    :param args: The command line arguments.
    """
    rows = []
    for item in os.listdir(args.predicted_structures_dir):
        if not item.endswith(".pdb"):
            continue

        item_id = os.path.splitext(item)[0]
        predicted_protein_filepath = os.path.join(args.predicted_structures_dir, f"{item_id}.pdb")
        ligand_smiles_filepath = os.path.join(args.targets_dir, f"{item_id}.smiles.txt")
        assert os.path.exists(
            predicted_protein_filepath
        ), f"Predicted protein file not found: {predicted_protein_filepath}"
        assert os.path.exists(
            ligand_smiles_filepath
        ), f"Ligand SMILES file not found: {ligand_smiles_filepath}"

        # only parse protein chains (e.g., not nucleic acids)
        protein_seqs = extract_sequences_from_protein_structure_file(predicted_protein_filepath)
        protein_seq = ":".join([s for s in protein_seqs if len(s) > 0])

        ligand_smiles_df = pd.read_csv(ligand_smiles_filepath, delim_whitespace=True)
        mol_smiles = ":".join(ligand_smiles_df["SMILES"].tolist())
        mol_numbers = ligand_smiles_df["ID"].tolist()
        mol_names = ligand_smiles_df["Name"].tolist()
        mol_tasks = "P"  # NOTE: for CASP15, we only request predicted complex structures, binding affinities (i.e., `A` or `PA`)

        rows.append((protein_seq, mol_smiles, item_id, mol_numbers, mol_names, mol_tasks))

    df = pd.DataFrame(
        rows,
        columns=[
            "protein_input",
            "ligand_smiles",
            "name",
            "ligand_numbers",
            "ligand_names",
            "ligand_tasks",
        ],
    )
    df.to_csv(args.output_csv_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a CSV file with the protein sequences and ligand SMILES strings for the CASP15 dataset."
    )
    parser.add_argument(
        "-t",
        "--targets-dir",
        "--targets_dir",
        type=str,
        default="data/casp15_set/targets",
        help="The directory containing the CASP15 targets.",
    )
    parser.add_argument(
        "-p" "--predicted-structures-dir",
        "--predicted_structures_dir",
        type=str,
        default="data/casp15_set/casp15_holo_aligned_predicted_structures",
        help="The directory containing the CASP15 (ground truth binding site-aligned) predicted protein structures.",
    )
    parser.add_argument(
        "-o",
        "--output-csv-filepath",
        "--output_csv_filepath",
        type=str,
        default="data/test_cases/casp15/ensemble_prediction_inputs.csv",
        help="The output CSV file.",
    )
    args = parser.parse_args()

    create_casp15_ensemble_input_csv(args)
