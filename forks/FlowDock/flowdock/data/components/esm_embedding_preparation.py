# Adapted from: https://github.com/gcorso/DiffDock

import os
from argparse import ArgumentParser

import pandas as pd
import rootutils
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    convert_protein_pts_to_pdb,
    pdb_filepath_to_protein,
)

log = RankedLogger(__name__, rank_zero_only=True)

parser = ArgumentParser()
parser.add_argument("--out_file", type=str, default="data/pdbbind/pdbbind_sequences.fasta")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["pdbbind", "moad", "dockgen", "pdbsidechain"],
    default="pdbbind",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/pdbbind/PDBBind_processed/",
)
parser.add_argument(
    "--pdbbind_split_filepath",
    type=str,
    default=None,
    help="Path to a CSV file containing the PDB codes to include in the PDBBind dataset.",
)
parser.add_argument(
    "--pdbsidechain_metadata_csv_path",
    type=str,
    default="data/pdbsidechain/pdb_2021aug02/list.csv",
)
parser.add_argument(
    "--skip_extracting_existing_pdbsidechain_pdb_files",
    action="store_true",
    help="If True, will avoid overwriting the PDB files already generated from the corresponding vdM PyTorch files.",
)
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f"Provided data directory {args.data_dir} does not exist.")

biopython_parser = PDBParser()

data_dir = args.data_dir

if args.dataset == "pdbbind":
    sequences = []
    ids = []

    if args.pdbbind_split_filepath is not None and os.path.exists(args.pdbbind_split_filepath):
        split_pdb_codes = set(pd.read_csv(args.pdbbind_split_filepath, header=None).iloc[:, 0])
        names = [item for item in os.listdir(data_dir) if item in split_pdb_codes]
    else:
        names = os.listdir(data_dir)
    for name in tqdm(names):
        if name == ".DS_Store":
            continue
        if os.path.exists(os.path.join(data_dir, name, f"{name}_protein_processed.pdb")):
            rec_pdb_path = os.path.join(data_dir, name, f"{name}_protein_processed.pdb")
        else:
            rec_pdb_path = os.path.join(data_dir, name, f"{name}_protein.pdb")
        try:
            protein = pdb_filepath_to_protein(rec_pdb_path)
        except Exception as e:
            log.warning(f"Error in parsing {rec_pdb_path} due to: {e}")
            continue
        for i, seq in enumerate(protein.letter_sequences):
            sequences.append(seq[1])
            ids.append(f"{name}_chain_{i}")
    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

elif args.dataset == "moad":
    names = os.listdir(data_dir)
    names = [n[:6] for n in names]
    sequences = []
    ids = []

    for name in tqdm(names):
        if name == ".DS_Store":
            continue
        if not os.path.exists(os.path.join(data_dir, f"{name}_protein.pdb")):
            log.warning(f"We are skipping {name} because there was no {name}_protein.pdb")
            continue
        rec_pdb_path = os.path.join(data_dir, f"{name}_protein.pdb")
        try:
            protein = pdb_filepath_to_protein(rec_pdb_path)
        except Exception as e:
            log.warning(f"Error in parsing {rec_pdb_path} due to: {e}")
            continue
        for i, seq in enumerate(protein.letter_sequences):
            sequences.append(seq[1])
            ids.append(f"{name}_chain_{i}")
    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

elif args.dataset == "dockgen":
    names = os.listdir(data_dir)
    sequences = []
    ids = []

    for name in tqdm(names):
        if name == ".DS_Store":
            continue
        if not os.path.exists(os.path.join(data_dir, name, f"{name}_protein_processed.pdb")):
            log.warning(
                f"We are skipping {name} because there was no {name}_protein_processed.pdb"
            )
            continue
        rec_pdb_path = os.path.join(data_dir, name, f"{name}_protein_processed.pdb")
        try:
            protein = pdb_filepath_to_protein(rec_pdb_path)
        except Exception as e:
            log.warning(f"Error in parsing {rec_pdb_path} due to: {e}")
            continue
        for i, seq in enumerate(protein.letter_sequences):
            sequences.append(seq[1])
            ids.append(f"{name}_chain_{i}")
    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")

elif args.dataset == "pdbsidechain":
    metadata = pd.read_csv(args.pdbsidechain_metadata_csv_path)
    sequences = []
    ids = []
    for row in tqdm(metadata.itertuples()):
        name = row.CHAINID
        chain_id = name.split("_")[1]
        rec_pt_path = os.path.join(data_dir, name[1:3], f"{name}.pt")
        rec_pdb_path = os.path.join(data_dir, name[1:3], f"{name}.pdb")
        if not os.path.exists(rec_pt_path):
            log.warning(f"We are skipping {name} because there were no files matching `{name}.pt`")
            continue
        if (
            not os.path.exists(rec_pdb_path)
            or not args.skip_extracting_existing_pdbsidechain_pdb_files
        ):
            try:
                convert_protein_pts_to_pdb([rec_pt_path], rec_pdb_path)
                if not os.path.exists(rec_pdb_path):
                    log.warning(f"Failed to convert PyTorch {rec_pt_path} to PDB {rec_pdb_path}")
                    continue

            except Exception as e:
                log.warning(f"Error in converting {rec_pt_path} to {rec_pdb_path} due to: {e}")
                continue
        try:
            protein = pdb_filepath_to_protein(rec_pdb_path)
        except Exception as e:
            log.warning(f"Error in parsing {rec_pdb_path} due to: {e}")
            continue
        if not (
            len(protein.letter_sequences) > 0
            and len(protein.letter_sequences[0]) == 3
            and protein.letter_sequences[0][1] is not None
        ):
            log.warning(f"We are skipping {name} because the extracted sequence was `None`")
            continue
        sequences.append(protein.letter_sequences[0][1])
        ids.append(f"{name.split('_')[0]}_chain_{chain_id}")
    records = []
    for index, seq in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ""
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")
