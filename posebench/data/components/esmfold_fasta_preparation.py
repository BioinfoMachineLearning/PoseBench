# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from posebench import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="esmfold_fasta_preparation.yaml",
)
def main(cfg: DictConfig):
    """Derive the FASTA files for all protein sequences in either the PoseBusters Benchmark or
    Astex Diverse sets, in preparation for computing ESM2 sequence embeddings in batch.

    Args: An OmegaConf `DictConfig`.
    """
    dataset = cfg.dataset
    data_dir = cfg.data_dir
    if dataset not in ["posebusters_benchmark", "astex_diverse"]:
        raise ValueError(f"Dataset {dataset} is not supported.")
    names = os.listdir(data_dir)

    biopython_parser = PDBParser()

    three_to_one = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "MSE": "M",  # NOTE: this is almost the same amino acid as `MET`; the sulfur is just replaced by `Selen`
        "PHE": "F",
        "PRO": "P",
        "PYL": "O",
        "SER": "S",
        "SEC": "U",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "ASX": "B",
        "GLX": "Z",
        "XAA": "X",
        "XLE": "J",
    }

    sequences = []
    ids = []
    pbar = tqdm(names)
    for name in pbar:
        pbar.set_description(f"Processing PDB {name} for ESM sequence embedding generation.")
        processed_name = f"{name}_protein.pdb"
        if name == ".DS_Store":
            continue
        if os.path.exists(os.path.join(data_dir, name, processed_name)):
            rec_path = os.path.join(data_dir, name, processed_name)
        else:
            continue
        try:
            structure = biopython_parser.get_structure("random_id", rec_path)
            structure = structure[0]
        except Exception as e:
            logger.error(f"Due to exception {e}, could not parse PDB {name}.")
            continue
        for chain_index, chain in enumerate(structure):
            seq = ""
            for residue in chain:
                if residue.get_resname() == "HOH":
                    continue
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == "CA":
                        c_alpha = list(atom.get_vector())
                    if atom.name == "N":
                        n = list(atom.get_vector())
                    if atom.name == "C":
                        c = list(atom.get_vector())
                if (
                    c_alpha is not None and n is not None and c is not None
                ):  # only append residue if it is an amino acid and not a weird residue-like entity
                    try:
                        seq += three_to_one[residue.get_resname()]
                    except Exception as e:
                        seq += "-"
                        logger.error(
                            f"Due to exception {e}, encountered unknown amino acid {residue.get_resname()} in the complex {name}. Replacing it with a dash (i.e., `-`)."
                        )
            if list(set(seq)) not in [[""], ["-"]]:
                sequences.append(seq)
                ids.append(f"{name}_chain_{chain_index}")
    records = []
    for index, seq in zip(ids, sequences):
        if seq and list(set(seq)) not in [[""], ["-"]]:
            record = SeqRecord(Seq(seq), str(index))
            record.description = ""
            records.append(record)
    SeqIO.write(records, cfg.out_file, "fasta")


if __name__ == "__main__":
    main()
