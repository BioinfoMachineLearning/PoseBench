# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging

import hydra
import rootutils
from beartype.typing import Dict
from omegaconf import DictConfig

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


def read_fasta(filename: str) -> Dict[str, str]:
    """Read sequences from a FASTA file and return them as a dictionary.

    :param filename: Path to the input FASTA file.
    :return: Dictionary containing sequences with identifiers as keys.
    """
    sequences = {}
    current_id = ""
    with open(filename) as file:
        for line in file:
            if line.startswith(">"):
                current_id = line.strip()[1:]
                sequences[current_id] = ""
            else:
                sequences[current_id] += line.strip()
    return sequences


def combine_sequences(sequences: Dict[str, str]) -> Dict[str, str]:
    """Combine sequences of the same protein complex by their PDB codes.

    :param sequences: Dictionary containing sequences with identifiers
        as keys.
    :return: Dictionary containing combined sequences with PDB codes as
        keys.
    """
    combined_sequences = {}
    for identifier, sequence in sequences.items():
        pdb_code = identifier.split("_chain_")[0]
        if pdb_code in combined_sequences:
            combined_sequences[pdb_code] += ":" + sequence
        else:
            combined_sequences[pdb_code] = sequence
    return combined_sequences


def write_combined_fasta(combined_sequences: Dict[str, str], output_filename: str) -> None:
    """Write combined sequences to an output FASTA file.

    :param combined_sequences: Dictionary containing combined sequences
        with PDB codes as keys.
    :param output_filename: Path to the output FASTA file.
    """
    with open(output_filename, "w") as file:
        for pdb_code, sequence in combined_sequences.items():
            file.write(f">{pdb_code}\n{sequence}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="esmfold_sequence_preparation.yaml",
)
def main(cfg: DictConfig):
    """Read protein chain sequences from a FASTA file, combine sequences of the
    same protein complex, and write the combined sequences to an output FASTA
    file.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    sequences = read_fasta(cfg.input_fasta_file)
    combined_sequences = combine_sequences(sequences)
    write_combined_fasta(combined_sequences, cfg.output_fasta_file)

    logger.info("ESMFold FASTA file preparation complete.")


if __name__ == "__main__":
    main()
