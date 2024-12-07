import hydra
import rootutils
from beartype.typing import Dict, Literal
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock import register_custom_omegaconf_resolvers
from flowdock.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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


def combine_sequences(
    sequences: Dict[str, str], dataset: Literal["pdbbind", "moad", "dockgen", "pdbsidechain"]
) -> Dict[str, str]:
    """Combine sequences of the same protein complex by their PDB codes.

    :param sequences: Dictionary containing sequences with identifiers as keys.
    :param dataset: Name of the dataset for which to combine sequences.
    :return: Dictionary containing combined sequences with PDB codes as keys.
    """
    combined_sequences = {}
    for identifier, sequence in sequences.items():
        if sequence and list(set(sequence)) not in [[""], ["-"]]:
            if dataset == "pdbbind":
                pdb_code = identifier.split("_")[0]
            elif dataset == "moad":
                pdb_code = "_".join(identifier.split("_")[0:2])
            elif dataset == "dockgen":
                pdb_code = identifier.split("_chain_")[0]
            elif dataset == "pdbsidechain":
                # NOTE: the van der Mers dataset selects only a single chain from each PDB complex structure
                pdb_code = identifier.split("_")[0] + "_" + identifier.split("_")[-1]
            else:
                raise ValueError(f"Invalid dataset: {dataset}")
            if pdb_code in combined_sequences:
                combined_sequences[pdb_code] += ":" + sequence
            else:
                combined_sequences[pdb_code] = sequence
    return combined_sequences


def write_combined_fasta(combined_sequences: Dict[str, str], output_filename: str) -> None:
    """Write combined sequences to an output FASTA file.

    :param combined_sequences: Dictionary containing combined sequences with PDB codes as keys.
    :param output_filename: Path to the output FASTA file.
    """
    with open(output_filename, "w") as file:
        for pdb_code, sequence in combined_sequences.items():
            file.write(f">{pdb_code}\n{sequence}\n")


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data",
    config_name="esmfold_sequence_preparation.yaml",
)
def main(cfg: DictConfig):
    """Read protein chain sequences from a FASTA file, combine sequences of the same protein
    complex, and write the combined sequences to an output FASTA file.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    sequences = read_fasta(cfg.input_fasta_file)
    combined_sequences = combine_sequences(sequences, cfg.dataset)
    write_combined_fasta(combined_sequences, cfg.output_fasta_file)

    log.info("ESMFold FASTA file preparation complete.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
