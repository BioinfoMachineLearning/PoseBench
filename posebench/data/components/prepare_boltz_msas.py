# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import numpy as np
import pandas as pd
import rootutils
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

from posebench.utils.data_utils import extract_sequences_from_protein_structure_file

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ID_TO_HHBLITS_AA = {
    0: "A",
    1: "C",  # Also U.
    2: "D",  # Also B.
    3: "E",  # Also Z.
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",  # Includes J and O.
    21: "-",
}


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="prepare_boltz_msas.yaml",
)
def main(cfg: DictConfig):
    """Prepare Boltz MSAs for PoseBench."""
    assert os.path.exists(cfg.input_msa_dir), f"Input MSA directory not found: {cfg.input_msa_dir}"
    os.makedirs(cfg.output_msa_dir, exist_ok=True)

    for msa_file in tqdm(os.listdir(cfg.input_msa_dir), desc="Preparing Boltz MSAs"):
        if not msa_file.endswith(".npz"):
            continue

        item = msa_file.split("_protein")[0]
        input_msa_path = os.path.join(cfg.input_msa_dir, msa_file)

        casp_dataset_requested = os.path.basename(cfg.input_data_dir) == "targets"
        if casp_dataset_requested:
            protein_filepath = os.path.join(cfg.input_data_dir, f"{item}_lig.pdb")
        else:
            if cfg.pocket_only_baseline:
                protein_filepath = os.path.join(
                    cfg.input_data_dir,
                    f"{cfg.dataset}_holo_aligned_predicted_structures_bs_cropped",
                    f"{item}_holo_aligned_predicted_protein.pdb",
                )
                if not os.path.exists(protein_filepath):
                    logger.warning(
                        f"Protein structure file not found for PDB ID {item}. Skipping..."
                    )
                    continue
            else:
                protein_item = item.split("_")[0] if cfg.dataset == "dockgen" else item
                protein_file_suffix = "_processed" if cfg.dataset == "dockgen" else "_protein"
                protein_filepath = os.path.join(
                    cfg.input_data_dir, item, f"{protein_item}{protein_file_suffix}.pdb"
                )
        protein_sequence_list = [
            seq
            for seq in extract_sequences_from_protein_structure_file(protein_filepath)
            if len(seq) > 0
        ]

        try:
            input_msa = dict(np.load(input_msa_path))

            assert (
                len(protein_sequence_list) == input_msa["n"]
            ), f"Number of chains in protein structure file ({len(protein_sequence_list)}) does not match number of MSA chains ({input_msa['n']}) for {item}. Skipping..."

            for chain_index in range(input_msa["n"]):
                output_msa_path = os.path.join(
                    cfg.output_msa_dir, item + f"_chain_{chain_index}.csv"
                )
                if os.path.exists(output_msa_path) and cfg.skip_existing:
                    logger.info(f"MSA already exists: {output_msa_path}. Skipping...")
                    continue

                protein_sequence = protein_sequence_list[chain_index]
                msa_sequence = "".join(
                    ID_TO_HHBLITS_AA[c] for c in input_msa[f"msa_{chain_index}"][0]
                )

                max_sequence_len = max(len(protein_sequence), len(msa_sequence))

                if protein_sequence != msa_sequence and len(protein_sequence) == len(msa_sequence):
                    logger.warning(
                        f"Input protein sequence {protein_sequence} does not match first MSA sequence {msa_sequence} for chain {chain_index} in {item}. Using input protein sequence instead..."
                    )
                    msa_sequence = protein_sequence
                elif protein_sequence != msa_sequence:
                    logger.warning(
                        f"Input protein sequence {protein_sequence} does not match first MSA sequence length of {msa_sequence} for chain {chain_index} in {item}. Using input protein sequence instead and right-padding rest of MSA..."
                    )
                    msa_sequence = protein_sequence

                output_msas = [
                    {
                        "sequence": (
                            msa_sequence
                            if seq_index == 0
                            else "".join(ID_TO_HHBLITS_AA[c] for c in seq).rjust(
                                max_sequence_len, "-"
                            )
                        ),
                        "key": (
                            seq_index
                            if input_msa[f"is_paired_{chain_index}"][seq_index].item() is True
                            else ""
                        ),
                    }
                    for seq_index, seq in enumerate(input_msa[f"msa_{chain_index}"])
                ]
                output_msa_df = pd.DataFrame(output_msas)

                logger.info(
                    f"Converting chain MSA to DataFrame: {input_msa_path} -> {output_msa_path}"
                )
                output_msa_df.to_csv(output_msa_path, index=False)

        except Exception as e:
            logger.error(f"Failed to process MSA {input_msa_path} due to: {e}. Skipping...")


if __name__ == "__main__":
    main()
