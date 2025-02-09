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
    config_name="prepare_chai_msas.yaml",
)
def main(cfg: DictConfig):
    """Prepare Chai MSAs for PoseBench."""
    assert os.path.exists(cfg.input_msa_dir), f"Input MSA directory not found: {cfg.input_msa_dir}"
    os.makedirs(cfg.output_msa_dir, exist_ok=True)

    for msa_file in tqdm(os.listdir(cfg.input_msa_dir), desc="Preparing Chai-1 MSAs"):
        if not msa_file.endswith(".npz"):
            continue

        item = msa_file.split("_protein")[0]
        input_msa_path = os.path.join(cfg.input_msa_dir, msa_file)

        try:
            input_msa = dict(np.load(input_msa_path))

            for chain_index in range(input_msa["n"]):
                output_msa_path = os.path.join(
                    cfg.output_msa_dir, item + f"_chain_{chain_index}.aligned.pqt"
                )
                if os.path.exists(output_msa_path) and cfg.skip_existing:
                    logger.info(f"MSA already exists: {output_msa_path}. Skipping...")
                    continue

                output_msas = [
                    {
                        "sequence": "".join(ID_TO_HHBLITS_AA[c] for c in seq),
                        "source_database": "query" if seq_index == 0 else "uniref90",
                        "pairing_key": (
                            f"sequence:{seq_index}"
                            if input_msa[f"is_paired_{chain_index}"][seq_index].item() is True
                            else ""
                        ),
                        "comment": "",
                    }
                    for seq_index, seq in enumerate(input_msa[f"msa_{chain_index}"])
                ]
                output_msa_df = pd.DataFrame(output_msas)

                logger.info(
                    f"Converting chain MSA to DataFrame: {input_msa_path} -> {output_msa_path}"
                )
                output_msa_df.to_parquet(output_msa_path)

        except Exception as e:
            logger.error(f"Failed to process MSA {input_msa_path} due to: {e}. Skipping...")


if __name__ == "__main__":
    main()
