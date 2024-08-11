# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from Bio import PDB
from omegaconf import DictConfig
from tqdm import tqdm

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def convert_mmcif_to_pdb(mmcif_file: str, pdb_file: str):
    """Convert an mmCIF file to a PDB file."""
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", mmcif_file)

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="convert_mmcif_to_pdb.yaml",
)
def main(cfg: DictConfig):
    """Convert an input directory of mmCIF files to an output directory of PDB files."""
    os.makedirs(cfg.output_pdb_dir, exist_ok=True)

    for id in tqdm(
        os.listdir(cfg.input_mmcif_dir),
        desc=f"Converting mmCIF to PDB for {cfg.dataset}",
    ):
        new_id = id.replace("fold_", "")
        if cfg.lowercase_id:
            # Support the DockGen dataset's hybrid lowercase-uppercase pdb id-CCD ID format
            new_id_parts = new_id.split("_")
            new_id = "_".join([part.lower() for part in new_id_parts[:2]] + new_id_parts[2:])
        else:
            new_id = new_id.upper()
        mmcif_filepath = os.path.join(
            cfg.input_mmcif_dir, id, f"{id}_model_{cfg.model_index_to_select}.cif"
        )
        pdb_filepath = os.path.join(cfg.output_pdb_dir, f"{new_id}.pdb")
        if os.path.isfile(mmcif_filepath):
            convert_mmcif_to_pdb(mmcif_filepath, pdb_filepath)

    logger.info(f"Converted mmCIF files to PDB files for {cfg.dataset} dataset.")


if __name__ == "__main__":
    main()
