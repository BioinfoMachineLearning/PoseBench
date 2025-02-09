# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os

import hydra
import rootutils
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from posebench.utils.data_utils import (
    AMINO_ACID_THREE_TO_ONE,
    MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP,
    parse_inference_inputs_from_dir,
)

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
    config_name="fasta_preparation.yaml",
)
def main(cfg: DictConfig):
    """Prepare reference FASTA sequence file for protein chains in the
    dataset."""
    if cfg.dataset not in ["posebusters_benchmark", "astex_diverse", "dockgen", "casp15"]:
        raise ValueError(f"Dataset {cfg.dataset} is not supported.")

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

    data_dir = [
        name
        for name in os.listdir(cfg.data_dir)
        if os.path.isdir(os.path.join(cfg.data_dir, name)) and (pdb_ids is None or name in pdb_ids)
    ]

    if cfg.dataset == "casp15":
        with open_dict(cfg):
            cfg.data_dir = os.path.join(cfg.data_dir, "targets")

        data_dir = [name for name in os.listdir(cfg.data_dir) if name.endswith("_lig.pdb")]

    # prepare to parse all biomolecule sequences if requested
    smiles_and_pdb_id_list = None

    if cfg.include_all_biomolecules:
        with open_dict(cfg):
            cfg.out_file = cfg.out_file.replace(".fasta", "_all.fasta")

        smiles_and_pdb_id_list = parse_inference_inputs_from_dir(
            cfg.data_dir,
            pdb_ids=pdb_ids,
        )

    chain_suffix = ""
    if cfg.include_all_biomolecules:
        chain_suffix = "protein:"

    entries = []
    for name in tqdm(
        data_dir,
        desc="Processing reference PDB file for reference FASTA sequence file generation",
    ):
        data_subdir = os.path.join(cfg.data_dir, name)
        pdb_filepath = os.path.join(data_subdir, f"{name}_protein.pdb")

        if cfg.dataset == "casp15":
            pdb_filepath = os.path.dirname(pdb_filepath)

        if not os.path.exists(pdb_filepath):
            # NOTE: this supports the DockGen dataset's file formatting
            pdb_filepath = os.path.join(data_subdir, f"{name.split('_')[0]}_processed.pdb")
        if not os.path.exists(pdb_filepath):
            logger.warning(f"Skipping {name} as PDB file not found.")
            continue

        # load the first model of the PDB file
        biopython_parser = PDBParser(QUIET=True)
        models = biopython_parser.get_structure("random_id", pdb_filepath)
        structure = models[0]

        structure_seqs = []
        for chain in structure:
            aa_residues = [residue for residue in chain if is_aa(residue)]
            if cfg.dataset == "casp15":
                # NOTE: for CASP15, we exclude modified (hetero) amino acid residues serving as ligands
                name = name.split("_")[0]
                aa_residues = [residue for residue in aa_residues if residue.id[0] == " "]

            aa_residue_names = [
                MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP[residue.resname]
                for residue in aa_residues
                if residue.resname in MODIFIED_TO_NATURAL_AMINO_ACID_RESNAME_MAP
            ]
            seq = "".join(
                AMINO_ACID_THREE_TO_ONE.get(resname, "X") for resname in aa_residue_names
            )

            # skip if not a protein chain
            if not seq:
                continue

            structure_seqs.append((f"{chain_suffix}{name}_chain_{chain.id}", seq))

        # append SMILES chains if available
        if smiles_and_pdb_id_list is not None:
            target_smiles_and_pdb_id_list = [
                (smi, pdb_id)
                for smiles, pdb_id in smiles_and_pdb_id_list
                for smi in smiles.split(".")
                if pdb_id == name
            ]
            if target_smiles_and_pdb_id_list:
                entries.extend(structure_seqs)
                for chain_index, (smiles, _) in enumerate(target_smiles_and_pdb_id_list):
                    entries.append((f"ligand:{name}_chain_{chain_index}", smiles))
        else:
            entries.extend(structure_seqs)

    records = []
    for seq_id, seq in entries:
        record = SeqRecord(Seq(seq), id=str(seq_id))
        record.description = ""
        records.append(record)
    SeqIO.write(records, cfg.out_file, "fasta")


if __name__ == "__main__":
    main()
