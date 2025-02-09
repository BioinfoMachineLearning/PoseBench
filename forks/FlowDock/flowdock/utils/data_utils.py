import copy
import dataclasses
import io
import os
import pickle  # nosec
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype import beartype
from beartype.typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from Bio.PDB import PDBParser, Polypeptide
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from openfold.np.protein import Protein as OFProtein
from openfold.np.protein import to_pdb as of_to_pdb
from openfold.utils import tensor_utils
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from flowdock.data.components.process_mols import read_molecule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components import residue_constants
from flowdock.data.components.mol_features import (
    attach_pair_idx_and_encodings,
    collate_numpy_samples,
    process_mol_file,
)
from flowdock.data.components.residue_constants import restype_1to3 as af_restype_1to3
from flowdock.data.components.residue_constants import restypes as af_restypes
from flowdock.models.components.transforms import LatentCoordinateConverter
from flowdock.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

MODEL_BATCH = Dict[str, Any]

MOAD_UNIT_CONVERSION_DICT = {
    np.nan: np.nan,  # NaN to NaN
    "uM": 1e-6,  # micromolar to M
    "nM": 1e-9,  # nanomolar to M
    "mM": 1e-3,  # millimolar to M
    "pM": 1e-12,  # picomolar to M
    "M": 1,  # Molar to M
    "M^-1": 1,  # Reciprocal Molar to M
    "fM": 1e-15,  # femtomolar to M
}

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

# From: https://github.com/uw-ipd/RoseTTAFold2
NUM_TO_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "MAS",
]
AA_TO_NUM = {x: i for i, x in enumerate(NUM_TO_AA)}
AA_TO_LONG = [
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # ala
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " NE ",
        " CZ ",
        " NH1",
        " NH2",
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        " HE ",
        "1HH1",
        "2HH1",
        "1HH2",
        "2HH2",
    ),  # arg
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " OD1",
        " ND2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HD2",
        "2HD2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # asn
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " OD1",
        " OD2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # asp
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " SG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # cys
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " OE1",
        " NE2",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HE2",
        "2HE2",
        None,
        None,
        None,
        None,
        None,
    ),  # gln
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " OE1",
        " OE2",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # glu
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        "1HA ",
        "2HA ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # gly
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " ND1",
        " CD2",
        " CE1",
        " NE2",
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD2",
        " HE1",
        " HE2",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # his
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG1",
        " CG2",
        " CD1",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        " HB ",
        "1HG2",
        "2HG2",
        "3HG2",
        "1HG1",
        "2HG1",
        "1HD1",
        "2HD1",
        "3HD1",
        None,
        None,
    ),  # ile
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HG ",
        "1HD1",
        "2HD1",
        "3HD1",
        "1HD2",
        "2HD2",
        "3HD2",
        None,
        None,
    ),  # leu
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        " CE ",
        " NZ ",
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        "1HE ",
        "2HE ",
        "1HZ ",
        "2HZ ",
        "3HZ ",
    ),  # lys
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " SD ",
        " CE ",
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HE ",
        "2HE ",
        "3HE ",
        None,
        None,
        None,
        None,
    ),  # met
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " CE1",
        " CE2",
        " CZ ",
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HD2",
        " HE1",
        " HE2",
        " HZ ",
        None,
        None,
        None,
        None,
    ),  # phe
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " HA ",
        "1HB ",
        "2HB ",
        "1HG ",
        "2HG ",
        "1HD ",
        "2HD ",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # pro
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " OG ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HG ",
        " HA ",
        "1HB ",
        "2HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # ser
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " OG1",
        " CG2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HG1",
        " HA ",
        " HB ",
        "1HG2",
        "2HG2",
        "3HG2",
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # thr
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " NE1",
        " CE2",
        " CE3",
        " CZ2",
        " CZ3",
        " CH2",
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HE1",
        " HZ2",
        " HH2",
        " HZ3",
        " HE3",
        None,
        None,
        None,
    ),  # trp
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG ",
        " CD1",
        " CD2",
        " CE1",
        " CE2",
        " CZ ",
        " OH ",
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        " HD1",
        " HE1",
        " HE2",
        " HD2",
        " HH ",
        None,
        None,
        None,
        None,
    ),  # tyr
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        " CG1",
        " CG2",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        " HB ",
        "1HG1",
        "2HG1",
        "3HG1",
        "1HG2",
        "2HG2",
        "3HG2",
        None,
        None,
        None,
        None,
    ),  # val
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # unk
    (
        " N  ",
        " CA ",
        " C  ",
        " O  ",
        " CB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        " H  ",
        " HA ",
        "1HB ",
        "2HB ",
        "3HB ",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),  # mask
]


@dataclasses.dataclass()
class FDProtein:
    """Protein structure representation."""

    # The first entry stores amino acid sequence in letter representation.
    # The second entry stores a 0-1 mask for observed standard residues.
    # Non-standard residues are mapped to <mask> to interact with protein language models.
    letter_sequences: List[Tuple[str, str, np.ndarray]]

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, atom_type_num, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Added
    # Integer for atom type.
    atomtypes: np.ndarray  # [num_res, element_type_num]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, atom_type_num]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, atom_type_num]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


@beartype
def combine_molecules(molecule_list: List[Chem.Mol]) -> Chem.Mol:
    """Combine a list of RDKit molecules into a single molecule.

    :param molecule_list: A list of RDKit molecules.
    :return: A single RDKit molecule.
    """
    # Initialize the combined molecule with the first molecule in the list
    new_mol = molecule_list[0]

    # Iterate through the remaining molecules and combine them pairwise
    for mol in molecule_list[1:]:
        new_mol = Chem.CombineMols(new_mol, mol)

    return new_mol


@beartype
def pdb_filepath_to_protein(
    pdb_filepath: str,
    model_id: int = 0,
    atom_occupancy_min_threshold: int = 0.5,
    filter_out_hetero_residues: bool = True,
    allow_insertion_code: bool = True,
    accept_only_valid_backbone_residues: bool = True,
    chain_id: Optional[Union[List[str], str]] = None,
    bounding_box: Optional[np.ndarray] = None,
    res_start: Optional[int] = None,
    res_end: Optional[int] = None,
) -> FDProtein:
    """Takes a PDB filepath and constructs a FDProtein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored. All water residues will be ignored.
      All hetero residues will be ignored if `filter_out_hetero_residues` is `True`.
      All residues without valid positions for their N, Ca, C, and O atoms will be ignored.

    Adapted from: https://github.com/aqlaboratory/openfold and https://github.com/zrqiao/NeuralPLexer

    :param pdb_filepath: The filepath to the PDB file to parse.
    :param model_id: The model number to parse.
    :param atom_occupancy_min_threshold: The minimum occupancy threshold for atoms.
    :param filter_out_hetero_residues: If True, then hetero residues will be ignored.
    :param allow_insertion_code: If True, residues with insertion codes are parsed.
    :param accept_only_valid_backbone_residues: If True, only residues with valid N, Ca, C, and O atoms are parsed.
    :param chain_id: If `chain_id` is specified (e.g., `[A, B]`), then only those chains
        are parsed. Otherwise, all chains are parsed.
    :param bounding_box: If provided, only chains with backbone intersecting with
        the box are parsed.
    :param res_start: If provided, only residues with index >= res_start are parsed.
    :param res_end: If provided, only residues with index <= res_end are parsed.
    :return: A new `FDProtein` parsed from the PDB contents.
    """
    assert pdb_filepath.endswith(".pdb"), f"Invalid file extension: {pdb_filepath}"
    assert os.path.exists(pdb_filepath), f"File not found: {pdb_filepath}"

    with open(pdb_filepath) as pdb_fh:
        pdb_str = pdb_fh.read()

    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if not (0 <= model_id < len(models)):
        raise ValueError(f"Model ID {model_id} is out of range")
    model = models[model_id]
    if isinstance(chain_id, str):
        chain_id = [chain_id]

    seq = []
    atom_positions = []
    aatype = []
    atomtypes = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.get_id() not in chain_id:
            log.info(f"In `pdb_filepath_to_protein()`, skipping chain {chain.get_id()}")
            continue
        if bounding_box is not None:
            ca_pos = np.array([res["CA"].get_coord() for res in chain if res.has_id("CA")])
            ca_in_box = (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1])
            if not np.any(np.all(ca_in_box, axis=1), axis=0):
                log.info(
                    f"In `pdb_filepath_to_protein()`, skipping chain {chain.get_id()} as it is not in the bounding box"
                )
                continue
        for res_idx, res in enumerate(chain):
            if res_start is not None and res_idx < res_start:
                continue
            if res_end is not None and res_idx > res_end:
                continue
            if res.get_resname() == "HOH" or (
                filter_out_hetero_residues and len(res.get_id()[0]) > 1
            ):
                log.info(
                    f"In `pdb_filepath_to_protein()`, skipping residue {res.get_id()} as it is a water residue or a hetero residue."
                )
                continue
            # strict bounding
            if bounding_box is not None:
                if not res.has_id("CA"):
                    continue
                ca_pos = res["CA"].get_coord()
                ca_in_box = (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1])
                if not np.all(ca_in_box):
                    continue
            if res.id[2] != " ":
                if allow_insertion_code:
                    log.warning(
                        f"PDB contains an insertion code at chain {chain.id} and residue "
                        f"index {res.id[1]} and `allow_insertion_code` is set to True. "
                        "Please ensure the residue indices are consecutive before performing downstream analysis."
                    )
                else:
                    raise ValueError(
                        f"PDB contains an insertion code at chain {chain.id} and residue "
                        f"index {res.id[1]}. Such samples are not supported by default."
                    )
            # NOTE: like the AlphaFold parser, we parse all non-standard residues
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            eletypes = np.zeros((residue_constants.atom_type_num,))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            # sidechain_atom_order = 3
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                # (potentially) remove atoms that are too flexible
                if atom.occupancy and atom.occupancy < atom_occupancy_min_threshold:
                    # NOTE: this is not a standard AlphaFold filter;
                    # it was originally used in the NeuralPLexer parser
                    continue
                eletypes[residue_constants.atom_order[atom.name]] = residue_constants.element_id[
                    atom.element
                ]
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if accept_only_valid_backbone_residues and (np.sum(mask[:3]) < 3 or mask[4] < 1):
                # as requested, skip if the backbone atoms are not resolved (NOTE: atom37 ordering is N, Ca, C, CB, O, ... -> we only check N, Ca, C, and O)
                log.warning(
                    f"In `pdb_filepath_to_protein()`, skipping residue {res.id[1]} in chain {chain.id} as not enough backbone atoms are reported."
                )
                continue
            seq.append(res_shortname)
            aatype.append(restype_idx)
            atomtypes.append(eletypes)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # NOTE: chain IDs are usually characters, so we will map these to integers.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    # evaluate the gapless protein sequence for each chain
    seqs = []
    last_chain_idx = -1
    last_chain_seq = None
    last_chain_mask = None
    for site_idx in range(len(seq)):
        if chain_ids[site_idx] != last_chain_idx:
            if last_chain_seq is not None:
                last_chain_seq = "".join(last_chain_seq)
                seqs.append((last_chain_idx, "".join(last_chain_seq), np.array(last_chain_mask)))
            last_chain_idx = chain_ids[site_idx]
            last_chain_seq = []
            last_chain_mask = []
            last_res_id = -999
        if residue_index[site_idx] <= last_res_id:
            raise ValueError(
                f"PDB residue index is not monotonous at chain {chain.id} and residue "
                f"index {res.id[1]}. The sample is discarded."
            )
        elif last_res_id == -999:
            gap_size = 0
        else:
            gap_size = residue_index[site_idx] - last_res_id - 1
        for _ in range(gap_size):
            last_chain_seq.append("<mask>")
            last_chain_mask.append(False)
        last_chain_seq.append(seq[site_idx])
        last_chain_mask.append(True)
    seqs.append((last_chain_idx, "".join(last_chain_seq), np.array(last_chain_mask)))

    for chain_seq in seqs:
        if np.mean(chain_seq[2]) < 0.75:
            raise ValueError(
                f"The PDB structure residue coverage for {chain.id}"
                f"is below 75%. The sample is discarded."
            )

    return FDProtein(
        letter_sequences=seqs,
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        atomtypes=np.array(atomtypes),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def get_protein_indexer(rec_features: Dict[str, Any], edge_cutoff: int = 50) -> Dict[str, Any]:
    """Get the protein indexer.

    :param rec_features: Protein features.
    :param edge_cutoff: Edge cutoff.
    :return: Protein indexer.
    """
    # Using a large cutoff here; dynamically remove edges along diffusion
    res_xyzs = rec_features["res_atom_positions"]
    n_res = len(res_xyzs)
    res_atom_masks = rec_features["res_atom_mask"]
    ca_xyzs = res_xyzs[:, 1, :]
    distances = np.linalg.norm(ca_xyzs[:, np.newaxis, :] - ca_xyzs[np.newaxis, :, :], axis=2)
    edge_mask = distances < edge_cutoff
    # Mask out residues where the backbone is not resolved
    res_mask = np.all(~res_atom_masks[:, :3], axis=1)
    edge_mask[res_mask, :] = 0
    edge_mask[:, res_mask] = 0
    res_ids = np.broadcast_to(np.arange(n_res), (n_res, n_res))
    src_nid, dst_nid = res_ids[edge_mask], res_ids.T[edge_mask]

    indexer = {
        "gather_idx_a_chainid": rec_features["res_chain_id"],
        "gather_idx_a_structid": np.zeros((n_res,), dtype=np.int_),
        "gather_idx_ab_a": src_nid,
        "gather_idx_ab_b": dst_nid,
    }
    return indexer


def process_protein(
    af_protein: FDProtein,
    bounding_box: Optional[np.ndarray] = None,
    no_indexer: bool = True,
    sample_name: str = "",
    sequences_to_embeddings: Optional[Dict[str, np.ndarray]] = None,
    plddt: Optional[Iterable[float]] = None,
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process protein data.

    :param af_protein: FDProtein object.
    :param bounding_box: Bounding box to filter atoms.
    :param no_indexer: If True, the indexer is not added to the output.
    :param sample_name: Name of the sample.
    :param prefix_key: Prefix key.
    :param sequences_to_embeddings: Optional dictionary mapping sequences to embeddings.
    :param plddt: Optional pLDDT values.
    :param chain_id: Optional chain ID for parsing of LM embeddings.
    :return: Processed protein data.
    """
    lm_embeddings = None if sequences_to_embeddings is None else []
    if sequences_to_embeddings is not None:
        if chain_id is not None and len(af_protein.letter_sequences) == 1:
            chain_seq = af_protein.letter_sequences[0][1]
            chain_mask = af_protein.letter_sequences[0][2]
            chain_seq_masked = "".join(np.array(list(chain_seq))[chain_mask])
            lm_embeddings.append(sequences_to_embeddings[chain_seq_masked + f":{chain_id}"])
        else:
            for i, (_, chain_seq, chain_mask) in enumerate(af_protein.letter_sequences):
                chain_seq_masked = "".join(np.array(list(chain_seq))[chain_mask])
                if i in sequences_to_embeddings:
                    lm_embeddings.append(sequences_to_embeddings[i])
                elif chain_seq_masked + f":{i}" in sequences_to_embeddings:
                    lm_embeddings.append(sequences_to_embeddings[chain_seq_masked + f":{i}"])
                else:
                    raise ValueError(
                        f"Sequence {chain_seq_masked}:{i} not found in the provided embeddings."
                    )
        lm_embeddings = np.concatenate(lm_embeddings, axis=0)
        assert len(lm_embeddings) == len(
            af_protein.aatype
        ), f"LM sequence length must match OpenFold-parsed sequence length: {len(lm_embeddings)} != {len(af_protein.aatype)}"
    if bounding_box:
        raise NotImplementedError
        ca_pos = af_protein.atom_positions[:, 1]
        ca_in_box = np.all(
            (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1]),
            axis=1,
        )
        af_protein.atom_positions = af_protein.atom_positions[ca_in_box]
        af_protein.aatype = af_protein.aatype[ca_in_box]
        af_protein.atomtypes = af_protein.atomtypes[ca_in_box]
        af_protein.atom_mask = af_protein.atom_mask[ca_in_box]
        af_protein.chain_index = af_protein.chain_index[ca_in_box]
        af_protein.b_factors = af_protein.b_factors[ca_in_box]
    chain_seqs = [
        (sample_name + seq_data[0], seq_data[1]) for seq_data in af_protein.letter_sequences
    ]
    chain_masks = [seq_data[2] for seq_data in af_protein.letter_sequences]
    features = {
        "res_atom_positions": af_protein.atom_positions,
        "res_type": np.int_(af_protein.aatype),
        "res_atom_types": np.int_(af_protein.atomtypes),
        "res_atom_mask": np.bool_(af_protein.atom_mask),
        "res_chain_id": np.int_(af_protein.chain_index),
        "residue_index": np.int_(af_protein.residue_index),
        "sequence_res_mask": np.bool_(np.concatenate(chain_masks)),
    }
    if lm_embeddings is not None:
        features.update({"lm_embeddings": lm_embeddings})
    if plddt:
        features.update({"pLDDT": np.array(plddt) / 100})
    n_res = len(af_protein.atom_positions)
    metadata = {
        "num_structid": 1,
        "num_a": n_res,
        "num_b": n_res,
        "num_chainid": max(af_protein.chain_index) + 1,
    }
    if no_indexer:
        return {
            "metadata": metadata,
            "indexer": {
                "gather_idx_a_chainid": features["res_chain_id"],
                "gather_idx_a_structid": np.zeros((n_res,), dtype=np.int_),
            },
            "features": features,
            "misc": {"sequence_data": chain_seqs},
        }
    return {
        "metadata": metadata,
        "indexer": get_protein_indexer(features),
        "features": features,
        "misc": {"sequence_data": chain_seqs},
    }


def merge_protein_and_ligands(
    lig_samples: List[Dict[str, Any]],
    rec_sample: Dict[str, Any],
    n_lig_patches: int,
    label: Optional[str] = None,
    random_lig_placement: bool = False,
    subsample_frames: bool = False,
) -> Dict[str, Any]:
    """Merge protein and ligands.

    :param lig_samples: List of ligand samples.
    :param rec_sample: Receptor sample.
    :param n_lig_patches: Number of ligand patches.
    :param label: Optional label.
    :param random_lig_placement: If True, randomly place ligands into the box.
    :param subsample_frames: If True, subsample frames.
    :return: Merged protein and ligands.
    """
    # Assign frame sampling rate to each ligand
    num_ligands = len(lig_samples)
    if num_ligands > 0:
        num_frames_sqrt = np.sqrt(
            np.array([lig_sample["metadata"]["num_ijk"] for lig_sample in lig_samples])
        )
        if (n_lig_patches > sum(num_frames_sqrt)) and subsample_frames:
            n_lig_patches = random.randint(int(sum(num_frames_sqrt)), n_lig_patches)  # nosec
        max_n_frames_arr = num_frames_sqrt * (n_lig_patches / sum(num_frames_sqrt))
        max_n_frames_arr = max_n_frames_arr.astype(np.int_)
        lig_samples = [
            attach_pair_idx_and_encodings(lig_sample, max_n_frames=max_n_frames_arr[lig_idx])
            for lig_idx, lig_sample in enumerate(lig_samples)
        ]

    if random_lig_placement:
        # Data augmentation, randomly placing into box
        rec_coords = rec_sample["features"]["res_atom_positions"][
            rec_sample["features"]["res_atom_mask"]
        ]
        box_lbound, box_ubound = (
            np.amin(rec_coords, axis=0),
            np.amax(rec_coords, axis=0),
        )
        for sid, lig_sample in enumerate(lig_samples):
            lig_coords = lig_sample["features"]["sdf_coordinates"]
            lig_center = np.mean(lig_coords, axis=0)
            is_clash = True
            padding = 0
            while is_clash:
                padding += 1.0
                new_center = np.random.uniform(low=box_lbound - padding, high=box_ubound + padding)
                new_lig_coords = lig_coords + (new_center - lig_center)[None, :]
                intermol_distmat = np.linalg.norm(
                    new_lig_coords[None, :] - rec_coords[:, None, :],
                    axis=2,
                )
                if np.amin(intermol_distmat) > 4.0:
                    is_clash = False
            lig_samples[sid]["features"]["augmented_coordinates"] = new_lig_coords
            del lig_samples[sid]["features"]["sdf_coordinates"]
    lig_sample_merged = collate_numpy_samples(lig_samples)
    merged = {
        "metadata": {**lig_sample_merged["metadata"], **rec_sample["metadata"]},
        "features": {**lig_sample_merged["features"], **rec_sample["features"]},
        "indexer": {**lig_sample_merged["indexer"], **rec_sample["indexer"]},
        "misc": {**lig_sample_merged["misc"], **rec_sample["misc"]},
    }
    merged["metadata"]["num_structid"] = 1
    if "num_molid" in merged["metadata"]:
        merged["indexer"]["gather_idx_i_structid"] = np.zeros(
            lig_sample_merged["metadata"]["num_i"], dtype=np.int_
        )
        merged["indexer"]["gather_idx_ijk_structid"] = np.zeros(
            lig_sample_merged["metadata"]["num_ijk"], dtype=np.int_
        )
    assert np.sum(merged["features"]["res_atom_mask"]) > 0
    if label is not None:
        merged["labels"] = np.array([label])
    return merged


@beartype
def convert_protein_pts_to_pdb(
    processed_pt_filenames: List[str],
    processed_pdb_filename: str,
) -> None:
    """Convert protein chain structures in `.pt` format to a single `.pdb` file.

    :param processed_pt_filenames: Filepaths to `.pt` files containing the protein structure.
    :param processed_pdb_filename: Filepath to the output `.pdb` file.
    """
    structure = Structure("protein_structure")
    model = Model(0)
    structure.add(model)

    for processed_pt_filename in processed_pt_filenames:
        chain_id = Path(processed_pt_filename).stem.split("_")[1]
        chain = Chain(chain_id)
        model.add(chain)

        protein_chain_data = torch.load(processed_pt_filename)
        for residue_id in range(len(protein_chain_data["seq"])):
            aa = Polypeptide.one_to_three(protein_chain_data["seq"][residue_id])
            aa_atoms = AA_TO_LONG[AA_TO_NUM[aa]][:14]

            residue = Residue((" ", residue_id + 1, " "), aa, "")
            chain.add(residue)

            for atom_name, coord, mask_atom, bfac, occ in zip(
                aa_atoms,
                protein_chain_data["xyz"][residue_id],
                protein_chain_data["mask"][residue_id],
                protein_chain_data["bfac"][residue_id],
                protein_chain_data["occ"][residue_id],
            ):
                if atom_name is None or mask_atom.item() == 0:  # skip masked atoms
                    continue
                atom = Atom(
                    atom_name.strip(), coord.numpy(), bfac.item(), occ.item(), " ", atom_name, 0
                )
                residue.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(processed_pdb_filename)


@beartype
def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype: Optional[np.ndarray] = None,
    b_factors: Optional[np.ndarray] = None,
) -> OFProtein:
    """Create a full protein from an Nx37x3 array of atom positions, where N is the number of
    residues in the protein.

    :param atom37: Nx37x3 array of atom positions
    :param atom37_mask: Nx37x3 array of atom masks
    :param aatype: N-sized array of amino acid types
    :param b_factors: Nx37x3 array of B-factors
    :return: OFProtein object
    """
    assert atom37.ndim == 3, "atom37 must be 3D"
    assert atom37.shape[-1] == 3, "atom37 must have 3 coordinates"
    assert atom37.shape[-2] == 37, "atom37 must have 37 atoms per residue"

    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)

    return OFProtein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


@beartype
def get_mol_with_new_conformer_coords(mol: Chem.Mol, new_coords: np.ndarray) -> Chem.Mol:
    """Create a new version of an RDKit `Chem.Mol` with new conformer coordinates.

    :param mol: RDKit Mol object.
    :param new_coords: Numpy array of shape (num_atoms, 3) with new 3D coordinates.
    :return: A new RDKit Mol object with the updated conformer coordinates.
    """
    num_atoms = mol.GetNumAtoms()
    assert new_coords.shape == (
        num_atoms,
        3,
    ), f"`new_coords` must have shape `({num_atoms}, 3), but it has shape `{new_coords.shape}`"

    # Create a new molecule
    new_mol = Chem.Mol(copy.deepcopy(mol))
    new_mol.RemoveAllConformers()

    # Create a new conformer and set atom positions
    new_conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        x, y, z = new_coords[i].astype(np.double)
        new_conf.SetAtomPosition(i, Point3D(x, y, z))

    # Add the conformer to the molecule
    new_mol.AddConformer(new_conf, assignId=True)

    return new_mol


@beartype
def get_rc_tensor(rc_np: np.ndarray, aatype: torch.Tensor) -> torch.Tensor:
    """Get a residue constant tensor from a numpy array based on the amino acid type.

    :param rc_np: Numpy array of residue constants.
    :param aatype: Amino acid type tensor.
    :return: Residue constant tensor.
    """
    return torch.tensor(rc_np, device=aatype.device)[aatype]


@beartype
def atom37_to_atom14(
    aatype: torch.Tensor, all_atom_pos: torch.Tensor, all_atom_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert OpenFold's `atom37` positions to `atom14` positions.

    :param aatype: Amino acid type tensor of shape `(..., num_residues)`.
    :param all_atom_pos: All-atom (`atom37`) positions tensor of shape `(..., num_residues, 37, 3)`.
    :param all_atom_mask: All-atom (`atom37`) mask tensor of shape `(..., num_residues, 37)`.
    :return: Tuple of `atom14` positions and `atom14` mask of shapes `(..., num_residues, 14, 3)` and `(..., num_residues, 14)`, respectively.
    """
    residx_atom14_to_atom37 = get_rc_tensor(
        residue_constants.RESTYPE_ATOM14_TO_ATOM37, aatype  # (..., num_residues)
    )  # (..., num_residues, 14)
    no_batch_dims = len(aatype.shape) - 1
    atom14_mask = tensor_utils.batched_gather(
        all_atom_mask,  # (..., num_residues, 37)
        residx_atom14_to_atom37,  # (..., num_residues, 14)
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ).to(
        all_atom_pos.dtype
    )  # (..., num_residues, 14)
    # create a mask for known groundtruth positions
    atom14_mask *= get_rc_tensor(residue_constants.RESTYPE_ATOM14_MASK, aatype)
    # gather the groundtruth positions
    atom14_positions = tensor_utils.batched_gather(
        all_atom_pos,  # (..., num_residues, 37, 3)
        residx_atom14_to_atom37,
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    )
    atom14_positions = atom14_mask[..., None] * atom14_positions
    return (atom14_positions, atom14_mask)  # (..., num_residues, 14, 3)  # (..., num_residues, 14)


@beartype
def output_to_pdb(output: Dict) -> List[str]:
    """Return a PDB (file) string given a model output dictionary.

    :param output: A dictionary containing the model output.
    :return: A list of PDB strings.
    """
    final_atom_positions = output["all_atom_positions"]
    output = {k: v.to("cpu").numpy() for k, v in output.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = output["all_atom_mask"]
    pdbs = []
    for i in range(output["aatype"].shape[0]):
        aa = output["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=output["plddt"][i],
            chain_index=output["chain_index"][i] if "chain_index" in output else None,
        )
        pdbs.append(of_to_pdb(pred))
    return pdbs


@beartype
def load_hetero_data_graph_as_pdb_file_pair(
    input_filepaths: Tuple[str, List[str]], temp_dir_path: Path
) -> Dict[str, Dict[str, Union[str, int]]]:
    """Load a pair of protein PDB files (and their metadata also including ligand statistics) from
    a single preprocessed `HeteroData` graph file and given RDKit ligands, respectively.

    :param input_filepaths: Tuple of filepaths to the preprocessed `HeteroData` graph and the RDKit ligands.
    :param temp_dir_path: Path to temporary directory for storing the PDB files.
    :return: A dictionary containing the filepaths to the PDB files and their lengths as well as ligand statistics.
    """
    hetero_graph_filepath = input_filepaths[0]
    rdkit_ligand_filepaths = input_filepaths[1]
    with open(hetero_graph_filepath, "rb") as f:
        hetero_graph = pickle.load(f)  # nosec
    rdkit_ligand_mol_frags = []
    for rdkit_ligand_filepath in rdkit_ligand_filepaths:
        with open(rdkit_ligand_filepath, "rb") as f:
            rdkit_ligand = pickle.load(f)  # nosec
        rdkit_ligand_mol_frags.extend(
            Chem.GetMolFrags(rdkit_ligand, asMols=True, sanitizeFrags=False)
        )
    apo_graph = hetero_graph["apo_receptor"]
    holo_graph = hetero_graph["receptor"]
    output = {
        "aatype": torch.stack([holo_graph.aatype for _ in range(2)]),
        "all_atom_positions": torch.stack(
            [
                apo_graph.all_atom_positions,
                holo_graph.all_atom_positions,
            ]
        ),
        "all_atom_mask": torch.stack([holo_graph.all_atom_mask for _ in range(2)]),
        "residue_index": torch.stack(
            [(torch.arange(len(holo_graph.aatype)) + 1) for _ in range(2)]
        ),
        "plddt": torch.stack(
            [(100.0 * torch.ones_like(holo_graph.all_atom_mask)) for _ in range(2)]
        ),
    }
    pdbs = output_to_pdb(output)
    apo_pdb_string, holo_pdb_string = pdbs[0], pdbs[1]
    apo_output_file = temp_dir_path / "apo_protein.pdb"
    holo_output_file = temp_dir_path / "holo_protein.pdb"
    apo_output_file.write_text(apo_pdb_string)
    holo_output_file.write_text(holo_pdb_string)
    return {
        "apo_protein": {
            "filepath": str(apo_output_file),
            "length": len(apo_graph.aatype),
        },
        "holo_protein": {
            "filepath": str(holo_output_file),
            "length": len(holo_graph.aatype),
        },
        "ligand": {
            "num_atoms_per_mol_frag": [mol.GetNumAtoms() for mol in rdkit_ligand_mol_frags],
        },
    }


def get_standard_aa_features():
    """Get standard amino acid features."""
    standard_pdb_filepath = os.path.join(
        Path(__file__).parent.parent.absolute(),
        "data",
        "components",
        "chemical",
        "20AA_template_peptide.pdb",
    )
    standard_aa_template_protein = pdb_filepath_to_protein(standard_pdb_filepath)
    standard_aa_template_featset = process_protein(standard_aa_template_protein)
    standard_aa_graph_featset = [
        process_mol_file(
            os.path.join(
                Path(__file__).parent.parent.absolute(),
                "data",
                "components",
                "chemical",
                f"{af_restype_1to3[aa_code]}.pdb",
            ),
            sanitize=True,
            pair_feats=True,
        )
        for aa_code in af_restypes
    ]
    return standard_aa_template_featset, standard_aa_graph_featset


def erase_holo_coordinates(
    batch: MODEL_BATCH, x: torch.Tensor, latent_converter: LatentCoordinateConverter
) -> torch.Tensor:
    """Erase the holo protein and ligand coordinates in the input tensor, leaving the apo
    coordinate untouched.

    :param batch: A batch dictionary.
    :param x: Input tensor.
    :param latent_converter: Latent converter.
    :return: Holo-erased tensor.
    """
    if batch["misc"]["protein_only"]:
        ca_lat, apo_ca_lat, cother_lat, apo_cother_lat = torch.split(
            x,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_cother_per_sample,
            ],
            dim=1,
        )
        x_erased = torch.cat(
            [
                torch.zeros_like(ca_lat),
                apo_ca_lat,
                torch.zeros_like(cother_lat),
                apo_cother_lat,
            ],
            dim=1,
        )
    else:
        (
            ca_lat,
            apo_ca_lat,
            cother_lat,
            apo_cother_lat,
            ca_lat_centroid_coords,
            apo_ca_lat_centroid_coords,
            lig_lat,
        ) = torch.split(
            x,
            [
                latent_converter._n_res_per_sample,
                latent_converter._n_res_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_cother_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_molid_per_sample,
                latent_converter._n_ligha_per_sample,
            ],
            dim=1,
        )
        x_erased = torch.cat(
            [
                torch.zeros_like(ca_lat),
                apo_ca_lat,
                torch.zeros_like(cother_lat),
                apo_cother_lat,
                torch.zeros_like(ca_lat_centroid_coords),
                apo_ca_lat_centroid_coords,
                torch.zeros_like(lig_lat),
            ],
            dim=1,
        )
    return x_erased


def prepare_batch(batch: MODEL_BATCH):
    """Prepare batch for forward pass.

    :param batch: A batch dictionary.
    """
    if "outputs" not in batch:
        batch["outputs"] = {}
    if "indexer" in batch and "gather_idx_a_cotherid" not in batch["indexer"]:
        cother_mask = batch["features"]["res_atom_mask"].bool().clone()
        cother_mask[:, 1] = False
        atom37_mask = torch.zeros_like(cother_mask, dtype=torch.long)
        atom37_mask += torch.arange(0, atom37_mask.size(0), device=atom37_mask.device).unsqueeze(
            -1
        )
        batch["indexer"]["gather_idx_a_cotherid"] = atom37_mask[cother_mask]
    if "features" in batch and "apo_res_alignment_mask" not in batch["features"]:
        batch["features"]["apo_res_alignment_mask"] = torch.ones_like(
            batch["features"]["res_atom_mask"][:, 1], dtype=torch.bool
        )
    if "num_molid" in batch["metadata"].keys() and batch["metadata"]["num_molid"] > 0:
        batch["misc"]["protein_only"] = False
    else:
        batch["misc"]["protein_only"] = True


def centralize_complex_graph(complex_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Centralize the protein and ligand coordinates in the complex graph.

    Note that the holo protein and ligand coordinates are centralized using the holo protein Ca
    atoms' centroid coordinates, whereas the apo protein coordinates are instead centralized using
    the apo Ca atoms' centroid coordinates. Afterwards, both versions of the protein coordinates
    are aligned at the origin.

    :param complex_graph: A complex graph dictionary.
    :return: Centralized complex graph dictionary.
    """
    ca_atom_centroid_coords = complex_graph["features"]["res_atom_positions"][:, 1].mean(
        dim=0, keepdim=True
    )
    complex_graph["features"]["res_atom_positions"] -= ca_atom_centroid_coords[:, None, :]
    complex_graph["features"]["sdf_coordinates"] -= ca_atom_centroid_coords
    if "apo_res_atom_positions" in complex_graph["features"]:
        apo_ca_atom_centroid_coords = complex_graph["features"]["apo_res_atom_positions"][
            :, 1
        ].mean(dim=0, keepdim=True)
        complex_graph["features"]["apo_res_atom_positions"] -= apo_ca_atom_centroid_coords[
            :, None, :
        ]
    return complex_graph


@beartype
def convert_to_molar(
    value: float,
    unit: Union[str, float],
    unit_conversions: Dict[str, Union[int, float]] = MOAD_UNIT_CONVERSION_DICT,
) -> Optional[float]:
    """Convert a binding affinity value to molar units.

    :param value: The binding affinity value as a float in original units if available or as NaN if
        not available.
    :param unit: The binding affinity unit as a string.
    :return: The binding affinity value in molar units if a conversion factor exists. None
        otherwise.
    """
    if unit in unit_conversions:
        conversion_factor = unit_conversions[unit]
        return value * conversion_factor
    else:
        return None


@beartype
def parse_pdbbind_binding_affinity_data_file(
    data_filepath: str, default_ligand_ccd_id: str = "XXX"
) -> Dict[str, Dict[str, float]]:
    """Extract binding affinities from the PDBBind database's metadata.

    :param data_filepath: Path to the PDBBind database's metadata file.
    :param default_ligand_ccd_id: The default CCD ID to use for PDBBind ligands, since PDBBind
        complexes only have a single ligand.
    :return: A dictionary mapping PDB codes to ligand CCD IDs and their corresponding binding
        affinities.
    """
    binding_affinity_scores_dict = {}
    with open(data_filepath) as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) in {8, 9}:
                pdb_code = columns[0]
                pK_value = float(columns[3])
                # NOTE: we have to handle for multi-ligands here
                if pdb_code in binding_affinity_scores_dict:
                    assert (
                        pK_value == binding_affinity_scores_dict[pdb_code][default_ligand_ccd_id]
                    ), "PDBBind complexes should only have a single ligand."
                else:
                    binding_affinity_scores_dict[pdb_code] = {default_ligand_ccd_id: pK_value}
    return binding_affinity_scores_dict


@beartype
def parse_moad_binding_affinity_data_file(data_filepath: str) -> Dict[str, Dict[str, float]]:
    """Extract binding affinities from the Binding MOAD dataset's metadata.

    :param data_filepath: Path to the Binding MOAD dataset's metadata file.
    :return: A dictionary mapping PDB codes to ligand CCD IDs and their corresponding binding
        affinities.
    """
    # read in CSV file carefully and manually install column names
    df = pd.read_csv(data_filepath, header=None, skiprows=1)
    df.columns = [
        "protein_class",
        "protein_family",
        "protein_id",
        "ligand_name",
        "ligand_validity",
        "affinity_measure",
        "=",
        "affinity_value",
        "affinity_unit",
        "smiles_string",
        "misc",
    ]
    # split up `ligand_name` column into its individual parts
    df[["ligand_ccd_id", "ligand_ccd_id_index", "ligand_het_code"]] = df["ligand_name"].str.split(
        pat=":", n=2, expand=True
    )
    df.drop(columns=["ligand_name"], inplace=True)
    # assign the corresponding PDB ID to each ligand (row) entry
    df["protein_id"].ffill(inplace=True)
    # filter for only `valid` ligands (rows) with `Ki` or `Kd` affinity measures
    df = df[df["ligand_validity"] == "valid"]
    df = df[df["affinity_measure"].isin(["Ki", "Kd"])]
    # standardize affinity values in molar units
    df["affinity_molar"] = df.apply(
        lambda row: convert_to_molar(row["affinity_value"], row["affinity_unit"]), axis=1
    )
    # normalize affinity values to pK, with a range of approximately [0, 14]
    df["pK"] = -np.log10(df["affinity_molar"])
    binding_affinity_scores_dict = (
        df.groupby("protein_id")
        .apply(
            lambda group: {
                row["ligand_ccd_id"] + ":" + str(row["ligand_ccd_id_index"]): row["pK"]
                for _, row in group.iterrows()
            }
        )
        .to_dict()
    )
    return binding_affinity_scores_dict


def min_max_normalize_array(array: np.ndarray) -> np.ndarray:
    """Min-max normalize an array.

    :param array: Array to min-max normalize.
    :return: Min-max normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)


def create_full_pdb_with_zero_coordinates(sequence: str, filename: str):
    """Create a PDB file with all atom coordinates set to zero for given protein sequences,
    including all atoms (backbone and simplified side chain). Multiple protein chains are delimited
    by "|".

    :param sequence: Protein chain sequences in single-letter code format, separated by "|".
    :param filename: Output filename for the PDB file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Backbone atoms for all amino acids
    backbone_atoms = ["N", "CA", "C", "O"]

    # Simplified representation of side chain atoms for each amino acid
    side_chain_atoms = {
        "A": ["CB"],
        "R": ["CB", "CG", "CD", "NE", "CZ"],
        "N": ["CB", "CG", "OD1"],
        "D": ["CB", "CG", "OD1"],
        "C": ["CB", "SG"],
        "E": ["CB", "CG", "CD"],
        "Q": ["CB", "CG", "CD", "OE1"],
        "G": [],
        "H": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "I": ["CB", "CG1", "CG2", "CD1"],
        "L": ["CB", "CG", "CD1", "CD2"],
        "K": ["CB", "CG", "CD", "CE", "NZ"],
        "M": ["CB", "CG", "SD", "CE"],
        "F": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "P": ["CB", "CG", "CD"],
        "S": ["CB", "OG"],
        "T": ["CB", "OG1", "CG2"],
        "W": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "Y": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "V": ["CB", "CG1", "CG2"],
    }

    with open(filename, "w") as pdb_file:
        atom_index = 1
        chain_id = "A"  # Start with chain 'A'

        for chain in sequence.split("|"):
            residue_index = 1
            for residue in chain:
                # Add backbone atoms
                for atom in backbone_atoms:
                    pdb_file.write(
                        f"ATOM  {atom_index:5d}  {atom:<3s} {af_restype_1to3.get(residue, 'UNK')} {chain_id}{residue_index:4d}    "
                        f"   0.000   0.000   0.000  1.00  0.00           C\n"
                    )
                    atom_index += 1

                # Add side chain atoms
                for atom in side_chain_atoms.get(residue, []):  # type: ignore
                    pdb_file.write(
                        f"ATOM  {atom_index:5d}  {atom:<3s} {af_restype_1to3.get(residue, 'UNK')} {chain_id}{residue_index:4d}    "
                        f"   0.000   0.000   0.000  1.00  0.00           C\n"
                    )
                    atom_index += 1

                residue_index += 1
            # Increment chain ID for next chain
            chain_id = chr(ord(chain_id) + 1)


@beartype
def parse_inference_inputs_from_dir(
    input_data_dir: Union[str, Path], pdb_ids: Optional[Set[Any]] = None
) -> List[Tuple[str, str]]:
    """Parse a data directory containing subdirectories of protein-ligand complexes and return
    corresponding SMILES strings and PDB IDs.

    :param input_data_dir: Path to the input data directory.
    :param pdb_ids: Optional set of IDs by which to filter processing.
    :return: A list of tuples each containing a SMILES string and a PDB ID.
    """
    smiles_and_pdb_id_list = []
    for pdb_name in os.listdir(input_data_dir):
        if any(substr in pdb_name.lower() for substr in ["sequence", "structure"]):
            # e.g., skip ESMFold sequence files and structure directories
            continue
        if pdb_ids is not None and pdb_name not in pdb_ids:
            # e.g., skip PoseBusters Benchmark PDBs that contain crystal contacts
            # reference: https://github.com/maabuu/posebusters/issues/26
            continue
        pdb_dir = os.path.join(input_data_dir, pdb_name)
        if os.path.isdir(pdb_dir):
            mol = None
            pdb_id = os.path.split(pdb_dir)[-1]
            # NOTE: we first try to parse `.mol2` and if necessary `.sdf` files, since e.g., PDBBind 2020's `.sdf` files do not contain chirality tags
            if os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")):
                mol = read_molecule(
                    os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2"), remove_hs=True, sanitize=True
                )
            if mol is None and os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")):
                mol = read_molecule(
                    os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf"), remove_hs=True, sanitize=True
                )
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
            # NOTE: Binding MOAD/DockGen uses `.pdb` files to store its ligands
            if mol is None and os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb")):
                mol = read_molecule(
                    os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb"), remove_hs=True, sanitize=True
                )
                if mol is None:
                    mol = read_molecule(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb"),
                        remove_hs=True,
                        sanitize=False,
                    )
            if mol is None:
                raise ValueError(f"No ligand file found for PDB ID {pdb_id}")
            mol_smiles = Chem.MolToSmiles(mol)
            if mol_smiles is None:
                raise ValueError(f"Failed to generate SMILES string for PDB ID {pdb_id}")
            smiles_and_pdb_id_list.append((mol_smiles, pdb_id))
    return smiles_and_pdb_id_list


@beartype
def create_temp_ligand_frag_files(input_sdf_file: str) -> List[str]:
    """Creates temporary SDF files for each fragment in the input SDF file."""
    # Get the fragments of the input molecule
    mol = Chem.MolFromMolFile(input_sdf_file)
    fragments = Chem.GetMolFrags(mol, asMols=True)

    temp_files = []
    for frag in fragments:
        # Create a temporary SDF file for each fragment
        temp_file = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
        temp_files.append(temp_file.name)

        # Write each fragment to the temporary SDF file
        writer = AllChem.SDWriter(temp_file.name)
        writer.write(frag)
        writer.close()

    return temp_files
