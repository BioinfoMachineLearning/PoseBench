# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from (https://github.com/maabuu/posebusters_em) and (https://github.com/luwei0917/DynamicBind)
# -------------------------------------------------------------------------------------------------------------------------------------
"""Energy minimization of a ligand in a protein pocket as used in the
PoseBusters/DynamicBind paper.

This code is based on the OpenMM user guide:
http://docs.openmm.org/latest/userguide
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import rootutils
from beartype.typing import Any, Dict, Optional
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser, Select
from omegaconf import DictConfig
from openff.toolkit.topology import Molecule
from openff.units import Quantity as openff_Quantity
from openmm import (
    CustomExternalForce,
    LangevinIntegrator,
    Platform,
    System,
    XmlSerializer,
    unit,
)
from openmm.app import Atom, ForceField, HBonds, Modeller, PDBFile, PDBxFile, Simulation
from openmm.unit import kelvin, kilojoule, molar, mole, nanometer, picosecond
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromMolFile, MolToMolFile
from rdkit.Chem.rdmolops import AddHs, RemoveHs
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from wrapt_timeout_decorator import timeout

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.utils.data_utils import combine_molecules

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FORCE_FIELDS_IMPLICIT = ["amber14-all.xml", "implicit/gbn2.xml"]
FORCE_FIELDS_EXPLICIT = ["amber14-all.xml", "amber14/tip3pfb.xml"]
OPTIMIZATION_TIMEOUT_IN_SECONDS = 600

bond_lengths = {
    ("C", "CA"): (1.5335,),
    ("C", "O"): (1.23,),
    ("CA", "CB"): (1.5365,),
    ("CA", "N"): (1.4664,),
    ("CB", "CG"): (1.5331,),
    ("CE", "SD"): (1.8117,),
    ("CG", "SD"): (1.8135,),
    ("C", "N"): (1.3359,),
    ("CB", "OG"): (1.4144,),
    ("CB", "SG"): (1.8146,),
    ("CD", "CG"): (1.5308,),
    ("CD", "N"): (1.457,),
    ("CB", "CG1"): (1.5365,),
    ("CB", "CG2"): (1.5341,),
    ("CD1", "CG1"): (1.5334,),
    ("CG", "OD1"): (1.2477,),
    ("CG", "OD2"): (1.2507,),
    ("CD", "CE"): (1.5312,),
    ("CE", "NZ"): (1.4831,),
    ("CD", "NE"): (1.4748,),
    ("CZ", "NE"): (1.3205,),
    ("CZ", "NH1"): (1.3041,),
    ("CZ", "NH2"): (1.3018,),
    ("CB", "OG1"): (1.4086,),
    ("CD1", "CG"): (1.5223,),
    ("CD2", "CG"): (1.4589,),
    ("CD1", "CE1"): (1.406,),
    ("CD2", "CE2"): (1.4063,),
    ("CE1", "CZ"): (1.406,),
    ("CE2", "CZ"): (1.4059,),
    ("CD", "NE2"): (1.3141,),
    ("CD", "OE1"): (1.2493,),
    ("CD", "OE2"): (1.2524,),
    ("CG", "ND2"): (1.3112,),
    ("CD1", "NE1"): (1.3804,),
    ("CD2", "CE3"): (1.4115,),
    ("CE2", "CZ2"): (1.401,),
    ("CE2", "NE1"): (1.3774,),
    ("CE3", "CZ3"): (1.4076,),
    ("CH2", "CZ2"): (1.4046,),
    ("CH2", "CZ3"): (1.4062,),
    ("CD2", "NE2"): (1.3971,),
    ("CE1", "ND1"): (1.3424,),
    ("CE1", "NE2"): (1.3406,),
    ("CG", "ND1"): (1.3768,),
    ("CZ", "OH"): (1.3576,),
    ("C", "OXT"): (1.2491,),
    ("SG", "SG"): (2.0324,),
}


chi_atoms = dict(
    chi1=dict(
        ARG=["N", "CA", "CB", "CG"],
        ASN=["N", "CA", "CB", "CG"],
        ASP=["N", "CA", "CB", "CG"],
        CYS=["N", "CA", "CB", "SG"],
        GLN=["N", "CA", "CB", "CG"],
        GLU=["N", "CA", "CB", "CG"],
        HIS=["N", "CA", "CB", "CG"],
        ILE=["N", "CA", "CB", "CG1"],
        LEU=["N", "CA", "CB", "CG"],
        LYS=["N", "CA", "CB", "CG"],
        MET=["N", "CA", "CB", "CG"],
        PHE=["N", "CA", "CB", "CG"],
        PRO=["N", "CA", "CB", "CG"],
        SER=["N", "CA", "CB", "OG"],
        THR=["N", "CA", "CB", "OG1"],
        TRP=["N", "CA", "CB", "CG"],
        TYR=["N", "CA", "CB", "CG"],
        VAL=["N", "CA", "CB", "CG1"],
    ),
    altchi1=dict(
        VAL=["N", "CA", "CB", "CG2"],
    ),
    chi2=dict(
        ARG=["CA", "CB", "CG", "CD"],
        ASN=["CA", "CB", "CG", "OD1"],
        ASP=["CA", "CB", "CG", "OD1"],
        GLN=["CA", "CB", "CG", "CD"],
        GLU=["CA", "CB", "CG", "CD"],
        HIS=["CA", "CB", "CG", "ND1"],
        ILE=["CA", "CB", "CG1", "CD1"],
        LEU=["CA", "CB", "CG", "CD1"],
        LYS=["CA", "CB", "CG", "CD"],
        MET=["CA", "CB", "CG", "SD"],
        PHE=["CA", "CB", "CG", "CD1"],
        PRO=["CA", "CB", "CG", "CD"],
        TRP=["CA", "CB", "CG", "CD1"],
        TYR=["CA", "CB", "CG", "CD1"],
    ),
    altchi2=dict(
        ASP=["CA", "CB", "CG", "OD2"],
        LEU=["CA", "CB", "CG", "CD2"],
        PHE=["CA", "CB", "CG", "CD2"],
        TYR=["CA", "CB", "CG", "CD2"],
    ),
    chi3=dict(
        ARG=["CB", "CG", "CD", "NE"],
        GLN=["CB", "CG", "CD", "OE1"],
        GLU=["CB", "CG", "CD", "OE1"],
        LYS=["CB", "CG", "CD", "CE"],
        MET=["CB", "CG", "SD", "CE"],
    ),
    chi4=dict(
        ARG=["CG", "CD", "NE", "CZ"],
        LYS=["CG", "CD", "CE", "NZ"],
    ),
    chi5=dict(
        ARG=["CD", "NE", "CZ", "NH1"],
    ),
)
chi_names = ["chi%s" % i for i in range(1, 6)]

# NOTE: a tuple containing left bond atoms and right bond atoms -> a list of atoms that should rotate with the bond
chi1_bond_dict = {
    "ALA": None,
    "ARG": ("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN": ("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP": ("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS": ("CA", "CB", ["SG"]),
    "GLN": ("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU": ("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY": None,
    "HIS": ("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE": ("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU": ("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS": ("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET": ("CA", "CB", ["CG", "SD", "CE"]),
    "PHE": ("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO": ("CA", "CB", ["CG", "CD"]),
    "SER": ("CA", "CB", ["OG"]),
    "THR": ("CA", "CB", ["CG2", "OG1"]),
    "TRP": ("CA", "CB", ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR": ("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL": ("CA", "CB", ["CG1", "CG2"]),
}

chi2_bond_dict = {
    "ALA": None,
    "ARG": ("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN": ("CB", "CG", ["ND2", "OD1"]),
    "ASP": ("CB", "CG", ["OD1", "OD2"]),
    "CYS": None,
    "GLN": ("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU": ("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY": None,
    "HIS": ("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE": ("CB", "CG1", ["CD1"]),
    "LEU": ("CB", "CG", ["CD1", "CD2"]),
    "LYS": ("CB", "CG", ["CD", "CE", "NZ"]),
    "MET": ("CB", "CG", ["SD", "CE"]),
    "PHE": ("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO": ("CB", "CG", ["CD"]),
    "SER": None,
    "THR": None,
    "TRP": ("CB", "CG", ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR": ("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL": None,
}


chi3_bond_dict = {
    "ALA": None,
    "ARG": ("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN": None,
    "ASP": None,
    "CYS": None,
    "GLN": ("CG", "CD", ["NE2", "OE1"]),
    "GLU": ("CG", "CD", ["OE1", "OE2"]),
    "GLY": None,
    "HIS": None,
    "ILE": None,
    "LEU": None,
    "LYS": ("CG", "CD", ["CE", "NZ"]),
    "MET": ("CG", "SD", ["CE"]),
    "PHE": None,
    "PRO": None,
    "SER": None,
    "THR": None,
    "TRP": None,
    "TYR": None,
    "VAL": None,
}

chi4_bond_dict = {
    "ARG": ("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS": ("CD", "CE", ["NZ"]),
}

chi5_bond_dict = {
    "ARG": ("NE", "CZ", ["NH1", "NH2"]),
}

chi_dict = {
    1: chi1_bond_dict,
    2: chi2_bond_dict,
    3: chi3_bond_dict,
    4: chi4_bond_dict,
    5: chi5_bond_dict,
}


class NoHydrogen(Select):
    """Select class that filters out hydrogen atoms."""

    def accept_atom(self, atom: Any) -> bool:
        """Return True if the atom element is not H or D."""
        if atom.element == "H" or atom.element == "D":
            return False
        return True


def generate_conformer(mol: Chem.Mol):
    """Generate an RDKit conformer for a molecule.

    :param mol: The molecule for which to generate an RDKit conformer.
    """
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        logger.warning(
            "RDKit coordinates could not be generated without using random coordinates. Using random coordinates now."
        )
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)


def remove_hydrogens_from_pdb(input_pdb_filepath: str, output_pdb_filepath: str):
    """Remove hydrogen atoms from a PDB file.

    :param input_pdb_filepath: Path to the input PDB file.
    :param output_pdb_filepath: Path to the output PDB file.
    """
    parser = (
        MMCIFParser(QUIET=True)
        if os.path.splitext(input_pdb_filepath)[-1] == ".cif"
        else PDBParser(QUIET=True)
    )
    s = parser.get_structure("x", input_pdb_filepath)
    io = MMCIFIO() if os.path.splitext(output_pdb_filepath)[-1] == ".cif" else PDBIO()
    io.set_structure(s)
    io.save(output_pdb_filepath, select=NoHydrogen())


def remove_mol_hydrogens_and_reorder(mol: Chem.Mol) -> Chem.Mol:
    """Remove hydrogens from a molecule and re-order the atoms.

    :param mol: The molecule to process.
    :return: The processed molecule.
    """
    mol = Chem.RemoveAllHs(mol)
    _ = Chem.MolToSmiles(mol)
    mol_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, mol_order)
    return mol


def will_restrain(atom: Atom, residue_set: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set.

    :param atom: Atom to check.
    :param residue_set: Residue set to use for checking.
    """
    if residue_set == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif residue_set == "c_alpha":
        return atom.name == "CA"
    else:
        raise ValueError(f"Unsupported residue set: {residue_set}")


def get_all_protein_atoms(input_pdb_filepath: str) -> list[Any]:
    """Returns a list of all protein atoms in the PDB file.

    :param input_pdb_filepath: Path to the input PDB file.
    :return: A list of all protein atoms in the PDB file.
    """
    if os.path.splitext(input_pdb_filepath)[-1] == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    s = parser.get_structure(input_pdb_filepath, input_pdb_filepath)
    all_atoms = list(s.get_atoms())
    return all_atoms


def randomly_rotate_protein_side_chains(
    structure: Any,
    violation_residue_idx: list[int],
    min_angle: float = 0.6,
    max_angle: float = np.pi,
    num_chi_groups: int = 5,
    eps: float = 1e-6,
) -> Any:
    """Randomly rotate side chains of the protein residues with local geometry
    violations.

    :param structure: Structure to rotate.
    :param violation_residue_idx: List of residue indices with local
        geometry violations.
    :param min_angle: Minimum angle to rotate.
    :param max_angle: Maximum angle to rotate.
    :param num_chi_groups: Number of side chain groups.
    :param eps: Epsilon value.
    """
    for res in structure.get_residues():
        if res.get_full_id()[3][1] not in violation_residue_idx:
            continue
        chi_mask = [0] * 5
        if res.id[0] != " ":
            continue
        resname = res.resname
        if resname in ("ALA", "GLY"):
            continue
        for x, chi in enumerate(chi_names):
            chi_res = chi_atoms[chi]
            if resname not in chi_res:
                continue
            chi_mask[x] = 1
        pred_chi = np.random.uniform(min_angle, max_angle, 5)
        for i in range(num_chi_groups):
            if chi_mask[i] == 0:
                continue
            try:
                chi_bond_dict = chi_dict[i + 1]
            except KeyError:
                logger.warning(f"Dictionary for `chi{i + 1}_bond_dict` not found.")
                continue
            atom1, atom2, rotate_atom_list = chi_bond_dict[resname]
            if (atom1 not in res) or (atom2 not in res):
                continue
            atom1_coord = res[atom1].coord
            atom2_coord = res[atom2].coord
            rot_vec = atom2_coord - atom1_coord
            rot_vec = pred_chi[i] * (rot_vec) / (np.linalg.norm(rot_vec) + eps)
            rot_mat = R.from_rotvec(rot_vec).as_matrix()

            for rotate_atom in rotate_atom_list:
                if rotate_atom not in res:
                    continue
                new_coord = (
                    np.matmul(res[rotate_atom].coord - res[atom1].coord, rot_mat.T)
                    + res[atom1].coord
                )
                res[rotate_atom].set_coord(new_coord)


def save_biopython_structure_to_pdb(
    structure: Any, output_pdb_filepath: str, ca_only: bool = False
):
    """Saves a BioPython protein `Structure` objects to a PDB file.

    :param structure: Structure to save.
    :param output_pdb_filepath: Path to the input PDB file.
    :param ca_only: Whether to save only the CA atoms or not.
    """
    if os.path.splitext(output_pdb_filepath)[-1] == ".pdb":
        io = PDBIO()
    elif os.path.splitext(output_pdb_filepath)[-1] == ".cif":
        io = MMCIFIO()
    else:
        raise ValueError(
            f"Unsupported file format for saving modified protein structures following relaxation: {os.path.splitext(output_pdb_filepath)[-1]}"
        )

    class MySelect(Select):
        def accept_atom(self, atom):
            if atom.get_name() == "CA":
                return True
            else:
                return False

    class RemoveHs(Select):
        def accept_atom(self, atom):
            if atom.element != "H":
                return True
            else:
                return False

    io.set_structure(structure)
    if ca_only:
        io.save(output_pdb_filepath, MySelect())
    else:
        io.save(output_pdb_filepath, RemoveHs())


def compute_protein_local_geometry_violations(input_pdb_filepath: str) -> tuple[int, list[int]]:
    """Computes the number of protein local geometry violations in the PDB
    file.

    :param input_pdb_filepath: Path to the latest protein input PDB
        file.
    :return: A tuple containing the number of protein local geometry
        violations and the list of residue indices with associated
        violations.
    """
    protein_all_atoms = get_all_protein_atoms(input_pdb_filepath)
    protein_pdb = (
        PDBxFile(input_pdb_filepath)
        if os.path.splitext(input_pdb_filepath)[-1] == ".cif"
        else PDBFile(input_pdb_filepath)
    )
    bond_index_table = [
        [bond.atom1.index, bond.atom2.index] for bond in protein_pdb.topology.bonds()
    ]
    violation_residue_idx = []
    deviation_greater_than_cutoff = 0
    for bond_index in bond_index_table:
        i, j = bond_index
        atom1 = protein_all_atoms[i]
        atom2 = protein_all_atoms[j]
        if atom1.id == "OXT" or atom2.id == "OXT":
            continue
        dis = atom1 - atom2
        deviation = abs(bond_lengths[(atom1.name, atom2.name)][0] - dis)
        violation = deviation > 0.3
        deviation_greater_than_cutoff += violation
        if violation:
            violation_residue_idx.extend([atom1.get_full_id()[3][1], atom2.get_full_id()[3][1]])
    return deviation_greater_than_cutoff, violation_residue_idx


def get_all_ligand_non_hydrogen_atoms(
    input_sdf_filepath: str | None,
    ref_input_sdf_filepath: str | None,
    input_mol: Chem.Mol | None = None,
    ref_mol: Chem.Mol | None = None,
) -> tuple[np.ndarray, list[list[bool]]]:
    """Get all non-hydrogen atoms in the ligand SDF file.

    :param input_sdf_filepath: Path to the latest ligand input SDF file.
    :param ref_input_sdf_filepath: Path to the reference (e.g.,
        original) ligand input SDF file.
    :param input_mol: The optional RDKit molecule to directly use for
        the input ligand.
    :param ref_mol: The optional RDKit molecule to directly use for the
        reference ligand.
    :return: A tuple containing the NumPy ligand atom coordinates and
        the list-of-lists ligand adjacency matrix.
    """
    try:
        if input_mol is not None:
            mol = input_mol
        else:
            mol = Chem.MolFromMolFile(input_sdf_filepath)
            if mol is None:
                raise Exception(
                    f"In `get_all_ligand_non_hydrogen_atoms()`, molecule could not be loaded from SDF file: {input_sdf_filepath}"
                )
    except Exception as e:
        if ref_mol is not None:
            mol = ref_mol
        else:
            mol = Chem.MolFromMolFile(ref_input_sdf_filepath)
        mol.RemoveAllConformers()
        mol = Chem.AddHs(mol)
        generate_conformer(mol)
    mol = Chem.RemoveAllHs(mol)
    Chem.MolToSmiles(mol)

    smiles_atom_output_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, smiles_atom_output_order)
    mol_atoms = list(mol.GetAtoms())
    mol_atoms = [a.GetSymbol() for a in mol_atoms]

    c = mol.GetConformer()
    mol_atom_coords = c.GetPositions()
    return mol_atom_coords, Chem.GetAdjacencyMatrix(mol).astype(bool)


def compute_ligand_local_geometry_violations(
    ref_input_sdf_filepath: str,
    input_sdf_filepath: str | None,
) -> int:
    """Computes the number of ligand local geometry violations in the SDF file.

    :param ref_input_sdf_filepath: Path to the reference (e.g.,
        original) ligand input SDF file.
    :param input_sdf_filepath: Path to the optional latest ligand input
        SDF file.
    :return: The cumulative number of ligand local geometry violations.
    """
    deviations_greater_than_cutoff = []
    sdf_filepath = input_sdf_filepath if input_sdf_filepath is not None else ref_input_sdf_filepath
    ligand_frags = Chem.GetMolFrags(Chem.MolFromMolFile(sdf_filepath), asMols=True)
    for ligand_frag in ligand_frags:
        ref_ligand_atom_coords, local_geometry_mask = get_all_ligand_non_hydrogen_atoms(
            ref_input_sdf_filepath,
            input_sdf_filepath,
            ref_mol=copy.deepcopy(ligand_frag),
        )
        ligand_atom_coords, _ = get_all_ligand_non_hydrogen_atoms(
            input_sdf_filepath,
            ref_input_sdf_filepath,
            input_mol=copy.deepcopy(ligand_frag),
        )
        assert len(ref_ligand_atom_coords) == len(
            ligand_atom_coords
        ), "The number of atoms in the reference and input ligands are not the same."
        ref_pair_distances = cdist(ref_ligand_atom_coords, ref_ligand_atom_coords)
        pair_distances = cdist(ligand_atom_coords, ligand_atom_coords)
        deviation_greater_than_cutoff = (
            abs(ref_pair_distances[local_geometry_mask] - pair_distances[local_geometry_mask])
            > 0.3
        ).sum()
        deviations_greater_than_cutoff.append(deviation_greater_than_cutoff)
    return sum(deviations_greater_than_cutoff)


def load_molecule(mol_path: Path, **kwargs) -> Molecule:
    """Load a molecule from a file.

    :param mol_path: Path to the molecule file.
    :param kwargs: Additional keyword arguments to pass to the
        Molecule.from_file method.
    :return: The loaded molecule.
    """
    mols = Molecule.from_file(str(mol_path), file_format="sdf", **kwargs)
    molecule = mols[0] if isinstance(mols, list) else mols
    molecule.name = str(mol_path)
    return molecule


def prep_ligand(
    ligand_file: Path,
    temp_file: Path,
    relax_protein: bool,
    allow_undefined_stereo: bool = True,
    mol: Chem.Mol | None = None,
) -> tuple[Molecule, str | None]:
    """Prepare a ligand for use in OpenMM.

    :param ligand_file: Path to the ligand file.
    :param temp_file: Path to the temporary file to save the prepared
        ligand to.
    :param relax_protein: Whether or not to relax the protein.
    :param allow_undefined_stereo: Whether or not to allow undefined
        stereochemistry.
    :param mol: The optional RDKit molecule to directly use for the
        ligand.
    :return: The prepared ligand.
    """
    try:
        mol_smiles = None
        if not temp_file.exists():
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            if relax_protein:
                if mol is None:
                    mol = MolFromMolFile(str(ligand_file))

                mol_with_hs = AddHs(mol, addCoords=True)
                mol_smiles = Chem.MolToSmiles(mol_with_hs)
                mol_smiles_atom_output_order = list(
                    mol_with_hs.GetPropsAsDict(includePrivate=True, includeComputed=True)[
                        "_smilesAtomOutputOrder"
                    ]
                )
                mol_with_hs = Chem.RenumberAtoms(mol_with_hs, mol_smiles_atom_output_order)
                mol = Molecule.from_smiles(
                    mol_smiles, allow_undefined_stereo=allow_undefined_stereo
                )

                MolToMolFile(mol.to_rdkit(), str(temp_file))
            else:
                if mol is None:
                    mol = MolFromMolFile(str(ligand_file), sanitize=True)
                mol = AddHs(mol, addCoords=True)

                MolToMolFile(mol, str(temp_file))

        return (
            load_molecule(temp_file, allow_undefined_stereo=allow_undefined_stereo),
            mol_smiles,
        )

    except Exception as e:
        logger.error(f"Error preparing ligand: {e}")
        temp_file.unlink(missing_ok=True)
        raise e


def prep_protein(
    protein_file: Path,
    temp_file: Path,
    relax_protein: bool,
    remove_initial_protein_hydrogens: bool,
    add_solvent: bool = False,
) -> PDBFile:
    """Prepare a protein for use in OpenMM.

    :param protein_file: Path to the protein file.
    :param temp_file: Path to the temporary file to save the prepared
        protein to.
    :param relax_protein: Whether or not to relax the protein.
    :param remove_initial_protein_hydrogens: Whether or not to remove
        the initial protein hydrogens.
    :param add_solvent: Whether or not to add solvent to the protein.
    :return: The prepared protein.
    """
    try:
        if not temp_file.exists():
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            if relax_protein or remove_initial_protein_hydrogens:
                remove_hydrogens_from_pdb(str(protein_file), str(temp_file))
            else:
                shutil.copy(protein_file, temp_file)
            fixer = PDBFixer(str(temp_file))

            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(keepWater=False)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            if add_solvent:
                fixer.addSolvent(fixer.topology.getUnitCellDimensions())

            with open(temp_file, "w", encoding="utf-8") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=relax_protein)

        return PDBFile(str(temp_file))

    except Exception as e:
        logger.error(f"Error preparing protein: {e}")
        temp_file.unlink(missing_ok=True)
        raise e


def add_ligand_to_complex(
    modeller: Modeller,
    ligand: Molecule,
) -> Modeller:
    """Add a ligand to a protein-ligand complex.

    :param modeller: The Modeller object for the protein-ligand complex.
    :param ligand: The prepared ligand.
    :return: The updated Modeller object for the protein-ligand complex.
    """
    topology = ligand.to_topology().to_openmm()
    positions = ligand.conformers[0].magnitude * unit.angstrom
    modeller.add(topology, positions)
    return modeller


def deserialize(path: str) -> System:
    """Deserialize an OpenMM system from a file.

    :param path: Path to the file to deserialize the system from.
    :return: The deserialized system.
    """
    with open(path, encoding="utf-8") as file:
        system = XmlSerializer.deserialize(file.read())
    return system


def serialize(system: System, path: str) -> str:
    """Serialize an OpenMM system to a file.

    :param system: The system to serialize.
    :param path: Path to the file to serialize the system to.
    :return: The path to the serialized system.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(XmlSerializer.serialize(system))
    return path


def get_fastest_platform() -> Platform:
    """Get the fastest available OpenMM platform.

    :return: The fastest available OpenMM platform.
    """
    platforms = [Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())]
    speeds = [platform.getSpeed() for platform in platforms]
    platform = platforms[speeds.index(max(speeds))]
    return platform


def generate_system(
    modeller: Modeller,
    ligands: Molecule,
    num_particles_protein: int,
    num_particles_total: int,
    name: str,
    force_fields: list[str],
    relax_protein: bool,
    cache_dir: Path | None = None,
) -> System:
    """Generate an OpenMM system for the protein-ligand complex.

    :param modeller: The Modeller object for the protein-ligand complex.
    :param ligands: The prepared ligands.
    :param num_particles_protein: The number of particles in the
        protein.
    :param num_particles_total: The total number of particles in the
        complex.
    :param name: The name of the system.
    :param force_fields: The force fields to use.
    :param relax_protein: Whether or not to relax the protein.
    :param cache_dir: The directory to cache the system in.
    :return: The generated OpenMM system.
    """
    # try load from cache
    if cache_dir is not None:
        system_path = cache_dir / f"{name}_system.xml"
        if system_path.exists():
            system = deserialize(str(system_path))
            # NOTE: perform rudimentary check to see if system is correct
            if system.getNumParticles() == num_particles_total:
                logger.info(f"Loaded system from cache: {system_path}")
                return system

    logger.info(f"Generating system for {name}")

    if relax_protein:
        # set up forcefield
        forcefield_kwargs = {
            "constraints": HBonds,
            # process iron atoms (reference: http://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem)
            "residueTemplates": {
                res: "FE"
                for res in modeller.topology.residues()
                if res.name == "FE" or all(a.element._symbol.upper() == "FE" for a in res._atoms)
            },
        }
        system_generator = SystemGenerator(
            forcefields=["amber/ff14SB.xml"],
            small_molecule_forcefield="gaff-2.11",
            forcefield_kwargs=forcefield_kwargs,
        )
        # set up system
        system = system_generator.create_system(
            modeller.topology,
            molecules=ligands,
        )
    else:
        # set up forcefield
        forcefield = ForceField(*force_fields)
        smirnoff = SMIRNOFFTemplateGenerator(molecules=ligands)
        forcefield.registerTemplateGenerator(smirnoff.generator)

        # set up system
        system = forcefield.createSystem(
            modeller.topology,
            residueTemplates={
                res: "FE"
                for res in modeller.topology.residues()
                if res.name == "FE" or all(a.element._symbol.upper() == "FE" for a in res._atoms)
            },
        )
        for i in range(num_particles_protein):
            system.setParticleMass(i, 0.0)

    # save to cache
    if cache_dir is not None:
        system_path = cache_dir / f"{name}_system.xml"
        serialize(system, str(system_path))

    return system


def setup_simulation(
    modeller: Modeller,
    system: System,
    platform: Platform,
    platform_properties: dict[str, Any],
    temperature: float = 300.0 * kelvin,
    friction_coeff: float = 1.0 / picosecond,
    step_size: float = 0.002 * picosecond,
) -> Simulation:
    """Set up an OpenMM simulation for the protein-ligand complex.

    :param modeller: The Modeller object for the protein-ligand complex.
    :param system: The OpenMM system for the protein-ligand complex.
    :param platform: The OpenMM platform to use.
    :param platform_properties: The properties to use with the OpenMM
        platform.
    :param temperature: The temperature to use with a
        LangevinIntegrator.
    :param friction_coeff: The friction coefficient to use with a
        LangevinIntegrator.
    :param step_size: The step size to use with a LangevinIntegrator.
    :return: The setup OpenMM simulation.
    """
    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)
    simulation = Simulation(modeller.topology, system, integrator, platform, platform_properties)
    simulation.context.setPositions(modeller.positions)
    return simulation


def save_with_rdkit(
    molecule: Molecule,
    file_path: Path,
    conformer_index: int = 0,
    name: str | None = None,
    mol: Chem.Mol | None = None,
):
    """Save a molecule to a file using RDKit.

    :param molecule: The molecule to save.
    :param file_path: The path to save the molecule to.
    :param conformer_index: The index of the conformer to save.
    :param name: The name to give the molecule.
    :param mol: The optional RDKit molecule to directly use for the
        ligand.
    """
    # NOTE: use RDKit because the `.to_file()` method does not allow picking conformation and only writes first one
    if mol is None:
        mol = molecule.to_rdkit()
    mol = RemoveHs(mol)
    if name is not None:
        mol.SetProp("_Name", name)
    MolToMolFile(mol, str(file_path), confId=conformer_index)


@timeout(OPTIMIZATION_TIMEOUT_IN_SECONDS, use_signals=False)
def optimize_ligand_in_pocket(
    protein_file: Path,
    ligand_file: Path,
    output_file: Path | None = None,
    protein_output_file: Path | None = None,
    complex_output_file: Path | None = None,
    protein_gap_mask: str | None = None,
    ligand_stiffness: float = 3000.0,
    protein_stiffness: float = 1000.0,
    protein_residue_set: str = "non_hydrogen",
    tolerance: float = 0.01,
    allow_undefined_stereo: bool = True,
    prep_only: bool = False,
    temp_dir: Path = Path("."),
    name: str | None = None,
    add_solvent: bool = False,
    relax_protein: bool = False,
    remove_initial_protein_hydrogens: bool = False,
    assign_each_ligand_unique_force: bool = False,
    model_ions: bool = True,
    cache_files: bool = True,
    assign_partial_charges_manually: bool = False,
    report_initial_energy_only: bool = False,
    platform_name: str = "fastest",
    cuda_device_index: int = 0,
    max_iterations: int = 0,
    energy: float = unit.kilocalories_per_mole,
    length: float = unit.angstroms,
) -> dict[str, Any]:
    """Optimize a ligand in a protein pocket using OpenMM.

    :param protein_file: Path to the protein file.
    :param ligand_file: Path to the ligand file.
    :param output_file: Path to save the optimized ligand to.
    :param protein_output_file: Path to save the optimized protein to.
    :param complex_output_file: Path to save the optimized protein-ligand complex to.
    :param protein_gap_mask: Mask of the protein regions in the PDB file for which to skip relaxation, represented as a sequence-length string.
    :param ligand_stiffness: Stiffness of the ligand chains.
    :param protein_stiffness: Stiffness of the protein chains.
    :param protein_residue_set: Residue set to use for relaxation. Can be either `non_hydrogen` or `c_alpha`.
    :param tolerance: The tolerance to use for energy minimization.
    :param allow_undefined_stereo: Whether or not to allow undefined stereochemistry.
    :param prep_only: Whether or not to only prepare the ligand and protein.
    :param temp_dir: The temporary directory to use.
    :param name: The name of the system.
    :param add_solvent: Whether or not to add solvent to the protein.
    :param relax_protein: Whether or not to relax the protein.
    :param remove_initial_protein_hydrogens: Whether or not to remove the initial protein hydrogens.
    :param assign_each_ligand_unique_force: Whether to assign each ligand a unique force constant.
    :param model_ions: Whether or not to model ions.
    :param cache_files: Whether or not to cache the prepared files.
    :param assign_partial_charges_manually: Whether or not to assign partial charges manually.
    :param report_initial_energy_only: Whether or not to report the initial energy only.
    :param platform_name: The name of the OpenMM platform to use.
    :param cuda_device_index: The index of the CUDA device to use.
    :param max_iterations: The maximum number of iterations to use for energy minimization.
    :param energy: When relaxing the protein, the unit of energy to use.
    :param length: When relaxing the protein, the unit of length to use.
    :return: A dictionary containing the energy before and after minimization, and the optimized
        ligand.
    """
    name = protein_file.stem if name is None else name

    protein_cache = temp_dir / f"{name}_prepped_protein.pdb"
    protein_complex = prep_protein(
        protein_file=protein_file,
        temp_file=protein_cache,
        relax_protein=relax_protein,
        remove_initial_protein_hydrogens=remove_initial_protein_hydrogens,
        add_solvent=False,
    )
    if not cache_files:
        protein_cache.unlink(missing_ok=True)

    # construct initial complex
    modeller = Modeller(protein_complex.topology, protein_complex.positions)

    # parse all input ligands
    ligand_mol = MolFromMolFile(str(ligand_file), sanitize=not relax_protein)
    ligand_frags = Chem.GetMolFrags(ligand_mol, asMols=True)
    ligands, ligands_smiles = [], []
    for ligand_index, ligand in enumerate(ligand_frags, start=1):
        ligand_cache = temp_dir / f"{name}_prepped_ligand_{ligand_index}.sdf"
        ligand, ligand_smiles = prep_ligand(
            ligand_file=ligand_file,
            temp_file=ligand_cache,
            relax_protein=relax_protein,
            allow_undefined_stereo=allow_undefined_stereo,
            mol=ligand,
        )
        if not cache_files:
            ligand_cache.unlink(missing_ok=True)

        if relax_protein or assign_partial_charges_manually:
            try:
                ligand.assign_partial_charges(partial_charge_method="mmff94")
            except Exception as e:
                ligand.assign_partial_charges(partial_charge_method="zeros")
                logger.warning(f"Unable to assign partial charges to {ligand_file} due to: {e}")

        modeller = add_ligand_to_complex(modeller, ligand)
        ligands.append(ligand)
        ligands_smiles.append(ligand_smiles)

    dimensions = protein_complex.getTopology().getUnitCellDimensions()
    if add_solvent and dimensions is not None:
        force_fields = FORCE_FIELDS_EXPLICIT
        modeller.addSolvent(
            dimensions, model="tip3p", padding=1.0 * nanometer, ionicStrength=0.15 * molar
        )
    else:
        force_fields = (
            FORCE_FIELDS_EXPLICIT + ["amber14/GLYCAM_06j-1.xml"]
            if model_ions
            else FORCE_FIELDS_IMPLICIT
        )

    num_particles_protein = len(protein_complex.positions)
    num_particles_ligand = sum([len(ligand.conformers[0].magnitude) for ligand in ligands])
    num_particles_total = len(modeller.getPositions())
    assert (
        num_particles_ligand == num_particles_total - num_particles_protein
    ), "Number of ligand particles does not match expectation."

    # generate system
    system = generate_system(
        modeller=modeller,
        ligands=ligands,
        force_fields=force_fields,
        num_particles_protein=num_particles_protein,
        num_particles_total=num_particles_total,
        cache_dir=temp_dir,
        name=name,
        relax_protein=relax_protein,
    )
    if prep_only:
        return {}

    platform = (
        get_fastest_platform()
        if platform_name == "fastest"
        else Platform.getPlatformByName(platform_name)
    )
    platform_properties = {}
    if platform.getName() == "CUDA":
        platform_properties["DeviceIndex"] = str(cuda_device_index)
        logger.info(f"Using platform: {platform.getName()}:{cuda_device_index}")
    else:
        logger.info(f"Using platform: {platform.getName()}")

    if relax_protein:
        # collect metadata
        if protein_gap_mask is None:
            protein_gap_mask = "0" * protein_complex.topology.getNumResidues()
        num_residues = len(protein_gap_mask)
        reference_modeller = modeller

        # add protein constraint
        force = CustomExternalForce("0.5 * protein_k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("protein_k", protein_stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for i, atom in enumerate(reference_modeller.topology.atoms()):
            if atom.residue.index < num_residues and protein_gap_mask[atom.residue.index] == "1":
                continue
            if atom.residue.index >= num_residues:
                continue
            if will_restrain(atom, protein_residue_set):
                force.addParticle(i, reference_modeller.positions[i])
        system.addForce(force)

        # add ligand constraint(s)
        if assign_each_ligand_unique_force:
            ligand_index = 0
            num_ligand_atoms_observed = 0
            num_atoms_in_ligand = len(ligands[0].atoms)
            ligand_force = CustomExternalForce(
                f"0.5 * ligand_k_{ligand_index} * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
            )
            ligand_force.addGlobalParameter(f"ligand_k_{ligand_index}", ligand_stiffness)
            for p in ["x0", "y0", "z0"]:
                ligand_force.addPerParticleParameter(p)
            for i, atom in enumerate(reference_modeller.topology.atoms()):
                if atom.residue.index < num_residues:
                    continue
                if num_ligand_atoms_observed == num_atoms_in_ligand:
                    # aggregate intermediate ligand forces
                    ligand_index += 1
                    if ligand_index < len(ligands):
                        system.addForce(ligand_force)
                        num_ligand_atoms_observed = 0
                        num_atoms_in_ligand = len(ligands[ligand_index].atoms)
                        ligand_force = CustomExternalForce(
                            f"0.5 * ligand_k_{ligand_index} * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
                        )
                        ligand_force.addGlobalParameter(
                            f"ligand_k_{ligand_index}", ligand_stiffness
                        )
                        for p in ["x0", "y0", "z0"]:
                            ligand_force.addPerParticleParameter(p)
                    else:
                        break
                if will_restrain(atom, protein_residue_set):
                    ligand_force.addParticle(i, reference_modeller.positions[i])
                    num_ligand_atoms_observed += 1
            if ligand_index > 0:
                # add initial (and/or final) ligand force
                system.addForce(ligand_force)
        else:
            ligand_force = CustomExternalForce("0.5 * ligand_k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
            ligand_force.addGlobalParameter("ligand_k", ligand_stiffness)
            for p in ["x0", "y0", "z0"]:
                ligand_force.addPerParticleParameter(p)
            for i, atom in enumerate(reference_modeller.topology.atoms()):
                if atom.residue.index < num_residues:
                    continue
                if will_restrain(atom, protein_residue_set):
                    ligand_force.addParticle(i, reference_modeller.positions[i])
            system.addForce(ligand_force)

    simulation = setup_simulation(
        modeller,
        system,
        platform,
        platform_properties,
        temperature=0.0 if relax_protein else 300.0 * kelvin,
        friction_coeff=0.01 if relax_protein else 1.0 / picosecond,
        step_size=0.0 if relax_protein else 0.002 * picosecond,
    )

    results_dict = {}
    if relax_protein:
        # save initial state
        state_before = simulation.context.getState(getEnergy=True, getPositions=True)
        results_dict["e_init"] = state_before.getPotentialEnergy().value_in_unit(energy)
        results_dict["pos_init"] = state_before.getPositions(asNumpy=True).value_in_unit(length)
        if report_initial_energy_only:
            return results_dict

        # minimize
        logger.info(f"Minimizing {name}")
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)

        # save final state
        logger.info(f"Saving {name}")
        state_after = simulation.context.getState(getEnergy=True, getPositions=True)
        results_dict["e_final"] = state_after.getPotentialEnergy().value_in_unit(energy)
        results_dict["pos_final"] = state_after.getPositions(asNumpy=True).value_in_unit(length)

        # record protein PDB file
        if protein_file.suffix == ".pdb":
            with open(protein_output_file, "w") as f:
                PDBFile.writeFile(
                    protein_complex.topology,
                    results_dict["pos_final"][:num_particles_protein],
                    f,
                    keepIds=True,
                )
            remove_hydrogens_from_pdb(str(protein_output_file), str(protein_output_file))
            # record complex PDB file
            if complex_output_file is not None:
                with open(complex_output_file, "w") as f:
                    PDBFile.writeFile(modeller.topology, results_dict["pos"], f, keepIds=True)

        # record protein mmCIF file
        elif protein_file.suffix == ".cif":
            with open(protein_output_file, "w") as f:
                PDBxFile.writeFile(
                    protein_complex.topology,
                    results_dict["pos_final"][:num_particles_protein],
                    f,
                    keepIds=True,
                )
            remove_hydrogens_from_pdb(str(protein_output_file), str(protein_output_file))
            # record complex mmCIF file
            if complex_output_file is not None:
                with open(complex_output_file, "w"):
                    PDBxFile.writeFile(modeller.topology, results_dict["pos"], f, keepIds=True)

        # record ligand file(s)
        if output_file is not None:
            new_mols = []
            num_ligand_atoms_observed = 0
            for ligand_index in range(len(ligands)):
                num_atoms_in_ligand = len(ligands[ligand_index].atoms)
                pos_start_index = num_particles_protein + num_ligand_atoms_observed
                pos_end_index = pos_start_index + num_atoms_in_ligand
                new_molecule = Molecule.from_smiles(
                    ligands_smiles[ligand_index], allow_undefined_stereo=allow_undefined_stereo
                )
                new_molecule.add_conformer(
                    openff_Quantity(
                        results_dict["pos_final"][pos_start_index:pos_end_index], units="angstrom"
                    )
                )
                new_mol = new_molecule.to_rdkit()
                new_mol = remove_mol_hydrogens_and_reorder(new_mol)
                new_mols.append(new_mol)
                num_ligand_atoms_observed += num_atoms_in_ligand
            new_mol = combine_molecules(new_mols)
            with Chem.SDWriter(str(output_file)) as f:
                f.write(new_mol)

    else:
        # save initial state
        state_before = simulation.context.getState(getEnergy=True, getPositions=False)
        results_dict["e_init"] = state_before.getPotentialEnergy()
        if report_initial_energy_only:
            return results_dict

        # minimize
        logger.info(f"Minimizing {name}")
        simulation.minimizeEnergy(
            tolerance=tolerance * kilojoule / mole / nanometer, maxIterations=max_iterations
        )

        # save final state
        logger.info(f"Saving {name}")
        state_after = simulation.context.getState(getEnergy=True, getPositions=True)
        results_dict["e_final"] = state_after.getPotentialEnergy()

        # save ligand(s)
        ligand_mols = []
        num_ligand_atoms_observed = 0
        for ligand_index in range(len(ligands)):
            num_atoms_in_ligand = len(ligands[ligand_index].atoms)
            pos_start_index = num_particles_protein + num_ligand_atoms_observed
            pos_end_index = pos_start_index + num_atoms_in_ligand
            ligand_positions = state_after.getPositions(asNumpy=True)[
                pos_start_index:pos_end_index
            ]
            ligands[ligand_index].add_conformer(ligand_positions)
            ligand_mols.append(ligands[ligand_index].to_rdkit())
            num_ligand_atoms_observed += num_atoms_in_ligand
        ligand = combine_molecules(ligand_mols)
        if output_file is not None:
            save_with_rdkit(ligand, output_file, conformer_index=1, name=name, mol=ligand)

        results_dict["ligands"] = ligands

    return results_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="minimize_energy.yaml",
)
def minimize_energy(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Minimize the energy of a ligand in a protein pocket using OpenMM."""
    logger.setLevel(cfg.log_level)

    protein_file_path = Path(cfg.protein_file)
    ligand_file_path = Path(cfg.ligand_file)
    output_file_path = Path(cfg.output_file)
    protein_output_file_path = Path(cfg.protein_output_file) if cfg.protein_output_file else None
    complex_output_file_path = Path(cfg.complex_output_file) if cfg.complex_output_file else None
    temp_directory = Path(cfg.temp_dir)
    prep_only = cfg.prep_only

    if not protein_file_path.exists():
        raise FileNotFoundError(f"File does not exist: {protein_file_path}")
    if not ligand_file_path.exists():
        raise FileNotFoundError(f"File does not exist: {ligand_file_path}")

    temp_protein_file_path, temp_ligand_file_path = protein_file_path, ligand_file_path
    if cfg.relax_protein and not prep_only:
        assert (
            protein_output_file_path is not None
        ), "Protein output file path must be provided when relaxing the protein."
        temp_protein_file_path = Path(str(protein_file_path).replace(".pdb", "_temp.pdb"))
        temp_ligand_file_path = Path(str(ligand_file_path).replace(".sdf", "_temp.sdf"))
        shutil.copyfile(protein_file_path, temp_protein_file_path)
        shutil.copyfile(ligand_file_path, temp_ligand_file_path)
        logger.info("Performing iteration 0 of complex relaxation")

    try:
        num_attempts = 0
        results_dict = optimize_ligand_in_pocket(
            protein_file=temp_protein_file_path,
            ligand_file=temp_ligand_file_path,
            output_file=output_file_path,
            protein_output_file=protein_output_file_path,
            complex_output_file=complex_output_file_path,
            temp_dir=temp_directory,
            prep_only=prep_only,
            name=cfg.name,
            platform_name=cfg.platform,
            cuda_device_index=cfg.cuda_device_index,
            add_solvent=cfg.add_solvent,
            relax_protein=cfg.relax_protein,
            remove_initial_protein_hydrogens=cfg.remove_initial_protein_hydrogens,
            assign_each_ligand_unique_force=cfg.assign_each_ligand_unique_force,
            report_initial_energy_only=cfg.report_initial_energy_only,
            model_ions=cfg.model_ions,
            cache_files=cfg.cache_files,
            assign_partial_charges_manually=cfg.assign_partial_charges_manually,
            tolerance=2.39 if cfg.relax_protein else 0.01,
        )
    except Exception as e:
        logger.error(
            f"Complex relaxation was not fully successful due to: {e}. Copying the input files as the relaxed output files."
        )
        # organize output files
        shutil.copyfile(ligand_file_path, output_file_path)
        try:
            if complex_output_file_path is not None:
                os.remove(complex_output_file_path)
        except OSError:
            pass

        logger.info(f"Finalizing complex relaxation with relax_protein={cfg.relax_protein}")

        if cfg.relax_protein:
            # organize output files
            shutil.copyfile(protein_file_path, protein_output_file_path)
            # clean up temporary files
            try:
                os.remove(temp_protein_file_path)
            except OSError:
                pass
            try:
                os.remove(temp_ligand_file_path)
            except OSError:
                pass

            temp_name = temp_protein_file_path.stem if cfg.name is None else cfg.name
            try:
                os.remove(temp_directory / f"{temp_name}_prepped_protein.pdb")
            except OSError:
                pass
            try:
                os.remove(temp_directory / f"{temp_name}_prepped_ligand.sdf")
            except OSError:
                pass

        return None

    if prep_only:
        sys.exit(0)

    if not cfg.relax_protein:
        if cfg.report_initial_energy_only:
            energy_before = results_dict["e_init"].value_in_unit(kilojoule / mole)
            logger.info(f"{ligand_file_path}, " + f"E_start: {energy_before:.2f} kJ/mol, ")
        else:
            energy_before = results_dict["e_init"].value_in_unit(kilojoule / mole)
            energy_after = results_dict["e_final"].value_in_unit(kilojoule / mole)
            logger.info(
                f"{ligand_file_path}, "
                + f"E_start: {energy_before:.2f} kJ/mol, "
                + f"E_end: {energy_after:.2f} kJ/mol, "
                + f"ΔE: {energy_after - energy_before:.2f} kJ/mol"
            )
    else:
        try:
            shutil.copyfile(protein_output_file_path, temp_protein_file_path)
            while results_dict["e_final"] > 0 and num_attempts < cfg.max_num_attempts:
                logger.info(f"Performing iteration {num_attempts + 1} of complex relaxation")
                results_dict = optimize_ligand_in_pocket(
                    protein_file=temp_protein_file_path,
                    ligand_file=temp_ligand_file_path,
                    output_file=output_file_path,
                    protein_output_file=protein_output_file_path,
                    complex_output_file=complex_output_file_path,
                    temp_dir=temp_directory,
                    prep_only=prep_only,
                    name=cfg.name,
                    platform_name=cfg.platform,
                    cuda_device_index=cfg.cuda_device_index,
                    add_solvent=cfg.add_solvent,
                    relax_protein=cfg.relax_protein,
                    assign_each_ligand_unique_force=cfg.assign_each_ligand_unique_force,
                    report_initial_energy_only=cfg.report_initial_energy_only,
                    model_ions=cfg.model_ions,
                    cache_files=cfg.cache_files,
                    tolerance=2.39 if cfg.relax_protein else 0.01,
                )
                num_attempts += 1
                shutil.copyfile(protein_output_file_path, temp_protein_file_path)

            # measure initial protein violations
            protein_score, violation_residue_idx = compute_protein_local_geometry_violations(
                str(temp_protein_file_path)
            )
            num_attempts = 0
            while (
                protein_score > 0 or results_dict["e_final"] > 0
            ) and num_attempts < cfg.max_num_attempts:
                parser = (
                    MMCIFParser(QUIET=True)
                    if os.path.splitext(temp_protein_file_path)[-1] == ".cif"
                    else PDBParser(QUIET=True)
                )
                structure = parser.get_structure("x", temp_protein_file_path)
                randomly_rotate_protein_side_chains(
                    structure, violation_residue_idx, min_angle=-np.pi / 2, max_angle=np.pi / 2
                )
                save_biopython_structure_to_pdb(structure, temp_protein_file_path, ca_only=False)
                results_dict = optimize_ligand_in_pocket(
                    protein_file=temp_protein_file_path,
                    ligand_file=temp_ligand_file_path,
                    output_file=output_file_path,
                    protein_output_file=protein_output_file_path,
                    complex_output_file=complex_output_file_path,
                    temp_dir=temp_directory,
                    prep_only=prep_only,
                    name=cfg.name,
                    platform_name=cfg.platform,
                    cuda_device_index=cfg.cuda_device_index,
                    add_solvent=cfg.add_solvent,
                    relax_protein=cfg.relax_protein,
                    assign_each_ligand_unique_force=cfg.assign_each_ligand_unique_force,
                    report_initial_energy_only=cfg.report_initial_energy_only,
                    model_ions=cfg.model_ions,
                    cache_files=cfg.cache_files,
                    tolerance=2.39 if cfg.relax_protein else 0.01,
                )
                num_attempts += 1
                shutil.copyfile(protein_output_file_path, temp_protein_file_path)
                # measure subsequent protein violations
                protein_score, violation_residue_idx = compute_protein_local_geometry_violations(
                    str(temp_protein_file_path)
                )
            # measure final ligand violations
            ligand_score = compute_ligand_local_geometry_violations(
                # NOTE: passing `None` for the first argument will ensure the relaxed ligand's
                # pairwise distances closely match a random RDKit conformer's pairwise distances
                None,
                str(temp_ligand_file_path),
            )
            if (
                protein_score > 0
                or ligand_score > 0
                or results_dict["e_final"] > cfg.max_final_e_value
            ):
                logger.warning(
                    f"Complex relaxation was not fully successful after all relaxation attempts and scoring, yielding a final protein score of {protein_score}, a final (maximum) ligand score of {ligand_score}, and a final E-value of {results_dict['e_final']}. Copying the input files as the relaxed output files."
                )
                shutil.copyfile(protein_file_path, protein_output_file_path)
                shutil.copyfile(ligand_file_path, output_file_path)
                try:
                    if complex_output_file_path is not None:
                        os.remove(complex_output_file_path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(
                f"Complex relaxation was not fully successful due to: {e}. Copying the input files as the relaxed output files."
            )
            # organize output files
            shutil.copyfile(protein_file_path, protein_output_file_path)
            shutil.copyfile(ligand_file_path, output_file_path)
            try:
                if complex_output_file_path is not None:
                    os.remove(complex_output_file_path)
            except OSError:
                pass

        logger.info(f"Finalizing complex relaxation with relax_protein={cfg.relax_protein}")

        # clean up temporary files
        try:
            os.remove(temp_protein_file_path)
        except OSError:
            pass
        try:
            os.remove(temp_ligand_file_path)
        except OSError:
            pass

        temp_name = temp_protein_file_path.stem if cfg.name is None else cfg.name
        try:
            os.remove(temp_directory / f"{temp_name}_prepped_protein.pdb")
        except OSError:
            pass
        try:
            os.remove(temp_directory / f"{temp_name}_prepped_ligand.sdf")
        except OSError:
            pass

    return results_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs/model",
    config_name="minimize_energy.yaml",
)
def main(cfg: DictConfig):
    """Minimize the energy of a ligand in a protein pocket."""
    minimize_energy(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
