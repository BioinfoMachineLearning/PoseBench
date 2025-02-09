# From: https://github.com/gcorso/DiffDock

import warnings

import rootutils
from rdkit import Chem
from rdkit.Chem import AllChem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def generate_conformer(mol):
    """Generate conformer for the ligand."""
    ps = AllChem.ETKDGv2()
    failures, id = 0, -1
    while failures < 3 and id == -1:
        if failures > 0:
            log.warning(f"rdkit coords could not be generated. trying again {failures}.")
        id = AllChem.EmbedMolecule(mol, ps)
        failures += 1
    if id == -1:
        log.warning(
            "rdkit coords could not be generated without using random coords. using random coords now."
        )
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
        return True
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol, confId=0)
    return False


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Read molecule from file and return RDKit molecule object."""
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ""
        for line in pdbqt_data:
            pdb_block += f"{line[:66]}\n"
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError(
            "Expect the format of the molecule_file to be "
            "one of .mol2, .sdf, .pdbqt and .pdb, got {}".format(molecule_file)
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except Exception as e:
                warnings.warn("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        return None

    return mol
