# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import glob
import logging
import os
import shutil
import subprocess  # nosec
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pypdb
import rootutils
from beartype import beartype
from beartype.typing import Any, List, Optional, Set, Tuple, Union
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from biopandas.pdb import PandasPdb
from prody import parsePDB, writePDB, writePDBStream
from rdkit import Chem
from rdkit.Chem import AllChem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench.data.components.esmfold_apo_to_holo_alignment import read_molecule

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    casp_dataset_requested = os.path.basename(input_data_dir) == "targets"
    if casp_dataset_requested:
        # parse CASP inputs uniquely
        smiles_filepaths = list(glob.glob(os.path.join(input_data_dir, "*.smiles.txt")))
        for smiles_filepath in smiles_filepaths:
            pdb_id = os.path.basename(smiles_filepath).split(".")[0]
            smiles_df = pd.read_csv(smiles_filepath, delim_whitespace=True)
            assert smiles_df.columns.tolist() == [
                "ID",
                "Name",
                "SMILES",
                "Relevant",
            ], "SMILES DataFrame must have columns ['ID', 'Name', 'SMILES', 'Relevant']."
            mol_smiles = "|".join(smiles_df["SMILES"].tolist())
            assert len(mol_smiles) > 0, f"SMILES string for {pdb_id} cannot be empty."
            smiles_and_pdb_id_list.append((mol_smiles, pdb_id))
    else:
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
                # NOTE: we first try to parse `.sdf` files if they exist for the current dataset
                if os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")):
                    mol = read_molecule(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf"),
                        remove_hs=True,
                        sanitize=True,
                    )
                # NOTE: DockGen uses `.pdb` files to store its ligands
                if mol is None and os.path.exists(os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb")):
                    mol = read_molecule(
                        os.path.join(pdb_dir, f"{pdb_id}_ligand.pdb"),
                        remove_hs=True,
                        sanitize=True,
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
                if mol_smiles is not None:
                    smiles_and_pdb_id_list.append((mol_smiles, pdb_id))
    return smiles_and_pdb_id_list


@beartype
def extract_sequences_from_protein_structure_file(
    protein_filepath: Union[str, Path], structure: Optional[Structure] = None
) -> List[str]:
    """Extract the chain sequences from a protein structure file.

    :param protein_filepath: Path to the protein structure file.
    :param structure: Optional BioPython structure object to use instead.
    :return: A list of protein sequences.
    """
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

    try:
        if structure is None:
            structure = biopython_parser.get_structure("random_id", protein_filepath)
        structure = structure[0]
    except Exception as e:
        logger.error(f"Due to exception {e}, could not parse protein {protein_filepath}.")
        return None

    sequences = []
    for chain in structure:
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
                        f"Due to exception {e}, encountered unknown amino acid {residue.get_resname()} in the protein {protein_filepath}. Replacing it with a dash (i.e., `-`)."
                    )
        sequences.append(seq)
    return sequences


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
def renumber_pdb_df_residues(input_pdb_file: str, output_pdb_file: str):
    """Renumber residues in a PDB file starting from 1 for each chain.

    :param input_pdb_file: Path to the input PDB file.
    """
    # Load the PDB file
    pdb = PandasPdb().read_pdb(input_pdb_file)

    # Iterate through each chain
    for _, chain_df in pdb.df["ATOM"].groupby("chain_id"):
        # Get the minimum residue index for the current chain
        min_residue_index = chain_df["residue_number"].min()

        # Reindex the residues starting from 1
        chain_df["residue_number"] -= min_residue_index - 1

        # Update the PDB dataframe with the reindexed chain
        pdb.df["ATOM"].loc[chain_df.index] = chain_df

    # Write the modified PDB file
    pdb.to_pdb(output_pdb_file)


@beartype
def renumber_biopython_structure_residues(
    structure: Structure, gap_insertion_point: Optional[str] = None
) -> Structure:
    """Renumber residues in a PDB file using BioPython starting from 1 for each chain.

    :param structure: BioPython structure object.
    :param gap_insertion_point: Optional `:`-separated string representing the chain-residue pair
        index of the residue at which to insert a single index gap.
    :return: BioPython structure object with renumbered residues.
    """
    # Iterate through each model in the structure
    if gap_insertion_point is not None:
        assert (
            len(gap_insertion_point.split(":")) == 2
        ), "When provided, gap insertion point must be in the format 'chain_id:residue_index'."
    gap_insertion_chain_id = (
        gap_insertion_point.split(":")[0] if gap_insertion_point is not None else None
    )
    gap_insertion_residue_index = (
        int(gap_insertion_point.split(":")[1]) if gap_insertion_point is not None else None
    )
    for model in structure:
        # Iterate through each chain in the model
        for chain in model:
            # Get the minimum residue index for the current chain
            min_residue_index = min(residue.id[1] for residue in chain)

            # Reindex the residues starting from 1
            gap_insertion_counter = 0
            for residue in chain:
                new_residue_index = residue.id[1] - min_residue_index + 1
                gap_index_found = (
                    gap_insertion_chain_id is not None
                    and gap_insertion_residue_index is not None
                    and chain.id == gap_insertion_chain_id
                    and new_residue_index == gap_insertion_residue_index
                )
                if gap_index_found:
                    gap_insertion_counter = 1
                residue.id = (" ", new_residue_index + gap_insertion_counter, residue.id[2])
                for atom in residue:
                    atom.serial_number = None  # Reset atom serial number

    return structure


def get_pdb_components_with_prody(pdb_id) -> tuple:
    """Split a protein-ligand pdb into protein and ligand components using ProDy.

    :param pdb_id: PDB ID
    :return: protein structure and ligand residues
    """
    pdb = parsePDB(pdb_id)
    protein = pdb.select("protein")
    ligand = pdb.select("not (protein or nucleotide or water)")
    return protein, ligand


def write_pdb_with_prody(protein, pdb_name, add_element_types=False):
    """Write a protein to a pdb file using ProDy.

    :param protein: protein object from prody
    :param pdb_name: base name for the pdb file
    :param add_element_types: whether to add element types to the pdb file
    """
    writePDB(pdb_name, protein)
    if add_element_types:
        with open(pdb_name.replace(".pdb", "_elem.pdb"), "w") as f:
            subprocess.run(  # nosec
                f"pdb_element {pdb_name}",
                shell=True,
                check=True,
                stdout=f,
            )
        shutil.move(pdb_name.replace(".pdb", "_elem.pdb"), pdb_name)
    logger.info(f"Wrote {pdb_name}")


def process_ligand_with_prody(
    ligand,
    res_name,
    chain,
    resnum,
    sanitize: bool = True,
    sub_smiles: Optional[str] = None,
) -> Chem.Mol:
    """
    Add bond orders to a pdb ligand using ProDy.
    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from pypdb
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3

    :param ligand: ligand as generated by prody
    :param res_name: residue name of ligand to extract
    :param chain: chain of ligand to extract
    :param resnum: residue number of ligand to extract
    :param sanitize: whether to sanitize the molecule
    :param sub_smiles: optional SMILES string of the ligand molecule
    :return: molecule with bond orders assigned
    """
    sub_smiles_provided = sub_smiles is not None

    output = StringIO()
    sub_mol = ligand.select(f"resname {res_name} and chain {chain} and resnum {resnum}")
    chem_desc = pypdb.describe_chemical(f"{res_name}")
    if chem_desc is not None and not sub_smiles_provided:
        sub_smiles = None
        for item in chem_desc.get("pdbx_chem_comp_descriptor", []):
            if item.get("type") == "SMILES":
                sub_smiles = item.get("descriptor")
                break

    if sub_smiles is not None:
        template = AllChem.MolFromSmiles(sub_smiles)
    else:
        template = None

    writePDBStream(output, sub_mol)
    pdb_string = output.getvalue()
    rd_mol = AllChem.MolFromPDBBlock(pdb_string, sanitize=sanitize)

    if sub_smiles_provided and template is not None:
        # Ensure the input ligand perfectly matches the template ligand
        assert (
            rd_mol.GetNumAtoms() == template.GetNumAtoms()
        ), "Number of atoms in both molecules is different."

    try:
        if template is not None:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template, rd_mol)
        else:
            new_mol = rd_mol
    except ValueError:
        new_mol = rd_mol

    return new_mol


def write_sdf(new_mol: Chem.Mol, pdb_name: str):
    """Write an RDKit molecule to an SD file.

    :param new_mol: RDKit molecule
    :param pdb_name: name of the output file
    """
    writer = Chem.SDWriter(pdb_name)
    writer.write(new_mol)
    logger.info(f"Wrote {pdb_name}")


def extract_protein_and_ligands_with_prody(
    input_pdb_file: str,
    protein_output_pdb_file: str,
    ligands_output_sdf_file: str,
    sanitize: bool = True,
    add_element_types: bool = False,
    ligand_smiles: Optional[str] = None,
) -> Optional[Chem.Mol]:
    """Using ProDy, extract protein atoms and ligand molecules from a PDB file and write them to
    separate files.

    :param input_pdb_file: The input PDB file.
    :param protein_output_pdb_file: The output PDB file for the protein atoms.
    :param ligands_output_sdf_file: The output SDF file for the ligand molecules.
    :param sanitize: Whether to sanitize the ligand molecules.
    :param add_element_types: Whether to add element types to the protein atoms.
    :param ligand_smiles: The SMILES string of the ligand molecule.
    :return: The combined final ligand molecule(s) as an RDKit molecule.
    """
    protein, ligand = get_pdb_components_with_prody(input_pdb_file)
    write_pdb_with_prody(protein, protein_output_pdb_file, add_element_types=add_element_types)

    ligand_resnames = ligand.getResnames()
    ligand_chids = ligand.getChids()
    ligand_resnums = ligand.getResnums()

    seen = set()
    resname_chain_resnum_list = []
    for resname, chid, resnum in zip(ligand_resnames, ligand_chids, ligand_resnums):
        if (resname, chid, resnum) not in seen:
            seen.add((resname, chid, resnum))
            resname_chain_resnum_list.append((resname, chid, resnum))

    new_mol = None
    new_mol_list = []
    ligand_smiles_components = ligand_smiles.split(".") if ligand_smiles is not None else None
    for i, resname_chain_resnum in enumerate(resname_chain_resnum_list, start=1):
        resname, chain, resnum = resname_chain_resnum
        sub_smiles = (
            ligand_smiles_components[i - 1]
            if ligand_smiles_components is not None
            and len(ligand_smiles_components) == len(resname_chain_resnum_list)
            else None
        )
        new_mol = process_ligand_with_prody(
            ligand, resname, chain, resnum, sanitize=sanitize, sub_smiles=sub_smiles
        )
        if new_mol is not None:
            new_mol_list.append(new_mol)
            write_sdf(
                new_mol,
                os.path.join(
                    os.path.dirname(ligands_output_sdf_file),
                    f"{Path(ligands_output_sdf_file).stem}_{resname}_{i}.sdf",
                ),
            )

    if len(new_mol_list):
        new_mol = combine_molecules(new_mol_list)
        write_sdf(new_mol, ligands_output_sdf_file)

    return new_mol


@beartype
def create_sdf_file_from_smiles(smiles: str, output_sdf_file: str) -> str:
    """Create an SDF file from a SMILES string.

    :param smiles: SMILES string of the molecule.
    :param output_sdf_file: Path to the output SDF file.
    :return: Path to the output SDF file.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    writer = Chem.SDWriter(output_sdf_file)
    writer.write(mol)
    return output_sdf_file


@beartype
def count_num_residues_in_pdb_file(pdb_filepath: str) -> int:
    """Count the number of Ca atoms (i.e., residues) in a PDB file.

    :param pdb_filepath: Path to PDB file.
    :return: Number of Ca atoms (i.e., residues) in the PDB file.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_filepath)
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    count += 1
    return count


@beartype
def count_pdb_inter_residue_clashes(pdb_filepath: str, clash_cutoff: float = 0.63) -> int:
    """
    Count the number of inter-residue clashes in a protein PDB file.
    From: https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/

    :param pdb_filepath: Path to the PDB file.
    :param clash_cutoff: The cutoff for what is considered a clash.
    :return: The number of inter-residue clashes in the structure.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_filepath)

    # Atomic radii for various atom types.
    # You can comment out the ones you don't care about or add new ones
    atom_radii = {
        #    "H": 1.20,  # Who cares about hydrogen??
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "F": 1.47,
        "P": 1.80,
        "CL": 1.75,
        "MG": 1.73,
    }

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {
        i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j]))
        for i in atom_radii
        for j in atom_radii
    }

    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")

    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)

    # Initialize a list to hold clashes
    clashes = []

    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))

        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]

        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]

            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue

            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (
                atom_2.name == "N" and atom_1.name == "C"
            ):
                continue

            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue

            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))

    return len(clashes) // 2
