import os

import numpy as np
import rootutils
from beartype import beartype
from beartype.typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from openfold.np.protein import Protein as OFProtein
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components import residue_constants
from flowdock.utils.data_utils import (
    PDB_CHAIN_IDS,
    PDB_MAX_CHAINS,
    FDProtein,
    create_full_prot,
    get_mol_with_new_conformer_coords,
)

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PROT_LIG_PAIRS = List[Tuple[OFProtein, Tuple[Chem.Mol, ...]]]


@beartype
def _chain_end(
    atom_index: Union[int, np.int64],
    end_resname: str,
    chain_name: str,
    residue_index: Union[int, np.int64],
) -> str:
    """Returns a PDB `TER` record for the end of a chain.

    Adapted from: https://github.com/jasonkyuyim/se3_diffusion

    :param atom_index: The index of the last atom in the chain.
    :param end_resname: The residue name of the last residue in the chain.
    :param chain_name: The chain name of the last residue in the chain.
    :param residue_index: The residue index of the last residue in the chain.
    :return: A PDB `TER` record.
    """
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


@beartype
def res_1to3(restypes: List[str], r: Union[int, np.int64]) -> str:
    """Convert a residue type from 1-letter to 3-letter code.

    :param restypes: List of residue types.
    :param r: Residue type index.
    :return: 3-letter code as a string.
    """
    return residue_constants.restype_1to3.get(restypes[r], "UNK")


@beartype
def to_pdb(prot: Union[OFProtein, FDProtein], model=1, add_end=True, add_endmdl=True) -> str:
    """Converts a `Protein` instance to a PDB string.

    Adapted from: https://github.com/jasonkyuyim/se3_diffusion

    :param prot: The protein to convert to PDB.
    :param model: The model number to use.
    :param add_end: Whether to add an `END` record.
    :param add_endmdl: Whether to add an `ENDMDL` record.
    :return: PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(int)
    chain_index = prot.chain_index.astype(int)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # construct a mapping from chain integer indices to chain ID strings
    chain_ids = {}
    for i in np.unique(chain_index):  # NOTE: `np.unique` gives sorted output
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    last_chain_index = chain_index[0]
    # add all atom sites
    for i in range(aatype.shape[0]):
        # close the previous chain if in a multichain PDB
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(restypes, aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # NOTE: atom index increases at the `TER` symbol

        res_name_3 = res_1to3(restypes, aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # NOTE: `Protein` supports only C, N, O, S, this works
            charge = ""
            # NOTE: PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # close the final chain
    pdb_lines.append(
        _chain_end(
            atom_index,
            res_1to3(restypes, aatype[-1]),
            chain_ids[chain_index[-1]],
            residue_index[-1],
        )
    )
    if add_endmdl:
        pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # pad all lines to 80 characters
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # add terminating newline


@beartype
def construct_prot_lig_pairs(outputs: Dict[str, Any], batch_index: int) -> PROT_LIG_PAIRS:
    """Construct protein-ligand pairs from model outputs.

    :param outputs: The model outputs.
    :param batch_index: The index of the current batch.
    :return: A list of protein-ligand object pairs.
    """
    protein_batch_indexer = outputs["protein_batch_indexer"]
    ligand_batch_indexer = outputs["ligand_batch_indexer"]

    protein_all_atom_mask = outputs["res_atom_mask"][protein_batch_indexer == batch_index]
    protein_all_atom_coordinates_mask = np.broadcast_to(
        np.expand_dims(protein_all_atom_mask, -1), (protein_all_atom_mask.shape[0], 37, 3)
    )
    protein_aatype = outputs["aatype"][protein_batch_indexer == batch_index]

    # assemble predicted structures
    prot_lig_pairs = []
    for protein_coordinates, ligand_coordinates in zip(
        outputs["protein_coordinates_list"], outputs["ligand_coordinates_list"]
    ):
        protein_all_atom_coordinates = (
            protein_coordinates[protein_batch_indexer == batch_index]
            * protein_all_atom_coordinates_mask
        )
        protein = create_full_prot(
            protein_all_atom_coordinates,
            protein_all_atom_mask,
            protein_aatype,
            b_factors=outputs["b_factors"][batch_index] if "b_factors" in outputs else None,
        )
        ligand = get_mol_with_new_conformer_coords(
            outputs["ligand_mol"][batch_index],
            ligand_coordinates[ligand_batch_indexer == batch_index],
        )
        ligands = tuple(Chem.GetMolFrags(ligand, asMols=True, sanitizeFrags=False))
        prot_lig_pairs.append((protein, ligands))

    # assemble ground-truth structures
    if "gt_protein_coordinates" in outputs and "gt_ligand_coordinates" in outputs:
        protein_gt_all_atom_coordinates = (
            outputs["gt_protein_coordinates"][protein_batch_indexer == batch_index]
            * protein_all_atom_coordinates_mask
        )
        gt_protein = create_full_prot(
            protein_gt_all_atom_coordinates,
            protein_all_atom_mask,
            protein_aatype,
        )
        gt_ligand = get_mol_with_new_conformer_coords(
            outputs["ligand_mol"][batch_index],
            outputs["gt_ligand_coordinates"][ligand_batch_indexer == batch_index],
        )
        gt_ligands = tuple(Chem.GetMolFrags(gt_ligand, asMols=True, sanitizeFrags=False))
        prot_lig_pairs.append((gt_protein, gt_ligands))

    return prot_lig_pairs


@beartype
def write_prot_lig_pairs_to_pdb_file(prot_lig_pairs: PROT_LIG_PAIRS, output_filepath: str):
    """Write a list of protein-ligand pairs to a PDB file.

    :param prot_lig_pairs: List of protein-ligand object pairs, where each ligand may consist of
        multiple ligand chains.
    :param output_filepath: Output file path.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w") as f:
        model_id = 1
        for prot, lig_mols in prot_lig_pairs:
            pdb_prot = to_pdb(prot, model=model_id, add_end=False, add_endmdl=False)
            f.write(pdb_prot)
            for lig_mol in lig_mols:
                f.write(
                    Chem.MolToPDBBlock(lig_mol).replace(
                        "END\n", "TER\n"
                    )  # enable proper ligand chain separation
                )
            f.write("END\n")
            f.write("ENDMDL\n")  # add `ENDMDL` line to separate models
            model_id += 1


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = False,
) -> FDProtein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return FDProtein(
        letter_sequences=None,
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]),
        chain_index=chain_index,
        b_factors=b_factors,
        atomtypes=None,
    )


def write_pdb_single(
    result: ModelOutput,
    out_path: str = os.path.join("test_results", "debug.pdb"),
    model: int = 1,
    b_factors: Optional[np.ndarray] = None,
):
    """Write a single model to a PDB file.

    :param result: Model results batch.
    :param out_path: Output path.
    :param model: Model ID.
    :param b_factors: Optional B-factors.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    protein = from_prediction(result["features"], result, b_factors=b_factors)
    out_string = to_pdb(protein, model=model)
    with open(out_path, "w") as of:
        of.write(out_string)


def write_pdb_models(
    results,
    out_path: str = os.path.join("test_results", "debug.pdb"),
    b_factors: Optional[np.ndarray] = None,
):
    """Write multiple models to a PDB file.

    :param results: Model results.
    :param out_path: Output path.
    :param b_factors: Optional B-factors.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as of:
        for mid, result in enumerate(results):
            protein = from_prediction(
                result["features"],
                result,
                b_factors=b_factors[mid] if b_factors is not None else None,
            )
            out_string = to_pdb(protein, model=mid + 1)
            of.write(out_string)
        of.write("END")


def write_conformer_sdf(
    mol: Chem.Mol,
    confs: Optional[np.array] = None,
    out_path: str = os.path.join("test_results", "debug.sdf"),
):
    """Write a molecule with conformers to an SDF file.

    :param mol: RDKit molecule.
    :param confs: Conformers.
    :param out_path: Output path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if confs is None:
        w = Chem.SDWriter(out_path)
        w.write(mol)
        w.close()
        return 0
    mol.RemoveAllConformers()
    for i in range(len(confs)):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for j in range(mol.GetNumAtoms()):
            x, y, z = confs[i, j].tolist()
            conf.SetAtomPosition(j, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

    w = Chem.SDWriter(out_path)
    try:
        for cid in range(len(confs)):
            w.write(mol, confId=cid)
    except Exception as e:
        w.SetKekulize(False)
        for cid in range(len(confs)):
            w.write(mol, confId=cid)
    w.close()
    return 0
