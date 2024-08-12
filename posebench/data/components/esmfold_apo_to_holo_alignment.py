# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from DiffDock: (https://github.com/gcorso/DiffDock)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import rootutils
from beartype.typing import Any, List, Optional, Tuple, Union
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import spatial
from scipy import spatial as spa
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation
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

biopython_parser = PDBParser()


def read_molecule(
    molecule_file: str, sanitize: bool = False, calc_charges: bool = False, remove_hs: bool = False
) -> Optional[Chem.Mol]:
    """Load an RDKit molecule from a given filepath."""
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".mol"):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
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
        raise ValueError(
            "Expected the format of `molecule_file` to be "
            f"one of `.mol`, `.mol2`, `.sdf`, `.pdbqt`, or `.pdb`, yet got {molecule_file}"
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # compute Gasteiger charges on the molecule
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except Exception:
                warnings.warn("Unable to compute charges for the input molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        logger.warning(f"RDKit was unable to read the molecule due to the error: {e}")
        return None

    return mol


def read_mols(
    dataset_dir: str,
    name: str,
    remove_hs: bool = False,
) -> List[Chem.Mol]:
    """Load RDKit `Mol` objects corresponding to a dataset's ligand name."""
    ligs = []
    for file in os.listdir(os.path.join(dataset_dir, name)):
        if file.endswith("_ligand.sdf") and "rdkit" not in file:
            lig = read_molecule(
                os.path.join(dataset_dir, name, file), remove_hs=remove_hs, sanitize=True
            )
            if lig is not None:
                ligs.append(lig)
    return ligs


def parse_pdb_from_path(path: str) -> Model:
    """Load a BioPython structure from a given PDB filepath."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure("random_id", path)
        rec = structure[0]
    return rec


def extract_receptor_structure(
    rec: Model,
    lig: Optional[Chem.Mol],
    lm_embedding_chains: Optional[List[Any]] = None,
    filter_out_hetero_residues: bool = False,
) -> Tuple[Model, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract a receptor protein structure from a given `Structure` object."""
    if lig is not None:
        conf = lig.GetConformer()
        lig_coords = conf.GetPositions()
    else:
        lig_coords = None
    min_distances = []
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for residue in chain:
            if residue.get_resname() == "HOH" or (
                filter_out_hetero_residues and len(residue.get_id()[0]) > 1
            ):
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))

            if c_alpha is not None and n is not None and c is not None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0 and lig_coords is not None:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf

        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if not count == 0:
            valid_chain_ids.append(chain.get_id())

    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    valid_lm_embeddings = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            if lm_embedding_chains is not None:
                if i >= len(lm_embedding_chains):
                    raise ValueError(
                        "Encountered valid chain ID that was not present in the LM embeddings."
                    )
                valid_lm_embeddings.append(lm_embedding_chains[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [
        item for sublist in valid_coords for item in sublist
    ]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    lm_embeddings = (
        np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
    )
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords), "Number of Ca atoms does not match N atoms."
    assert len(c_alpha_coords) == len(c_coords), "Number of Ca atoms does not match C atoms."
    assert sum(valid_lengths) == len(c_alpha_coords), "Number of Ca atoms does not match."
    return rec, coords, c_alpha_coords, n_coords, c_coords, lm_embeddings


def align_prediction(
    smoothing_factor: Optional[float],
    dataset_calpha_coords: np.ndarray,
    esmfold_calpha_coords: np.ndarray,
    dataset_ligand_coords: Optional[np.ndarray],
    return_rotation: bool = False,
) -> Union[Tuple[Rotation, np.ndarray, np.ndarray], float]:
    """Perform an alignment of apo and holo protein structures and ligand coordinates using an
    optimized smoothing factor.

    :param smoothing_factor: Smoothing factor controlling the alignment.
    :param dataset_calpha_coords: Array of Ca atom coordinates for a dataset's protein structure.
    :param esmfold_calpha_coords: Array of Ca atom coordinates for a dataset's protein structure.
    :param dataset_ligand_coords: Array of ligand coordinates from a dataset.
    :param return_rotation: Whether to return the rotation matrix and centroids (default: `False`).
    :return: If return_rotation is `True`, returns a tuple containing rotation matrix (`Rotation`), centroid of CA atoms for a dataset protein (`np.ndarray`),
             and centroid of CA atoms for ESMFold (`np.ndarray`). If return_rotation is `False`, returns the inverse root mean square error of reciprocal distances (`float`).
    """
    if dataset_ligand_coords is not None:
        dataset_dists = spa.distance.cdist(dataset_calpha_coords, dataset_ligand_coords)
        weights = np.exp(-1 * smoothing_factor * np.amin(dataset_dists, axis=1))
        dataset_calpha_centroid = np.sum(
            np.expand_dims(weights, axis=1) * dataset_calpha_coords, axis=0
        ) / np.sum(weights)
        esmfold_calpha_centroid = np.sum(
            np.expand_dims(weights, axis=1) * esmfold_calpha_coords, axis=0
        ) / np.sum(weights)
    else:
        weights = None
        dataset_calpha_centroid = np.mean(dataset_calpha_coords, axis=0)
        esmfold_calpha_centroid = np.mean(esmfold_calpha_coords, axis=0)
    centered_dataset_calpha_coords = dataset_calpha_coords - dataset_calpha_centroid
    centered_esmfold_calpha_coords = esmfold_calpha_coords - esmfold_calpha_centroid

    rotation, _ = spa.transform.Rotation.align_vectors(
        centered_dataset_calpha_coords, centered_esmfold_calpha_coords, weights
    )
    if return_rotation:
        return rotation, dataset_calpha_centroid, esmfold_calpha_centroid

    if dataset_ligand_coords is not None:
        centered_dataset_ligand_coords = dataset_ligand_coords - dataset_calpha_centroid
        aligned_esmfold_calpha_coords = rotation.apply(centered_esmfold_calpha_coords)
        aligned_esmfold_dataset_dists = spa.distance.cdist(
            aligned_esmfold_calpha_coords, centered_dataset_ligand_coords
        )
        inv_r_rmse = np.sqrt(
            np.mean(((1 / dataset_dists) - (1 / aligned_esmfold_dataset_dists)) ** 2)
        )
    else:
        inv_r_rmse = np.nan
    return inv_r_rmse


def get_alignment_rotation(
    pdb_id: str,
    dataset_protein_path: str,
    esmfold_protein_path: str,
    dataset_path: str,
) -> Tuple[Optional[Rotation], Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculate the alignment rotation between apo and holo protein structures and their ligand
    coordinates.

    :param pdb_id: PDB ID of the protein-ligand complex.
    :param dataset_protein_path: Filepath to the PDB file of the protein structure from a dataset.
    :param esmfold_protein_path: Filepath to the PDB file of the protein structure from ESMFold.
    :param dataset: Name of the dataset.
    :param dataset_path: Filepath to the PDB file containing ligand coordinates.
    :param lig_connection_radius: Radius for connecting ligand atoms.
    :param exclude_af2aa_excluded_ligs: Whether to exclude ligands excluded from the AF2-AA
        dataset.
    :param skip_parsed_ligands: Whether to skip parsing ligands if they have already been parsed.
    :return: A tuple containing rotation matrix (Optional[Rotation]), centroid of Ca atoms for a
        dataset protein (Optional[np.ndarray]), and centroid of Ca atoms for ESMFold
        (Optional[np.ndarray]).
    """
    try:
        dataset_rec = parse_pdb_from_path(dataset_protein_path)
    except Exception as e:
        logger.warning(
            f"Unable to parse dataset protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None
    try:
        esmfold_rec = parse_pdb_from_path(esmfold_protein_path)
    except Exception as e:
        logger.warning(
            f"Unable to parse ESMFold protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None
    dataset_ligand = read_mols(dataset_path, pdb_id, remove_hs=True)[0]

    try:
        dataset_calpha_coords = extract_receptor_structure(
            dataset_rec, dataset_ligand, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract dataset protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None
    try:
        esmfold_calpha_coords = extract_receptor_structure(
            esmfold_rec, dataset_ligand, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract ESMFold protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None
    try:
        dataset_ligand_coords = dataset_ligand.GetConformer().GetPositions()
    except Exception as e:
        logger.warning(
            f"Unable to extract dataset ligand structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None

    if dataset_calpha_coords.shape != esmfold_calpha_coords.shape:
        logger.warning(
            f"Receptor structures differ for PDB ID {pdb_id}. Skipping due to shape mismatch:",
            dataset_calpha_coords.shape,
            esmfold_calpha_coords.shape,
        )
        return None, None, None

    res = minimize(
        align_prediction,
        [0.1],
        bounds=Bounds([0.0], [1.0]),
        args=(dataset_calpha_coords, esmfold_calpha_coords, dataset_ligand_coords),
        tol=1e-8,
    )

    smoothing_factor = res.x
    rotation, dataset_calpha_centroid, esmfold_calpha_centroid = align_prediction(
        smoothing_factor,
        dataset_calpha_coords,
        esmfold_calpha_coords,
        dataset_ligand_coords,
        return_rotation=True,
    )

    return rotation, dataset_calpha_centroid, esmfold_calpha_centroid


def align_apo_structure_to_holo_structure(
    cfg: DictConfig, filename: str, atom_df_name: str = "ATOM"
):
    """Align a given ESMFold apo structure to its corresponding holo structure.

    :param cfg: Hydra config for the alignment.
    :param filename: Filename of the ESMFold apo structure.
    :param atom_df_name: Name of the atom DataFrame derived from the corresponding PDB file input.
    """
    pdb_id = "_".join(Path(filename).stem.split("_")[:2])
    esm_protein_filename = os.path.join(cfg.esmfold_structures_dir, f"{pdb_id}.pdb")
    processed_protein_name = f"{pdb_id}_protein.pdb"
    processed_protein_filename = os.path.join(cfg.data_dir, pdb_id, processed_protein_name)
    esm_protein_output_filename = os.path.join(
        cfg.output_dir, f"{pdb_id}_holo_aligned_esmfold_protein.pdb"
    )

    rotation, dataset_calpha_centroid, esmfold_calpha_centroid = get_alignment_rotation(
        pdb_id=pdb_id,
        dataset_protein_path=processed_protein_filename,
        esmfold_protein_path=esm_protein_filename,
        dataset_path=cfg.data_dir,
    )

    if any(
        [item is None for item in [rotation, dataset_calpha_centroid, esmfold_calpha_centroid]]
    ):
        return

    ppdb_esmfold = PandasPdb().read_pdb(esm_protein_filename)
    ppdb_esmfold_pre_rot = (
        ppdb_esmfold.df[atom_df_name][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
        .squeeze()
        .astype(np.float32)
    )
    ppdb_esmfold_aligned = (
        rotation.apply(ppdb_esmfold_pre_rot - esmfold_calpha_centroid) + dataset_calpha_centroid
    )

    ppdb_esmfold.df[atom_df_name][["x_coord", "y_coord", "z_coord"]] = ppdb_esmfold_aligned
    ppdb_esmfold.to_pdb(path=esm_protein_output_filename, records=[atom_df_name], gz=False)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data/components",
    config_name="esmfold_apo_to_holo_alignment.yaml",
)
def main(cfg: DictConfig):
    """Align all ESMFold apo structures to their corresponding holo structures.

    :param cfg: Hydra config for the alignments.
    """
    if cfg.dataset not in ["posebusters_benchmark", "astex_diverse"]:
        raise ValueError(f"Dataset {cfg.dataset} is not supported.")
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    structure_file_inputs = [
        file
        for file in os.listdir(cfg.esmfold_structures_dir)
        if not os.path.exists(
            os.path.join(cfg.output_dir, f"{Path(file).stem}_holo_aligned_esmfold_protein.pdb")
        )
    ]
    pbar = tqdm(
        structure_file_inputs,
        desc="Submitting apo-to-holo alignment tasks for parallel processing",
        total=len(structure_file_inputs),
    )

    # process files in parallel
    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = []
        for filename in pbar:
            futures.append(executor.submit(align_apo_structure_to_holo_structure, cfg, filename))

        # wait for all tasks to complete
        for future in tqdm(
            futures,
            desc="Aligning each ESMFold apo structure to its corresponding holo structure",
            total=len(futures),
        ):
            future.result()


if __name__ == "__main__":
    main()
