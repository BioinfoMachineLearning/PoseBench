import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import rootutils
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig, open_dict
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveHs
from scipy import spatial as spa
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from flowdock import utils`)
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

from flowdock import register_custom_omegaconf_resolvers
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    combine_molecules,
    convert_protein_pts_to_pdb,
    pdb_filepath_to_protein,
)

log = RankedLogger(__name__, rank_zero_only=True)

biopython_parser = PDBParser()


@beartype
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
        log.warning(f"RDKit was unable to read the molecule due to the error: {e}")
        return None

    return mol


@beartype
def read_mols(
    dataset_dir: str,
    name: str,
    remove_hs: bool = False,
    sanitize: bool = True,
) -> List[Chem.Mol]:
    """Load RDKit `Mol` objects corresponding to a dataset's ligand name."""
    ligs = []
    for file in os.listdir(os.path.join(dataset_dir, name)):
        if file.endswith(".mol2") and "rdkit" not in file:
            lig = read_molecule(
                os.path.join(dataset_dir, name, file), remove_hs=remove_hs, sanitize=sanitize
            )
            if lig is None and os.path.exists(
                os.path.join(dataset_dir, name, file[:-4] + ".sdf")
            ):  # read sdf file if mol2 file cannot be sanitized
                log.info(
                    "Using the .mol2 file failed. We found a .sdf file instead and are trying to use that. Be aware that the .sdf files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
                )
                lig = read_molecule(
                    os.path.join(dataset_dir, name, file[:-4] + ".sdf"),
                    remove_hs=remove_hs,
                    sanitize=sanitize,
                )
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(lig)
            if lig is not None:
                ligs.append(lig)
    return ligs


@beartype
def parse_pdb_from_path(path: str) -> Model:
    """Load a BioPython structure from a given PDB filepath."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure("random_id", path)
        rec = structure[0]
    return rec


@beartype
def parse_and_read_moad_mols(
    dataset_path: str,
    pdb_id: str,
) -> Optional[List[Optional[Chem.Mol]]]:
    """Parse (to separate PDB files) and read ligands from a given PDB filepath, for the Binding
    MOAD dataset.

    :param dataset_path: Path to the directory containing protein-ligand complex PDB files.
    :param pdb_id: PDB ID of the protein-ligand complex.
    :return: Optional list of optional RDKit molecules.
    """
    lig_mols = []
    lig_pdb_files = glob.glob(
        os.path.join(dataset_path, "pdb_superligand", f"{pdb_id}_superlig_*.pdb")
    )
    if len(lig_pdb_files) > 0:
        for lig_pdb_file in lig_pdb_files:
            lig_mol = read_molecule(lig_pdb_file)
            lig_mols.append(RemoveHs(lig_mol) if lig_mol is not None else None)
    return lig_mols if len(lig_mols) else None


@beartype
def parse_and_read_dockgen_mols(
    dataset_path: str,
    pdb_id: str,
) -> Optional[List[Optional[Chem.Mol]]]:
    """Parse (to separate PDB files) and read ligands from a given PDB filepath, for the DockGen
    dataset.

    :param dataset_path: Path to the directory containing protein-ligand complex PDB files.
    :param pdb_id: PDB ID of the protein-ligand complex.
    :return: Optional list of optional RDKit molecules.
    """
    lig_mols = []
    lig_pdb_files = glob.glob(os.path.join(dataset_path, pdb_id, f"{pdb_id}_ligand*.pdb"))
    if len(lig_pdb_files) > 0:
        for lig_pdb_file in lig_pdb_files:
            lig_mol = read_molecule(lig_pdb_file)
            lig_mols.append(RemoveHs(lig_mol, sanitize=False) if lig_mol is not None else None)
    return lig_mols if len(lig_mols) else None


@beartype
def align_prediction(
    smoothing_factor: Optional[Union[float, np.ndarray]],
    dataset_calpha_coords: np.ndarray,
    esmfold_calpha_coords: np.ndarray,
    dataset_ligand_coords: Optional[np.ndarray],
    return_rotation: bool = False,
) -> Union[Tuple[Rotation, np.ndarray, np.ndarray], Union[float, np.ndarray]]:
    """Perform an alignment of apo and holo protein structures and (optionally) ligand coordinates
    using an (optionally) optimized smoothing factor.

    :param smoothing_factor: Smoothing factor controlling the alignment.
    :param dataset_calpha_coords: Array of Ca atom coordinates for a dataset's protein structure.
    :param esmfold_calpha_coords: Array of Ca atom coordinates for a dataset's protein structure.
    :param dataset_ligand_coords: Optional array of ligand coordinates from a dataset.
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


@beartype
def get_alignment_rotation(
    pdb_id: str,
    dataset_protein_path: str,
    esmfold_protein_path: str,
    dataset: str,
    dataset_path: str,
) -> Tuple[Optional[Rotation], Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculate the alignment rotation between apo and holo protein structures and their ligand
    coordinates.

    :param pdb_id: PDB ID of the protein-ligand complex.
    :param dataset_protein_path: Filepath to the PDB file of the protein structure from a dataset.
    :param esmfold_protein_path: Filepath to the PDB file of the protein structure from ESMFold.
    :param dataset: Name of the dataset.
    :param dataset_path: Filepath to the PDB file containing ligand coordinates.
    :return: A tuple containing rotation matrix (Optional[Rotation]), centroid of Ca atoms for a
        dataset protein (Optional[np.ndarray]), and centroid of Ca atoms for ESMFold
        (Optional[np.ndarray]).
    """
    try:
        dataset_protein = pdb_filepath_to_protein(dataset_protein_path)
        dataset_calpha_coords = dataset_protein.atom_positions[:, 1, :]
    except Exception as e:
        log.warning(
            f"Unable to parse dataset Ca protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None
    try:
        esmfold_protein = pdb_filepath_to_protein(esmfold_protein_path)
        esmfold_calpha_coords = esmfold_protein.atom_positions[:, 1, :]
    except Exception as e:
        log.warning(
            f"Unable to parse ESMFold Ca protein structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None

    if dataset == "pdbbind":
        try:
            dataset_ligand = read_mols(dataset_path, pdb_id, remove_hs=True, sanitize=True)[0]
        except Exception as e:
            try:
                log.warning(
                    f"Unable to parse PDBBind ligand structure for PDB ID {pdb_id} with sanitization. Trying to load without sanitization..."
                )
                dataset_ligand = read_mols(dataset_path, pdb_id, remove_hs=True, sanitize=False)[0]
            except Exception as e:
                log.warning(
                    f"Unable to parse PDBBind ligand structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
                )
                return None, None, None
    elif dataset in ["moad", "dockgen"]:
        try:
            dataset_ligands = (
                parse_and_read_dockgen_mols(
                    dataset_path=dataset_path,
                    pdb_id=pdb_id,
                )
                if dataset == "dockgen"
                else parse_and_read_moad_mols(
                    dataset_path=dataset_path,
                    pdb_id=pdb_id,
                )
            )
        except Exception as e:
            log.warning(
                f"Unable to parse {dataset} ligand structures for PDB ID {pdb_id} due to the error: {e}. Skipping..."
            )
            return None, None, None
        if len(dataset_ligands) > 0:
            try:
                dataset_ligand = combine_molecules(dataset_ligands)
            except Exception as e:
                log.warning(
                    f"Unable to combine {dataset} ligand structures for PDB ID {pdb_id} due to the error: {e}. Skipping..."
                )
                return None, None, None
        else:
            log.warning(
                f"Unable to parse any {dataset} ligand structures for PDB ID {pdb_id}. Skipping..."
            )
            return None, None, None
    elif dataset == "pdbsidechain":
        # NOTE: the van der Mers dataset does not contain artificial ligands initially
        dataset_ligand = None
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

    try:
        if dataset_ligand is not None:
            dataset_ligand_coords = dataset_ligand.GetConformer().GetPositions()
        else:
            dataset_ligand_coords = None
    except Exception as e:
        log.warning(
            f"Unable to extract dataset ligand structure for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None

    if dataset_calpha_coords.shape != esmfold_calpha_coords.shape:
        log.warning(
            f"Dataset and ESMFold protein structures differ for PDB ID {pdb_id}. Skipping due to shape mismatch: dataset {dataset_calpha_coords.shape} vs ESMFold {esmfold_calpha_coords.shape}",
        )
        return None, None, None

    try:
        if dataset_ligand_coords is not None:
            res = minimize(
                align_prediction,
                [0.1],
                bounds=Bounds([0.0], [1.0]),
                args=(dataset_calpha_coords, esmfold_calpha_coords, dataset_ligand_coords),
                tol=1e-8,
            )
            smoothing_factor = res.x
        else:
            smoothing_factor = 0.1

        rotation, dataset_calpha_centroid, esmfold_calpha_centroid = align_prediction(
            smoothing_factor,
            dataset_calpha_coords,
            esmfold_calpha_coords,
            dataset_ligand_coords,
            return_rotation=True,
        )
    except Exception as e:
        log.warning(
            f"Unable to align protein structures for PDB ID {pdb_id} due to the error: {e}. Skipping..."
        )
        return None, None, None

    return rotation, dataset_calpha_centroid, esmfold_calpha_centroid


@beartype
def align_apo_structure_to_holo_structure(
    cfg: DictConfig, filename: str, atom_df_name: str = "ATOM"
):
    """Align a given ESMFold apo structure to its corresponding holo structure.

    :param cfg: Hydra config for the alignment.
    :param filename: Filename of the ESMFold apo structure.
    :param atom_df_name: Name of the atom DataFrame derived from the corresponding PDB file input.
    """
    pdb_id = Path(filename.split("_")[0]).stem if cfg.dataset == "pdbbind" else Path(filename).stem
    esm_protein_filename = os.path.join(cfg.esmfold_structures_dir, f"{pdb_id}.pdb")
    if cfg.dataset == "pdbsidechain":
        processed_pt_filename = os.path.join(cfg.data_dir, pdb_id[1:3], f"{pdb_id}.pt")
        if not os.path.exists(processed_pt_filename):
            log.info(
                f"Unable to find processed protein chain structures for PDB ID {pdb_id}. Skipping..."
            )
            return
        processed_protein_filename = os.path.join(cfg.data_dir, pdb_id[1:3], f"{pdb_id}.pdb")
        if not os.path.exists(processed_protein_filename):
            try:
                convert_protein_pts_to_pdb([processed_pt_filename], processed_protein_filename)
            except Exception as e:
                log.warning(
                    f"Unable to convert protein chain structures to PDB format for PDB ID {pdb_id} due to the error: {e}. Skipping..."
                )
                return
        assert os.path.exists(
            processed_protein_filename
        ), f"Processed `pdbsidechain` file {processed_protein_filename} does not exist."
    else:
        processed_protein_name = (
            f"{pdb_id}_protein_processed.pdb"
            if cfg.dataset in ["pdbbind", "dockgen"]
            else f"{pdb_id}_protein.pdb"
        )
        processed_protein_filename = os.path.join(
            (os.path.join(cfg.data_dir, "pdb_protein") if cfg.dataset == "moad" else cfg.data_dir),
            (pdb_id if cfg.dataset in ["pdbbind", "dockgen"] else ""),
            processed_protein_name,
        )
    esm_protein_output_filename = os.path.join(
        cfg.output_dir, f"{pdb_id}_holo_aligned_esmfold_protein.pdb"
    )

    rotation, dataset_calpha_centroid, esmfold_calpha_centroid = get_alignment_rotation(
        pdb_id=pdb_id,
        dataset_protein_path=processed_protein_filename,
        esmfold_protein_path=esm_protein_filename,
        dataset=cfg.dataset,
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
    config_path="../../../configs/data",
    config_name="esmfold_apo_to_holo_alignment.yaml",
)
def main(cfg: DictConfig):
    """Align all ESMFold apo structures to their corresponding holo structures.

    :param cfg: Hydra config for the alignments.
    """
    with open_dict(cfg):
        if cfg.dataset == "pdbbind":
            cfg.data_dir = os.path.join(cfg.data_dir, cfg.dataset, "PDBBind_processed")
        elif cfg.dataset == "moad":
            cfg.data_dir = os.path.join(cfg.data_dir, cfg.dataset, "BindingMOAD_2020_processed")
        elif cfg.dataset == "dockgen":
            cfg.data_dir = os.path.join(cfg.data_dir, "DockGen", "processed_files")
        elif cfg.dataset == "pdbsidechain":
            cfg.data_dir = os.path.join(cfg.data_dir, cfg.dataset, "pdb_2021aug02", "pdb")
        else:
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
    register_custom_omegaconf_resolvers()
    main()
