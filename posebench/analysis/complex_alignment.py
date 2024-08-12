# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import rootutils
from biopandas.pdb import PandasPdb
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.optimize import Bounds, minimize
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers
from posebench.data.components.esmfold_apo_to_holo_alignment import (
    align_prediction,
    extract_receptor_structure,
    parse_pdb_from_path,
    read_molecule,
)

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_aligned_complex(
    predicted_protein_pdb: str,
    predicted_ligand_sdf: Optional[str],
    reference_protein_pdb: str,
    reference_ligand_sdf: str,
    save_protein: bool = True,
    save_ligand: bool = True,
    aligned_filename_postfix: str = "_aligned",
    atom_df_name: str = "ATOM",
):
    """Align the predicted protein-ligand structures to the reference protein-ligand structures and
    save the aligned results.

    :param predicted_protein_pdb: Path to the predicted protein structure in PDB format
    :param predicted_ligand_sdf: Optional path to the predicted ligand structure in SDF format
    :param reference_protein_pdb: Path to the reference protein structure in PDB format
    :param reference_ligand_sdf: Path to the reference ligand structure in SDF format
    :param save_protein: Whether to save the aligned protein structure
    :param save_ligand: Whether to save the aligned ligand structure
    :param aligned_filename_postfix: Postfix to append to the aligned files
    :param atom_df_name: Name of the atom dataframe in the PDB file
    """
    # Load protein and ligand structures
    try:
        predicted_rec = parse_pdb_from_path(predicted_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse predicted protein structure {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    try:
        reference_rec = parse_pdb_from_path(reference_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse reference protein structure {reference_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    if predicted_ligand_sdf is not None:
        predicted_ligand = read_molecule(predicted_ligand_sdf, remove_hs=True, sanitize=True)
        if predicted_ligand is None:
            predicted_ligand = read_molecule(predicted_ligand_sdf, remove_hs=True, sanitize=False)
    reference_ligand = read_molecule(reference_ligand_sdf, remove_hs=True, sanitize=True)
    if reference_ligand is None:
        reference_ligand = read_molecule(reference_ligand_sdf, remove_hs=True, sanitize=False)
    try:
        predicted_calpha_coords = extract_receptor_structure(
            predicted_rec, reference_ligand, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract predicted protein structure coordinates for input {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    try:
        reference_calpha_coords = extract_receptor_structure(
            reference_rec, reference_ligand, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract reference protein structure coordinates for input {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    if predicted_ligand_sdf is not None:
        try:
            predicted_ligand_conf = predicted_ligand.GetConformer()
        except Exception as e:
            logger.warning(
                f"Unable to extract predicted ligand conformer for {predicted_ligand_sdf} due to the error: {e}. Skipping..."
            )
            return
    try:
        reference_ligand_coords = reference_ligand.GetConformer().GetPositions()
    except Exception as e:
        logger.warning(
            f"Unable to extract reference ligand structure for {reference_ligand_sdf} due to the error: {e}. Skipping..."
        )
        return

    if reference_calpha_coords.shape != predicted_calpha_coords.shape:
        logger.warning(
            f"Receptor structures differ for prediction {predicted_protein_pdb}. Skipping due to shape mismatch:",
            reference_calpha_coords.shape,
            predicted_calpha_coords.shape,
        )
        return

    # Optimize the alignment
    res = minimize(
        align_prediction,
        [0.1],
        bounds=Bounds([0.0], [1.0]),
        args=(reference_calpha_coords, predicted_calpha_coords, reference_ligand_coords),
        tol=1e-8,
    )
    smoothing_factor = res.x
    rotation, reference_calpha_centroid, predicted_calpha_centroid = align_prediction(
        smoothing_factor,
        reference_calpha_coords,
        predicted_calpha_coords,
        reference_ligand_coords,
        return_rotation=True,
    )

    # Transform and record protein
    predicted_protein = PandasPdb().read_pdb(predicted_protein_pdb)
    predicted_protein_pre_rot = (
        predicted_protein.df[atom_df_name][["x_coord", "y_coord", "z_coord"]]
        .to_numpy()
        .squeeze()
        .astype(np.float32)
    )
    predicted_protein_aligned = (
        rotation.apply(predicted_protein_pre_rot - predicted_calpha_centroid)
        + reference_calpha_centroid
    )
    predicted_protein.df[atom_df_name][
        ["x_coord", "y_coord", "z_coord"]
    ] = predicted_protein_aligned
    if save_protein:
        predicted_protein.to_pdb(
            path=predicted_protein_pdb.replace(".pdb", f"{aligned_filename_postfix}.pdb"),
            records=[atom_df_name],
            gz=False,
        )

    # Transform and record ligand
    if predicted_ligand_sdf is not None:
        predicted_ligand_aligned = (
            rotation.apply(predicted_ligand_conf.GetPositions() - predicted_calpha_centroid)
            + reference_calpha_centroid
        )
        for i in range(predicted_ligand.GetNumAtoms()):
            x, y, z = predicted_ligand_aligned[i]
            predicted_ligand_conf.SetAtomPosition(i, Point3D(x, y, z))
        if save_ligand:
            with Chem.SDWriter(
                predicted_ligand_sdf.replace(".sdf", f"{aligned_filename_postfix}.sdf")
            ) as f:
                f.write(predicted_ligand)


def align_complex_to_protein_only(
    predicted_protein_pdb: str,
    predicted_ligand_sdf: Optional[str],
    reference_protein_pdb: str,
    save_protein: bool = True,
    save_ligand: bool = True,
    aligned_filename_postfix: str = "_aligned",
    atom_df_name: str = "ATOM",
):
    """Align a predicted protein-ligand structure to a reference protein structure.

    :param predicted_protein_pdb: Path to the predicted protein structure in PDB format
    :param predicted_ligand_sdf: Optional path to the predicted ligand structure in SDF format
    :param reference_protein_pdb: Path to the reference protein structure in PDB format
    :param save_protein: Whether to save the aligned protein structure
    :param save_ligand: Whether to save the aligned ligand structure
    :param aligned_filename_postfix: Postfix to append to the aligned files
    :param atom_df_name: Name of the atom dataframe in the PDB file
    """
    # Load protein and ligand structures
    try:
        predicted_rec = parse_pdb_from_path(predicted_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse predicted protein structure {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    try:
        reference_rec = parse_pdb_from_path(reference_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse reference protein structure {reference_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    if predicted_ligand_sdf is not None:
        predicted_ligand = read_molecule(predicted_ligand_sdf, remove_hs=True, sanitize=True)
    try:
        predicted_calpha_coords = extract_receptor_structure(
            predicted_rec, None, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract predicted protein structure coordinates for input {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    try:
        reference_calpha_coords = extract_receptor_structure(
            reference_rec, None, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract reference protein structure coordinates for input {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return
    if predicted_ligand_sdf is not None:
        try:
            predicted_ligand_conf = predicted_ligand.GetConformer()
        except Exception as e:
            logger.warning(
                f"Unable to extract predicted ligand conformer for {predicted_ligand_sdf} due to the error: {e}. Skipping..."
            )
            return

    if reference_calpha_coords.shape != predicted_calpha_coords.shape:
        logger.warning(
            f"Receptor structures differ for prediction {predicted_protein_pdb}. Skipping due to shape mismatch:",
            reference_calpha_coords.shape,
            predicted_calpha_coords.shape,
        )
        return

    # Perform the alignment
    rotation, reference_calpha_centroid, predicted_calpha_centroid = align_prediction(
        None,
        reference_calpha_coords,
        predicted_calpha_coords,
        None,
        return_rotation=True,
    )

    # Transform and record protein
    if save_protein:
        predicted_protein = PandasPdb().read_pdb(predicted_protein_pdb)
        predicted_protein_pre_rot = (
            predicted_protein.df[atom_df_name][["x_coord", "y_coord", "z_coord"]]
            .to_numpy()
            .squeeze()
            .astype(np.float32)
        )
        predicted_protein_aligned = (
            rotation.apply(predicted_protein_pre_rot - predicted_calpha_centroid)
            + reference_calpha_centroid
        )
        predicted_protein.df[atom_df_name][
            ["x_coord", "y_coord", "z_coord"]
        ] = predicted_protein_aligned
        predicted_protein.to_pdb(
            path=predicted_protein_pdb.replace(".pdb", f"{aligned_filename_postfix}.pdb"),
            records=[atom_df_name],
            gz=False,
        )

    # Transform and record ligand
    if predicted_ligand_sdf is not None and save_ligand:
        predicted_ligand_aligned = (
            rotation.apply(predicted_ligand_conf.GetPositions() - predicted_calpha_centroid)
            + reference_calpha_centroid
        )
        for i in range(predicted_ligand.GetNumAtoms()):
            x, y, z = predicted_ligand_aligned[i]
            predicted_ligand_conf.SetAtomPosition(i, Point3D(x, y, z))
        with Chem.SDWriter(
            predicted_ligand_sdf.replace(".sdf", f"{aligned_filename_postfix}.sdf")
        ) as f:
            f.write(predicted_ligand)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="complex_alignment.yaml",
)
def main(cfg: DictConfig):
    """Align the predicted protein-ligand structures to the reference protein-ligand structures.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    input_data_dir = Path(cfg.input_data_dir)
    for config in ["", "_relaxed"]:
        output_dir = Path(cfg.output_dir + config)
        if not output_dir.exists() or cfg.method in ["neuralplexer", "rfaa"]:
            output_dir = Path(str(output_dir).replace("_relaxed", ""))

        # parse ligand files
        if cfg.method == "diffdock":
            output_ligand_files = sorted(
                list(output_dir.rglob(f"*rank{cfg.rank_to_align}_confidence*{config}.sdf"))
            )
        elif cfg.method == "dynamicbind":
            output_ligand_files = sorted(
                list(
                    output_dir.parent.rglob(
                        f"{output_dir.stem}*{os.sep}index0_idx_0{os.sep}*rank{cfg.rank_to_align}_ligand*{config}.sdf"
                    )
                )
            )
        elif cfg.method == "neuralplexer":
            output_ligand_files = list(
                output_dir.rglob(f"lig_rank{cfg.rank_to_align}_*{config}.sdf")
            )
            output_ligand_files = sorted(
                [
                    file
                    for file in output_ligand_files
                    if config == "_relaxed"
                    or (config == "" and "_relaxed" not in file.stem)
                    and "_aligned" not in file.stem
                ]
            )
        elif cfg.method == "rfaa":
            output_ligand_files = sorted(list(output_dir.rglob(f"*ligand{config}.sdf")))
        else:
            raise ValueError(f"Invalid method: {cfg.method}")

        # parse protein files
        if cfg.method == "diffdock":
            output_protein_files = sorted(
                list((Path(cfg.input_data_dir).parent / "ensemble_proteins").rglob("*.pdb"))
            )
            output_protein_files = sorted(
                [
                    file
                    for file in output_protein_files
                    if file.stem in [item.parent.stem for item in output_ligand_files]
                ]
            )
        elif cfg.method == "dynamicbind":
            output_protein_files = sorted(
                list(
                    output_dir.parent.rglob(
                        f"{output_dir.stem}*{os.sep}index0_idx_0{os.sep}*rank{cfg.rank_to_align}_receptor*{config}.pdb"
                    )
                )
            )
        elif cfg.method == "neuralplexer":
            output_protein_files = sorted(
                [
                    file
                    for file in list(output_dir.rglob(f"prot_rank{cfg.rank_to_align}_*.pdb"))
                    if "_aligned" not in file.stem
                ]
            )
        elif cfg.method == "rfaa":
            output_protein_files = sorted(list(output_dir.rglob("*protein.pdb")))
        else:
            raise ValueError(f"Invalid method: {cfg.method}")

        if len(output_ligand_files) < len(output_protein_files):
            if cfg.method == "neuralplexer":
                output_protein_files = sorted(
                    [
                        file
                        for file in list(output_dir.rglob(f"prot_rank{cfg.rank_to_align}_*.pdb"))
                        if "_aligned" not in file.stem
                        and any(
                            [file.parent.stem in item.parent.stem for item in output_ligand_files]
                        )
                    ]
                )
            elif cfg.method == "rfaa":
                output_protein_files = sorted(
                    [
                        item
                        for item in output_dir.rglob("*protein.pdb")
                        if any(
                            [item.parent.stem in file.parent.stem for file in output_ligand_files]
                        )
                    ]
                )
            else:
                raise ValueError(
                    f"Number of protein files ({len(output_protein_files)}) is less than the number of ligand files ({len(output_ligand_files)})."
                )
        assert len(output_protein_files) == len(
            output_ligand_files
        ), f"Numbers of protein ({len(output_protein_files)}) and ligand ({len(output_ligand_files)}) files do not match."

        # align protein-ligand complexes
        for protein_file, ligand_file in tqdm(zip(output_protein_files, output_ligand_files)):
            protein_id, ligand_id = protein_file.stem, ligand_file.stem
            if protein_id != ligand_id and cfg.method == "dynamicbind":
                protein_id, ligand_id = (
                    protein_file.parent.parent.stem,
                    ligand_file.parent.parent.stem,
                )
            if protein_id != ligand_id and cfg.method == "rfaa":
                protein_id, ligand_id = protein_file.stem.replace(
                    "_protein", ""
                ), ligand_file.stem.replace("_ligand", "")
            if protein_id != ligand_id:
                protein_id, ligand_id = protein_file.stem, ligand_file.parent.stem
            if protein_id != ligand_id:
                protein_id, ligand_id = protein_file.parent.stem, ligand_file.parent.stem
            if protein_id != ligand_id:
                raise ValueError(f"Protein and ligand IDs do not match: {protein_id}, {ligand_id}")
            pocket_postfix = "_bs_cropped" if cfg.pocket_only_baseline else ""
            reference_protein_pdbs = [
                item
                for item in input_data_dir.rglob(
                    f"*{protein_id.split(f'{cfg.dataset}_')[-1]}{'_lig.pdb' if cfg.dataset == 'casp15' else f'*_protein{pocket_postfix}.pdb'}"
                )
                if "esmfold_structures" not in str(item)
            ]
            if cfg.dataset == "dockgen":
                reference_protein_pdbs = [
                    item
                    for item in (input_data_dir / protein_id).rglob(
                        f"{protein_id}_protein_processed.pdb"
                    )
                ]
                reference_ligand_sdfs = [
                    item
                    for item in (input_data_dir / protein_id).rglob(f"{protein_id}_ligand.pdb")
                ]
            elif cfg.dataset == "casp15":
                reference_protein_pdbs = [None]
            else:
                reference_ligand_sdfs = [
                    item
                    for item in input_data_dir.rglob(
                        f"*{ligand_id.split(f'{cfg.dataset}_')[-1]}*_ligand.sdf"
                    )
                ]
            assert (
                len(reference_protein_pdbs) == 1
            ), f"Expected 1 reference protein PDB file, but found {len(reference_protein_pdbs)}."
            assert (
                len(reference_ligand_sdfs) == 1
            ), f"Expected 1 reference ligand SDF file, but found {len(reference_ligand_sdfs)}."
            reference_protein_pdb, reference_ligand_sdf = (
                reference_protein_pdbs[0],
                reference_ligand_sdfs[0],
            )
            if (
                cfg.force_process
                or not os.path.exists(
                    str(protein_file).replace(".pdb", f"{cfg.aligned_filename_postfix}.pdb")
                )
                or not os.path.exists(
                    str(ligand_file).replace(".sdf", f"{cfg.aligned_filename_postfix}.sdf")
                )
            ):
                if cfg.dataset == "casp15":
                    # NOTE: for the CASP15 set, it is not trivial to separate a protein from the ligand chains in a given complex,
                    # so here we instead align the predicted protein-ligand complex to the reference protein of the complex
                    align_complex_to_protein_only(
                        str(protein_file),
                        str(ligand_file),
                        str(reference_protein_pdb),
                        save_protein=cfg.method != "diffdock",
                        aligned_filename_postfix=cfg.aligned_filename_postfix,
                    )
                else:
                    save_aligned_complex(
                        str(protein_file),
                        str(ligand_file),
                        str(reference_protein_pdb),
                        str(reference_ligand_sdf),
                        save_protein=cfg.method != "diffdock",
                        aligned_filename_postfix=cfg.aligned_filename_postfix,
                    )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
