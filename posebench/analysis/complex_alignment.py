# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import rootutils
from beartype.typing import Literal
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def align_to_binding_site(
    predicted_protein: str,
    predicted_ligand: Optional[str],
    reference_protein: str,
    reference_ligand: Optional[str],
    dataset: Literal["dockgen", "casp15", "posebusters_benchmark", "astex_diverse"],
    aligned_filename_suffix: str = "_aligned",
    cutoff: float = 10.0,
    save_protein: bool = True,
    save_ligand: bool = True,
    verbose: bool = True,
):
    """Align the predicted protein-ligand complex to the reference complex
    using the reference protein's heavy atom ligand binding site residues.

    :param predicted_protein: File path to the predicted protein (PDB).
    :param predicted_ligand: File path to the optional predicted ligand
        (SDF).
    :param reference_protein: File path to the reference protein (PDB).
    :param reference_ligand: File path to the optional reference ligand
        (SDF).
    :param dataset: Dataset name (e.g., "dockgen", "casp15",
        "posebusters_benchmark", or "astex_diverse").
    :param aligned_filename_suffix: Suffix to append to the aligned
        files (default "_aligned").
    :param cutoff: Distance cutoff in Å to define the binding site
        (default 10.0).
    :param save_protein: Whether to save the aligned protein structure
        (default True).
    :param save_ligand: Whether to save the aligned ligand structure
        (default True).
    :param verbose: Whether to print the alignment RMSD and number of
        aligned atoms (default True).
    """
    from pymol import cmd

    reference_target = os.path.splitext(os.path.basename(reference_protein))[0].split("_protein")[
        0
    ]
    prediction_target = os.path.basename(os.path.dirname(predicted_protein))

    # Refresh PyMOL
    cmd.reinitialize()

    # Load structures
    cmd.load(reference_protein, "ref_protein")
    cmd.load(predicted_protein, "pred_protein")

    if reference_ligand is not None:
        cmd.load(reference_ligand, "ref_ligand")
    elif dataset == "casp15":
        # Select the ligand chain(s) in the reference protein PDB file
        cmd.select("ref_ligand", "ref_protein and not polymer")

    if predicted_ligand is not None:
        cmd.load(predicted_ligand, "pred_ligand")

    # Group predicted protein and ligand(s) together for alignment
    cmd.create(
        "pred_complex",
        ("pred_protein or pred_ligand" if predicted_ligand is not None else "pred_protein"),
    )

    # Select heavy atoms in the reference protein
    cmd.select("ref_protein_heavy", "ref_protein and not elem H")

    # Select heavy atoms in the reference ligand(s)
    cmd.select("ref_ligand_heavy", "ref_ligand and not elem H")

    # Define the reference binding site(s) based on the reference ligand(s)
    cmd.select("binding_site", f"ref_protein_heavy within {cutoff} of ref_ligand_heavy")

    # Align the predicted protein to the reference binding site(s)
    align_cmd = cmd.super if dataset == "dockgen" else cmd.align
    # NOTE: Since with DockGen we are aligning full predicted bioassemblies
    # to primary interacting chains, we instead use the `super` command to align
    # since it is more robust to large quaternary sequence differences
    alignment_result = align_cmd("pred_complex", "binding_site")

    # Report alignment RMSD and number of aligned atoms
    if verbose:
        logger.info(
            f"Alignment RMSD for {reference_target} with {alignment_result[1]} aligned atoms: {alignment_result[0]:.3f} Å"
        )

    # Apply the transformation to the individual objects
    cmd.matrix_copy("pred_complex", "pred_protein")
    cmd.matrix_copy("pred_complex", "pred_ligand")

    # # Maybe prepare to visualize the computed alignments
    # import shutil
    # assert (
    #     reference_target == prediction_target
    # ), f"Reference target {reference_target} does not match prediction target {prediction_target}"
    # complex_alignment_viz_dir = os.path.join("complex_alignment_viz", reference_target)
    # os.makedirs(complex_alignment_viz_dir, exist_ok=True)

    # Save the aligned protein
    if save_protein:
        cmd.save(
            predicted_protein.replace(".pdb", f"{aligned_filename_suffix}.pdb"),
            "pred_protein",
        )

        # # Maybe visualize the computed protein alignments
        # cmd.save(
        #     os.path.join(
        #         complex_alignment_viz_dir,
        #         os.path.basename(predicted_protein).replace(
        #             ".pdb", f"{aligned_filename_suffix}.pdb"
        #         ),
        #     ),
        #     "pred_protein",
        # )
        # shutil.copyfile(
        #     reference_protein,
        #     os.path.join(
        #         complex_alignment_viz_dir, os.path.basename(reference_protein)
        #     ),
        # )

    # Save the aligned ligand
    if save_ligand and predicted_ligand is not None:
        cmd.save(
            predicted_ligand.replace(".sdf", f"{aligned_filename_suffix}.sdf"),
            "pred_ligand",
        )

        # # Maybe visualize the computed ligand alignments
        # cmd.save(
        #     os.path.join(
        #         complex_alignment_viz_dir,
        #         os.path.basename(predicted_ligand).replace(
        #             ".sdf", f"{aligned_filename_suffix}.sdf"
        #         ),
        #     ),
        #     "pred_ligand",
        # )
        # shutil.copyfile(
        #     reference_ligand,
        #     os.path.join(complex_alignment_viz_dir, os.path.basename(reference_ligand)),
        # )


def align_complex_to_protein_only(
    predicted_protein_pdb: str,
    predicted_ligand_sdf: Optional[str],
    reference_protein_pdb: str,
    save_protein: bool = True,
    save_ligand: bool = True,
    aligned_filename_suffix: str = "_aligned",
    atom_df_name: str = "ATOM",
) -> int:
    """Align a predicted protein-ligand structure to a reference protein
    structure.

    :param predicted_protein_pdb: Path to the predicted protein
        structure in PDB format
    :param predicted_ligand_sdf: Optional path to the predicted ligand
        structure in SDF format
    :param reference_protein_pdb: Path to the reference protein
        structure in PDB format
    :param save_protein: Whether to save the aligned protein structure
    :param save_ligand: Whether to save the aligned ligand structure
    :param aligned_filename_suffix: suffix to append to the aligned
        files
    :param atom_df_name: Name of the atom dataframe in the PDB file
    :return: 0 if successful, 1 if unsuccessful
    """
    from biopandas.pdb import PandasPdb
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    from posebench.data.components.protein_apo_to_holo_alignment import (
        align_prediction,
        extract_receptor_structure,
        parse_pdb_from_path,
        read_molecule,
    )

    # Load protein and ligand structures
    try:
        predicted_rec = parse_pdb_from_path(predicted_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse predicted protein structure {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return 1
    try:
        reference_rec = parse_pdb_from_path(reference_protein_pdb)
    except Exception as e:
        logger.warning(
            f"Unable to parse reference protein structure {reference_protein_pdb} due to the error: {e}. Skipping..."
        )
        return 1
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
        return 1
    try:
        reference_calpha_coords = extract_receptor_structure(
            reference_rec, None, filter_out_hetero_residues=True
        )[2]
    except Exception as e:
        logger.warning(
            f"Unable to extract reference protein structure coordinates for input {predicted_protein_pdb} due to the error: {e}. Skipping..."
        )
        return 1
    if predicted_ligand_sdf is not None:
        try:
            predicted_ligand_conf = predicted_ligand.GetConformer()
        except Exception as e:
            logger.warning(
                f"Unable to extract predicted ligand conformer for {predicted_ligand_sdf} due to the error: {e}. Skipping..."
            )
            return 1

    if reference_calpha_coords.shape != predicted_calpha_coords.shape:
        logger.warning(
            f"Receptor structures differ for prediction {predicted_protein_pdb}. Skipping due to shape mismatch:",
            reference_calpha_coords.shape,
            predicted_calpha_coords.shape,
        )
        return 1

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
            path=predicted_protein_pdb.replace(".pdb", f"{aligned_filename_suffix}.pdb"),
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
            predicted_ligand_sdf.replace(".sdf", f"{aligned_filename_suffix}.sdf")
        ) as f:
            f.write(predicted_ligand)

    return 0


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="complex_alignment.yaml",
)
def main(cfg: DictConfig):
    """Align the predicted protein-ligand structures to the reference protein-
    ligand structures.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    with open_dict(cfg):
        # NOTE: besides their output directories, single-sequence baselines are treated like their multi-sequence counterparts
        output_dir = copy.deepcopy(cfg.output_dir)
        cfg.method = cfg.method.removesuffix("_ss")
        cfg.output_dir = output_dir

    input_data_dir = Path(cfg.input_data_dir)
    for config in ["", "_relaxed"]:
        output_dir = Path(cfg.output_dir + config)
        if not output_dir.exists() or cfg.method in [
            "neuralplexer",
            "flowdock",
            "rfaa",
            "chai-lab",
            "boltz",
            "alphafold3",
        ]:
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
        elif cfg.method in ["neuralplexer", "flowdock"]:
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
        elif cfg.method == "chai-lab":
            output_ligand_files = list(
                output_dir.rglob(f"pred.model_idx_{cfg.rank_to_align - 1}_ligand*{config}.sdf")
            )
            output_ligand_files = sorted(
                [
                    file
                    for file in output_ligand_files
                    if config == "_relaxed"
                    or (config == "" and "_relaxed" not in file.stem)
                    and "_aligned" not in file.stem
                    and "_LIG_" not in file.stem
                ]
            )
        elif cfg.method == "boltz":
            output_ligand_files = list(
                output_dir.rglob(f"*_model_{cfg.rank_to_align - 1}_ligand*{config}.sdf")
            )
            output_ligand_files = sorted(
                [
                    file
                    for file in output_ligand_files
                    if config == "_relaxed"
                    or (config == "" and "_relaxed" not in file.stem)
                    and "_aligned" not in file.stem
                    and "_LIG" not in file.stem
                ]
            )
        elif cfg.method == "alphafold3":
            output_ligand_files = list(output_dir.rglob(f"*_model_ligand{config}.sdf"))
            output_ligand_files = sorted(
                [
                    file
                    for file in output_ligand_files
                    if config == "_relaxed"
                    or (config == "" and "_relaxed" not in file.stem)
                    and "_aligned" not in file.stem
                    and "_LIG_" not in file.stem
                ]
            )
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
        elif cfg.method in ["neuralplexer", "flowdock"]:
            output_protein_files = sorted(
                [
                    file
                    for file in list(output_dir.rglob(f"prot_rank{cfg.rank_to_align}_*.pdb"))
                    if "_aligned" not in file.stem
                ]
            )
        elif cfg.method == "rfaa":
            output_protein_files = sorted(list(output_dir.rglob("*protein.pdb")))
        elif cfg.method == "chai-lab":
            output_protein_files = list(
                output_dir.rglob(f"pred.model_idx_{cfg.rank_to_align - 1}_protein*.pdb")
            )
            output_protein_files = sorted(
                [
                    file
                    for file in output_protein_files
                    if (config == "_relaxed" or (config == "" and "_relaxed" not in file.stem))
                    and "_aligned" not in file.stem
                ]
            )
        elif cfg.method == "boltz":
            output_protein_files = list(
                output_dir.rglob(f"*_model_{cfg.rank_to_align - 1}_protein*.pdb")
            )
            output_protein_files = sorted(
                [
                    file
                    for file in output_protein_files
                    if (config == "_relaxed" or (config == "" and "_relaxed" not in file.stem))
                    and "_aligned" not in file.stem
                ]
            )
        elif cfg.method == "alphafold3":
            output_protein_files = list(output_dir.rglob("*_model_protein.pdb"))
            output_protein_files = sorted(
                [
                    file
                    for file in output_protein_files
                    if (config == "_relaxed" or (config == "" and "_relaxed" not in file.stem))
                    and "_aligned" not in file.stem
                ]
            )
        else:
            raise ValueError(f"Invalid method: {cfg.method}")

        if len(output_ligand_files) < len(output_protein_files):
            if cfg.method in ["neuralplexer", "flowdock"]:
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
            elif cfg.method == "chai-lab":
                output_protein_files = sorted(
                    [
                        item
                        for item in output_dir.rglob(
                            f"pred.model_idx_{cfg.rank_to_align - 1}_protein*.pdb"
                        )
                        if "_aligned" not in item.stem
                        and any(
                            [item.parent.stem in file.parent.stem for file in output_ligand_files]
                        )
                    ]
                )
            elif cfg.method == "boltz":
                output_protein_files = sorted(
                    [
                        item
                        for item in output_dir.rglob(
                            f"*_model_{cfg.rank_to_align - 1}_protein*.pdb"
                        )
                        if "_aligned" not in item.stem
                        and any(
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
        for protein_file, ligand_file in tqdm(
            zip(output_protein_files, output_ligand_files), desc="Aligning complexes"
        ):
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
                protein_id, ligand_id = (
                    protein_file.parent.stem,
                    ligand_file.parent.stem,
                )
            if protein_id != ligand_id:
                raise ValueError(f"Protein and ligand IDs do not match: {protein_id}, {ligand_id}")
            pocket_suffix = "_bs_cropped" if cfg.pocket_only_baseline else ""
            reference_protein_pdbs = [
                item
                for item in input_data_dir.rglob(
                    f"*{protein_id.split(f'{cfg.dataset}_')[-1]}{'_lig.pdb' if cfg.dataset == 'casp15' else f'*_protein{pocket_suffix}.pdb'}"
                )
                if "predicted_structures" not in str(item)
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
                    str(protein_file).replace(".pdb", f"{cfg.aligned_filename_suffix}.pdb")
                )
                or not os.path.exists(
                    str(ligand_file).replace(".sdf", f"{cfg.aligned_filename_suffix}.sdf")
                )
            ):
                align_to_binding_site(
                    predicted_protein=str(protein_file),
                    predicted_ligand=str(ligand_file),
                    reference_protein=str(reference_protein_pdb),
                    reference_ligand=(
                        None if cfg.dataset == "casp15" else str(reference_ligand_sdf)
                    ),
                    dataset=cfg.dataset,
                    aligned_filename_suffix=cfg.aligned_filename_suffix,
                    save_protein=cfg.method not in ("diffdock", "fabind"),
                )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
