import glob
import os
import tempfile
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rootutils
from beartype import beartype
from beartype.typing import List
from Bio.PDB import PDBParser
from omegaconf import DictConfig, open_dict
from rdkit import Chem
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock import register_custom_omegaconf_resolvers
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import load_hetero_data_graph_as_pdb_file_pair
from flowdock.utils.metric_utils import calculate_usalign_metrics

log = RankedLogger(__name__, rank_zero_only=True)


@beartype
def remove_duplicate_filenames(file_paths: List[str]) -> List[str]:
    """Remove duplicate filenames (not filepaths) from a list of file paths.

    :param file_paths: List of file paths.
    :return: List of file paths with duplicate filenames removed.
    """
    filename_dict = {}
    for path in file_paths:
        filename = os.path.split(path)[1]
        if filename not in filename_dict:
            filename_dict[filename] = path
    return list(filename_dict.values())


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


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/data",
    config_name="esmfold_apo_to_holo_assessment.yaml",
)
@beartype
def main(cfg: DictConfig):
    """Assessment the alignments between all ESMFold apo structures and their corresponding holo
    structures while also collecting ligand statistics.

    :param cfg: Hydra config for the assessment.
    """
    with open_dict(cfg):
        if cfg.dataset == "pdbbind":
            cfg.data_dir = os.path.join(cfg.data_dir, "pdbbind", "PDBBind_processed")
        elif cfg.dataset == "moad":
            cfg.data_dir = os.path.join(cfg.data_dir, "moad", "BindingMOAD_2020_processed")
        elif cfg.dataset == "dockgen":
            cfg.data_dir = os.path.join(cfg.data_dir, "DockGen", "processed_files")
        elif cfg.dataset == "pdbsidechain":
            cfg.data_dir = os.path.join(cfg.data_dir, "pdbsidechain", "pdb_2021aug02", "pdb")
        else:
            raise ValueError(f"Dataset {cfg.dataset} is not supported.")
    assert os.path.exists(cfg.data_dir), f"Data directory {cfg.data_dir} does not exist."
    use_preprocessed_graphs = cfg.preprocessed_graph_dirs is not None and all(
        os.path.exists(dir) for dir in cfg.preprocessed_graph_dirs
    )
    if use_preprocessed_graphs:
        structure_file_inputs = []
        for dir in cfg.preprocessed_graph_dirs:
            heterograph_filepaths = sorted(glob.glob(os.path.join(dir, "heterographs*.pkl")))
            rdkit_ligand_filepaths = sorted(glob.glob(os.path.join(dir, "rdkit_ligands*.pkl")))
            for heterograph_filepath, rdkit_ligand_filepath in zip(
                heterograph_filepaths, rdkit_ligand_filepaths
            ):
                assert (
                    Path(heterograph_filepath).stem.split("heterographs")[1]
                    == Path(rdkit_ligand_filepath).stem.split("rdkit_ligands")[1]
                ), f"File indices do not match for protein-ligand file pair: {heterograph_filepath} and {rdkit_ligand_filepath}"
                structure_file_inputs.append(((heterograph_filepath, ""), [rdkit_ligand_filepath]))
        temp_dir_path = Path(tempfile.mkdtemp(prefix="apo_holo_alignment_assessment_pdbs_"))
    else:
        if cfg.dataset == "pdbbind":
            structure_file_inputs = [
                (
                    (
                        os.path.join(cfg.aligned_dir, f"{id}_holo_aligned_esmfold_protein.pdb"),
                        os.path.join(cfg.data_dir, id, f"{id}_protein_processed.pdb"),
                    ),
                    [os.path.join(cfg.data_dir, id, f"{id}_ligand.sdf")],
                )
                for id in os.listdir(cfg.data_dir)
                if os.path.exists(
                    os.path.join(cfg.aligned_dir, f"{id}_holo_aligned_esmfold_protein.pdb")
                )
                and (
                    os.path.exists(os.path.join(cfg.data_dir, id, f"{id}_ligand.sdf"))
                    or os.path.exists(os.path.join(cfg.data_dir, id, f"{id}_ligand.mol2"))
                )
            ]
        elif cfg.dataset == "moad":
            structure_file_inputs = []
            for filename in os.listdir(Path(cfg.data_dir) / "pdb_protein"):
                ligand_file_inputs = [
                    os.path.join(
                        cfg.data_dir,
                        "pdb_superligand",
                        f"{Path(filename).stem[:6]}_superlig_{i}.pdb",
                    )
                    for i in range(10)
                    if os.path.exists(
                        os.path.join(
                            cfg.data_dir,
                            "pdb_superligand",
                            f"{Path(filename).stem[:6]}_superlig_{i}.pdb",
                        )
                    )
                ]
                if os.path.exists(
                    os.path.join(
                        cfg.aligned_dir,
                        f"{Path(filename).stem[:6]}_holo_aligned_esmfold_protein.pdb",
                    )
                ) and len(ligand_file_inputs):
                    structure_file_inputs.append(
                        (
                            (
                                os.path.join(
                                    cfg.aligned_dir,
                                    f"{Path(filename).stem[:6]}_holo_aligned_esmfold_protein.pdb",
                                ),
                                os.path.join(
                                    cfg.data_dir,
                                    "pdb_protein",
                                    f"{Path(filename).stem[:6]}_protein.pdb",
                                ),
                            ),
                            ligand_file_inputs,
                        )
                    )
        elif cfg.dataset == "dockgen":
            structure_file_inputs = []
            for item in os.listdir(cfg.data_dir):
                ligand_file_inputs = list(
                    glob.glob(
                        os.path.join(
                            cfg.data_dir,
                            item,
                            f"{item}_ligand*.pdb",
                        )
                    )
                )
                if os.path.exists(
                    os.path.join(
                        cfg.aligned_dir,
                        f"{item}_holo_aligned_esmfold_protein.pdb",
                    )
                ) and len(ligand_file_inputs):
                    structure_file_inputs.append(
                        (
                            (
                                os.path.join(
                                    cfg.aligned_dir,
                                    f"{item}_holo_aligned_esmfold_protein.pdb",
                                ),
                                os.path.join(
                                    cfg.data_dir,
                                    item,
                                    f"{item}_protein_processed.pdb",
                                ),
                            ),
                            ligand_file_inputs,
                        )
                    )
        elif cfg.dataset == "pdbsidechain":
            structure_file_inputs = []
            for id in os.listdir(cfg.data_dir):
                for filename in os.listdir(os.path.join(cfg.data_dir, id)):
                    full_id = "_".join(Path(filename).stem.split("_")[:2])
                    if filename.endswith(".pdb") and os.path.exists(
                        os.path.join(
                            cfg.aligned_dir, f"{full_id}_holo_aligned_esmfold_protein.pdb"
                        )
                    ):
                        structure_file_inputs.append(
                            (
                                (
                                    os.path.join(
                                        cfg.aligned_dir,
                                        f"{full_id}_holo_aligned_esmfold_protein.pdb",
                                    ),
                                    os.path.join(cfg.data_dir, id, filename),
                                ),
                                [],
                            )
                        )
    flags = [
        "-mol",
        "prot",  # note: align only protein chains
        "-ter",
        "0",  # note: biological unit alignment
        "-split",
        "0",  # note: treat a whole file as a single chain
        "-se",  # note: calculate TMscore without alignment
    ]
    (
        valid_file_inputs,
        erroneous_file_inputs,
        tm_scores,
        rmsds,
        apo_lengths,
        holo_lengths,
        ligands_num_atoms,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for file_input in tqdm(
        structure_file_inputs,
        desc="Calculating structural metrics (as well as ligand statistics) for holo-aligned apo and holo structure pairs",
    ):
        try:
            flat_file_input = (*file_input[0], file_input[1])
            if use_preprocessed_graphs:
                pair_dict = load_hetero_data_graph_as_pdb_file_pair(
                    (flat_file_input[0], flat_file_input[2]), temp_dir_path=temp_dir_path
                )
                pair = (
                    pair_dict["apo_protein"]["filepath"],
                    pair_dict["holo_protein"]["filepath"],
                )
                apo_length, holo_length = (
                    pair_dict["apo_protein"]["length"],
                    pair_dict["holo_protein"]["length"],
                )
                ligand_num_atoms = pair_dict["ligand"]["num_atoms_per_mol_frag"]
            else:
                pair, rdkit_ligand_filepaths = file_input
                apo_length, holo_length = (
                    count_num_residues_in_pdb_file(pair[0]),
                    count_num_residues_in_pdb_file(pair[1]),
                )
                lig_mol_frags = []
                for rdkit_ligand_filepath in rdkit_ligand_filepaths:
                    ligand = None
                    try:
                        ligand = (
                            Chem.MolFromPDBFile(rdkit_ligand_filepath, sanitize=False)
                            if rdkit_ligand_filepath.endswith(".pdb")
                            else Chem.MolFromMolFile(rdkit_ligand_filepath, sanitize=False)
                        )
                    except Exception as e:
                        ligand = None
                    if ligand is None:
                        try:
                            ligand = Chem.MolFromMol2File(
                                rdkit_ligand_filepath.replace(".sdf", ".mol2"), sanitize=False
                            )
                        except Exception as e:
                            ligand = None
                    if ligand is None:
                        raise ValueError(f"Could not load ligand from {rdkit_ligand_filepath}")
                    lig_mol_frags.extend(
                        Chem.GetMolFrags(ligand, asMols=True, sanitizeFrags=False)
                    )
                ligand_num_atoms = [lig.GetNumAtoms() for lig in lig_mol_frags]
            tm_score_metrics = calculate_usalign_metrics(*pair, cfg.usalign_exec_path, flags=flags)
        except Exception as e:
            log.error(f"Error calculating structural metrics for {file_input} due to: {e}")
            erroneous_file_inputs.append(flat_file_input)
            continue
        if "TM-score_2" not in tm_score_metrics or "RMSD" not in tm_score_metrics:
            log.error(f"Error calculating structural metrics for {file_input}. Skipping...")
            erroneous_file_inputs.append(flat_file_input)
            continue
        valid_file_inputs.append(flat_file_input)
        tm_scores.append(tm_score_metrics["TM-score_2"])
        rmsds.append(tm_score_metrics["RMSD"])
        apo_lengths.append(apo_length)
        holo_lengths.append(holo_length)
        ligands_num_atoms.append(ligand_num_atoms)
    tm_scores = np.array(tm_scores)
    rmsds = np.array(rmsds)
    apo_lengths = np.array(apo_lengths)
    holo_lengths = np.array(holo_lengths)
    ligands_num_atoms_array = np.array(
        [np.array(num_atoms).mean() for num_atoms in ligands_num_atoms]
    )
    log.info(f"TM-score mean: {tm_scores.mean():.3f} +/- {tm_scores.std():.3f}")
    # ≈ 0.862 +/- 0.200 protein TMscore for PDBBind when assessing with raw input files
    # ≈ 0.803 +/- 0.257 protein TMscore for Binding MOAD when assessing with raw input files
    # ≈ 0.736 +/- 0.294 protein TMscore for DockGen when assessing with raw input files
    # ≈ 0.824 +/- 0.239 protein TMscore for PDBSidechain when assessing with raw input files
    log.info(f"RMSD mean: {rmsds.mean():.3f} +/- {rmsds.std():.3f}")
    # ≈ 2.165 +/- 1.830 protein RMSD for PDBBind when assessing with raw input files
    # ≈ 2.203 +/- 1.959 protein RMSD for Binding MOAD when assessing with raw input files
    # ≈ 3.202 +/- 2.683 protein RMSD for DockGen when assessing with raw input files
    # ≈ 2.269 +/- 1.906 protein RMSD for PDBSidechain when assessing with raw input files
    log.info(f"Apo length mean: {apo_lengths.mean():.3f} +/- {apo_lengths.std():.3f}")
    # ≈ 316.076 +/- 149.825 apo residues for PDBBind when assessing with raw input files
    # ≈ 339.254 +/- 153.889 apo residues for Binding MOAD when assessing with raw input files
    # ≈ 462.712 +/- 300.881 apo residues for DockGen when assessing with raw input files
    # ≈ 249.091 +/- 145.546 apo residues for PDBSidechain when assessing with raw input files
    log.info(f"Holo length mean: {holo_lengths.mean():.3f} +/- {holo_lengths.std():.3f}")
    # ≈ 316.699 +/- 149.910 holo residues for PDBBind when assessing with raw input files
    # ≈ 340.776 +/- 202.198 holo residues for Binding MOAD when assessing with raw input files
    # ≈ 463.791 +/- 301.404 holo residues for DockGen when assessing with raw input files
    # ≈ 249.235 +/- 145.551 holo residues for PDBSidechain when assessing with raw input files
    log.info(
        f"Ligand num atoms: {ligands_num_atoms_array.mean():.3f} +/- {ligands_num_atoms_array.std():.3f}"
    )
    # ≈ 62.062 +/- 45.614 ligand atoms for PDBBind when assessing with raw input files
    # ≈ 23.939 +/- 15.876 ligand atoms for Binding MOAD when assessing with raw input files
    # ≈ 26.097 +/- 15.048 ligand atoms for DockGen when assessing with raw input files
    # ≈ nan +/- nan ligand atoms for PDBSidechain when assessing with raw input files
    pd.DataFrame(
        {
            "Filepath": valid_file_inputs,
            "TM-score": tm_scores,
            "RMSD": rmsds,
            "Apo_Length": apo_lengths,
            "Holo_Length": holo_lengths,
            "Ligand_Num_Atoms": [
                ",".join([str(num) for num in num_atoms]) for num_atoms in ligands_num_atoms
            ],
        }
    ).to_csv(cfg.output_filepath, index=False)
    log.info(f"Saved TM-score and RMSD results to {cfg.output_filepath}")
    pd.DataFrame({"Filepath": erroneous_file_inputs}).to_csv(
        cfg.erroneous_output_filepath, index=False
    )
    log.info(f"Saved erroneous file inputs to {cfg.erroneous_output_filepath}")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
