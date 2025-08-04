# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for PoseBench: (https://github.com/BioinfoMachineLearning/PoseBench)
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import glob
import logging
import os
import tempfile
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from beartype.typing import List, Tuple
from omegaconf import DictConfig, open_dict
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from posebench import register_custom_omegaconf_resolvers, resolve_method_title
from posebench.utils.data_utils import parse_inference_inputs_from_dir

logging.basicConfig(format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True

# NOTE: the DockGen dataset's crystal ligand structures are stored in PDB format, so we cannot
# evaluate the PoseBusters validity rates of a method using molecular graph or bond assertions
DOCKGEN_BUST_TEST_COLUMNS = [
    # accuracy #
    "rmsd_≤_2å",
    # chemical validity and consistency #
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    # "molecular_formula",
    # "molecular_bonds",
    # "tetrahedral_chirality",
    # "double_bond_stereochemistry",
    # intramolecular validity #
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    # intermolecular validity #
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
]

BUST_TEST_COLUMNS = DOCKGEN_BUST_TEST_COLUMNS + [
    # # accuracy #
    # "rmsd_≤_2å",
    # # chemical validity and consistency #
    # "mol_pred_loaded",
    # "mol_true_loaded",
    # "mol_cond_loaded",
    # "sanitization",
    "molecular_formula",
    "molecular_bonds",
    "tetrahedral_chirality",
    "double_bond_stereochemistry",
    # # intramolecular validity #
    # "bond_lengths",
    # "bond_angles",
    # "internal_steric_clash",
    # "aromatic_ring_flatness",
    # "double_bond_flatness",
    # "internal_energy",
    # # intermolecular validity #
    # "minimum_distance_to_protein",
    # "minimum_distance_to_organic_cofactors",
    # "minimum_distance_to_inorganic_cofactors",
    # "volume_overlap_with_protein",
    # "volume_overlap_with_organic_cofactors",
    # "volume_overlap_with_inorganic_cofactors",
]

RANKED_METHODS = ["diffdock", "dynamicbind", "neuralplexer", "flowdock"]


def find_most_similar_frag(
    mol_true_frag: Chem.Mol, mol_pred_frags: List[Chem.Mol]
) -> Tuple[Chem.Mol, float, float]:
    """Find the most similar fragment to the true fragment among the predicted
    fragments.

    :param mol_true_frag: True fragment molecule.
    :param mol_pred_frags: List of predicted fragment molecules.
    :return: Tuple of the most similar fragment molecule, the Tanimoto
        similarity, and the RMSD.
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate the fingerprint for the true fragment
    fp_true = mfpgen.GetFingerprint(mol_true_frag)

    max_similarity = -1
    min_rmsd = float("inf")
    most_similar_frag = None

    for mol_pred_frag in mol_pred_frags:
        # Skip fragments with different number of atoms
        if mol_pred_frag.GetNumAtoms() != mol_true_frag.GetNumAtoms():
            continue

        # Generate the fingerprint for the predicted fragment
        mol_pred_frag.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol_pred_frag)  # Perceive rings for fingerprinting
        fp_pred = mfpgen.GetFingerprint(mol_pred_frag)

        # Calculate the Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp_true, fp_pred)

        # Calculate the RMSD
        rmsd = (
            AllChem.GetBestRMS(mol_true_frag, mol_pred_frag) if similarity > 0.5 else float("inf")
        )

        # Update the most similar fragment if the current one is more similar or has a lower RMSD
        if similarity > max_similarity or (similarity == max_similarity and rmsd < min_rmsd):
            max_similarity = similarity
            min_rmsd = rmsd
            most_similar_frag = mol_pred_frag

    return most_similar_frag, max_similarity, min_rmsd


def select_primary_ligands_in_df(
    mol_table: pd.DataFrame, select_most_similar_pred_frag: bool = True
) -> pd.DataFrame:
    """Select the primary ligands predictions from the molecule table
    DataFrame.

    NOTE: This function is used for single-primary-ligand datasets such as Astex Diverse, PoseBusters Benchmark, and DockGen
    to identify a method's (most likely) prediction for a specific primary ligand crystal structure when the method is tasked
    with predicting all cofactors as well (to enhance its molecular context for primary ligand predictions).

    :param mol_table: Molecule table DataFrame.
    :param select_most_similar_pred_frag: Whether to select the predicted ligand fragment most similar (chemically and structurally) to the true ligand fragment.
    :return: Molecule table DataFrame with primary ligand predictions.
    """
    new_rows = []
    for row in mol_table.itertuples():
        try:
            mol_true_file_fn = (
                Chem.MolFromPDBFile if str(row.mol_true).endswith(".pdb") else Chem.MolFromMolFile
            )

            mol_true = mol_true_file_fn(str(row.mol_true), removeHs=False)
            mol_pred = Chem.MolFromMolFile(str(row.mol_pred), removeHs=False)

            assert mol_true is not None, f"Failed to load the true molecule from {row.mol_true}."
            assert (
                mol_pred is not None
            ), f"Failed to load the predicted molecule from {row.mol_pred}."

            mol_true_frags = Chem.GetMolFrags(mol_true, asMols=True, sanitizeFrags=False)
            mol_pred_frags = Chem.GetMolFrags(mol_pred, asMols=True, sanitizeFrags=False)

            if select_most_similar_pred_frag:
                mol_pred_frags = [
                    find_most_similar_frag(mol_true_frag, mol_pred_frags)[0]
                    for mol_true_frag in mol_true_frags
                ]
                if not any(mol_pred_frags):
                    logger.warning(
                        f"None of the predicted fragments are similar enough to the true fragments for row {row.Index}. Skipping this row."
                    )
                    continue

            assert len(mol_true_frags) == len(
                mol_pred_frags
            ), "The number of fragments should be the same."

            for frag_index, (mol_true_frag, mol_pred_frag) in enumerate(
                zip(mol_true_frags, mol_pred_frags)
            ):
                new_row = row._asdict()
                new_row["mol_cond"] = row.mol_cond
                with tempfile.NamedTemporaryFile(
                    suffix=".sdf", delete=False
                ) as temp_true, tempfile.NamedTemporaryFile(
                    suffix=".sdf", delete=False
                ) as temp_pred:
                    assert (
                        mol_true_frag.GetNumAtoms() == mol_pred_frag.GetNumAtoms()
                    ), "The number of atoms in each fragment should be the same."

                    Chem.MolToMolFile(mol_true_frag, temp_true.name)
                    Chem.MolToMolFile(mol_pred_frag, temp_pred.name)
                    true_smiles = Chem.MolToSmiles(Chem.MolFromMolFile(temp_true.name))
                    pred_smiles = Chem.MolToSmiles(Chem.MolFromMolFile(temp_pred.name))

                    if true_smiles != pred_smiles:
                        logger.warning(
                            f"The SMILES strings of the index {frag_index} fragments ({true_smiles} vs. {pred_smiles}) differ for row {row.Index} after post-processing."
                        )

                    new_row["mol_true"] = temp_true.name
                    new_row["mol_pred"] = temp_pred.name
                new_rows.append(new_row)

        except Exception as e:
            logger.warning(
                f"An error occurred while splitting fragments for row {row.Index}: {e}. Skipping this row."
            )

    return pd.DataFrame(new_rows)


def create_mol_table(
    input_csv_path: Path,
    input_data_dir: Path,
    inference_dir: Path,
    cfg: DictConfig,
    relaxed: bool = False,
    add_pdb_ids: bool = False,
) -> pd.DataFrame:
    """Create a table of molecules and their corresponding ligand files.

    :param input_csv_path: Path to the input CSV file.
    :param input_data_dir: Path to the input data directory.
    :param inference_dir: Path to the inference directory.
    :param mol_table_filepath: Molecule table DataFrame.
    :param cfg: Hydra configuration dictionary.
    :param relaxed: Whether to use the relaxed poses.
    :param add_pdb_ids: Whether to add the PDB IDs to the molecule table
        DataFrame.
    :return: Molecule table DataFrame.
    """
    pdb_ids = None
    relaxed_protein = relaxed and cfg.relax_protein
    if cfg.dataset == "dockgen" and cfg.dockgen_test_ids_filepath is not None:
        # NOTE: for DockGen, we may have each method predict for all 189 test complexes
        assert os.path.exists(
            cfg.dockgen_test_ids_filepath
        ), f"Invalid test IDs file path for DockGen: {os.path.exists(cfg.dockgen_test_ids_filepath)}."
        with open(cfg.dockgen_test_ids_filepath) as f:
            pdb_ids = {line.replace(" ", "-") for line in f.read().splitlines()}

    if cfg.method in ["dynamicbind", "rfaa", "chai-lab", "boltz", "alphafold3"]:
        # NOTE: for methods such as DynamicBind and RoseTTAFold-All-Atom,
        # the input CSV file needs to be created manually from the input data directory
        input_smiles_and_pdb_ids = parse_inference_inputs_from_dir(input_data_dir, pdb_ids=pdb_ids)
        input_table = pd.DataFrame(input_smiles_and_pdb_ids, columns=["smiles", "pdb_id"])
    else:
        input_table = pd.read_csv(input_csv_path)
        if "id" in input_table.columns:
            input_table.rename(columns={"id": "pdb_id"}, inplace=True)
        if "name" in input_table.columns:
            input_table.rename(columns={"name": "pdb_id"}, inplace=True)
        if "pdb_id" not in input_table.columns:
            input_table["pdb_id"] = input_table["complex_name"].copy()
        if cfg.dataset == "dockgen":
            input_table = input_table[input_table["pdb_id"].isin(pdb_ids)]

    # parse molecule (e.g., protein-)conditioning files
    mol_table = pd.DataFrame()

    if add_pdb_ids:
        mol_table["pdb_id"] = input_table["pdb_id"]

    if cfg.method == "dynamicbind":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    Path(
                        str(inference_dir).replace("_relaxed", "") + f"_{x}_{cfg.repeat_index}"
                    ).rglob(f"*rank1_receptor_*{'relaxed*' if relaxed_protein else ''}.pdb")
                )[0]
                if len(
                    list(
                        Path(
                            str(inference_dir).replace("_relaxed", "") + f"_{x}_{cfg.repeat_index}"
                        ).rglob(f"*rank1_receptor_*{'_relaxed' if relaxed_protein else ''}.pdb")
                    )
                )
                else None
            )
        )
    elif cfg.method in ["neuralplexer", "flowdock"]:
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"prot_rank1_*_aligned{'_relaxed' if relaxed_protein else ''}.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"prot_rank1_*_aligned{'_relaxed' if relaxed_protein else ''}.pdb"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "rfaa":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                        f"{x}_protein_aligned{'_relaxed' if relaxed_protein else ''}.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                            f"{x}_protein_aligned{'_relaxed' if relaxed_protein else ''}.pdb"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "chai-lab":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"pred.model_idx_0_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"pred.model_idx_0_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "boltz":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"*_model_0_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"*_model_0_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "alphafold3":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"*_model_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"*_model_protein{'_relaxed' if relaxed_protein else ''}_aligned.pdb"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "ensemble":
        mol_table["mol_cond"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"*_rank1_*{'_relaxed' if relaxed_protein else ''}.pdb"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"*_rank1_*{'_relaxed' if relaxed_protein else ''}.pdb"
                        )
                    )
                )
                else None
            )
        )
    else:
        pocket_suffix = "_bs_cropped" if cfg.pocket_only_baseline else ""
        protein_structure_input_dir = (
            os.path.join(
                input_data_dir,
                f"{cfg.dataset}_holo_aligned_predicted_structures{pocket_suffix}",
            )
            if os.path.exists(
                os.path.join(
                    input_data_dir,
                    f"{cfg.dataset}_holo_aligned_predicted_structures{pocket_suffix}",
                )
            )
            else os.path.join(input_data_dir, f"{cfg.dataset}_predicted_structures")
        )
        protein_structure_file_suffix = (
            "_holo_aligned_predicted_protein"
            if os.path.exists(
                os.path.join(
                    input_data_dir,
                    f"{cfg.dataset}_holo_aligned_predicted_structures{pocket_suffix}",
                )
            )
            and cfg.dataset != "casp15"
            else ""
        )
        if relaxed_protein:
            protein_structure_input_dir = str(inference_dir).replace("_relaxed", "")
            protein_structure_file_suffix = "_relaxed"
            mol_table["mol_cond"] = input_table["pdb_id"].apply(
                lambda x: (
                    os.path.join(
                        protein_structure_input_dir,
                        "_".join(x.split("_")[:3]),
                        f"{'_'.join(x.split('_')[:2])}{protein_structure_file_suffix}.pdb",
                    )
                    if os.path.exists(
                        os.path.join(
                            protein_structure_input_dir,
                            "_".join(x.split("_")[:3]),
                            f"{'_'.join(x.split('_')[:2])}{protein_structure_file_suffix}.pdb",
                        )
                    )
                    else None
                )
            )
        else:
            mol_table["mol_cond"] = input_table["pdb_id"].apply(
                lambda x: (
                    os.path.join(
                        protein_structure_input_dir,
                        f"{x}{protein_structure_file_suffix}.pdb",
                    )
                    if os.path.exists(
                        os.path.join(
                            protein_structure_input_dir,
                            f"{x}{protein_structure_file_suffix}.pdb",
                        )
                    )
                    else None
                )
            )
    # parse true molecule files
    mol_true_file_ext = ".pdb" if cfg.dataset == "dockgen" else ".sdf"
    mol_table["mol_true"] = input_table["pdb_id"].apply(
        lambda x: os.path.join(input_data_dir, x, f"{x}_ligand{mol_true_file_ext}")
    )
    # parse predicted molecule files
    if cfg.method == "dynamicbind":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    Path(
                        str(inference_dir).replace("_relaxed", "")
                        + f"_{x}_{cfg.repeat_index}{'_relaxed' if relaxed else ''}"
                    ).rglob("*rank1_ligand_*.sdf")
                )[0]
                if len(
                    list(
                        Path(
                            str(inference_dir).replace("_relaxed", "")
                            + f"_{x}_{cfg.repeat_index}{'_relaxed' if relaxed else ''}"
                        ).rglob("*rank1_ligand_*.sdf")
                    )
                )
                else None
            )
        )
    elif cfg.method == "rfaa":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                        f"{x}_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                            f"{x}_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "chai-lab":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"pred.model_idx_0_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"pred.model_idx_0_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "boltz":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"*_model_0_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"*_model_0_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "alphafold3":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"*_model_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"*_model_ligand{'_relaxed' if relaxed else ''}_aligned.sdf"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "vina":
        mol_table["mol_pred"] = (
            input_table["pdb_id"]
            .transform(lambda x: "_".join(x.split("_")[:3]))
            .apply(
                lambda x: (
                    list(
                        (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                            f"{x}{'_relaxed' if relaxed else ''}.sdf"
                        )
                    )[0]
                    if len(
                        list(
                            (Path(str(inference_dir).replace("_relaxed", ""))).rglob(
                                f"{x}{'_relaxed' if relaxed else ''}.sdf"
                            )
                        )
                    )
                    else None
                )
            )
        )
    elif cfg.method == "tulip":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                list(
                    (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        f"rank1{'_relaxed' if relaxed else ''}.sdf"
                    )
                )[0]
                if len(
                    list(
                        (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            f"rank1{'_relaxed' if relaxed else ''}.sdf"
                        )
                    )
                )
                else None
            )
        )
    elif cfg.method == "ensemble":
        mol_table["mol_pred"] = input_table["pdb_id"].apply(
            lambda x: (
                [
                    file
                    for file in (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                        "*_rank1_*.sdf"
                    )
                    if (relaxed and "relaxed" in os.path.basename(file))
                    or (not relaxed and "relaxed" not in os.path.basename(file))
                ][0]
                if len(
                    [
                        file
                        for file in (Path(str(inference_dir).replace("_relaxed", "")) / x).rglob(
                            "*_rank1_*.sdf"
                        )
                        if (relaxed and "relaxed" in os.path.basename(file))
                        or (not relaxed and "relaxed" not in os.path.basename(file))
                    ]
                )
                else None
            )
        )
    else:
        pdb_ids = input_table["pdb_id"]
        if cfg.method in RANKED_METHODS:
            mol_table["mol_pred"] = pdb_ids.apply(
                lambda x: (
                    glob.glob(
                        os.path.join(
                            (
                                Path(str(inference_dir).replace("_relaxed", ""))
                                if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                or relaxed_protein
                                else inference_dir
                            ),
                            x,
                            (
                                "lig_rank1*_relaxed_aligned.sdf"
                                if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                else f"{x}_relaxed.sdf"
                            ),
                        )
                        if relaxed
                        else os.path.join(
                            (
                                Path(str(inference_dir).replace("_relaxed", ""))
                                if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                else inference_dir
                            ),
                            x,
                            (
                                "lig_rank1*_aligned.sdf"
                                if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                else "rank1.sdf"
                            ),
                        )
                    )[0]
                    if len(
                        glob.glob(
                            os.path.join(
                                (
                                    Path(str(inference_dir).replace("_relaxed", ""))
                                    if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                    or relaxed_protein
                                    else inference_dir
                                ),
                                x,
                                (
                                    "lig_rank1*_relaxed_aligned.sdf"
                                    if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                    else f"{x}_relaxed.sdf"
                                ),
                            )
                            if relaxed
                            else os.path.join(
                                (
                                    Path(str(inference_dir).replace("_relaxed", ""))
                                    if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                    else inference_dir
                                ),
                                x,
                                (
                                    "lig_rank1*_aligned.sdf"
                                    if cfg.method in ["neuralplexer", "flowdock", "rfaa"]
                                    else "rank1.sdf"
                                ),
                            )
                        )
                    )
                    else None
                )
            )
        else:
            mol_table["mol_pred"] = pdb_ids.apply(
                lambda x: (
                    glob.glob(
                        os.path.join(
                            inference_dir,
                            f"{x}_*{'_relaxed' if relaxed else ''}{'_aligned' if cfg.method in ['neuralplexer', 'flowdock', 'rfaa'] else ''}.sdf",
                        )
                    )[0]
                    if len(
                        glob.glob(
                            os.path.join(
                                inference_dir,
                                f"{x}_*{'_relaxed' if relaxed else ''}{'_aligned' if cfg.method in ['neuralplexer', 'flowdock', 'rfaa'] else ''}.sdf",
                            )
                        )
                    )
                    else None
                )
            )

    # drop rows with missing conditioning inputs or true ligand structures
    missing_true_indices = mol_table["mol_cond"].isna() | mol_table["mol_true"].isna()
    mol_table = mol_table.dropna(subset=["mol_cond", "mol_true"])
    input_table = input_table[~missing_true_indices]

    # check for missing (relaxed) predictions
    if mol_table["mol_pred"].isna().sum() > 0:
        if relaxed:
            missing_pred_indices = mol_table["mol_pred"].isna()
            unrelaxed_inference_dir = Path(str(inference_dir).replace("_relaxed", ""))
            if cfg.method == "diffdock":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(os.path.join(unrelaxed_inference_dir, x, "rank1.sdf"))[0]
                        if len(glob.glob(os.path.join(unrelaxed_inference_dir, x, "rank1.sdf")))
                        else None
                    )
                )
            elif cfg.method == "dynamicbind":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        list(
                            Path(str(unrelaxed_inference_dir) + f"_{x}_{cfg.repeat_index}").rglob(
                                "*rank1_ligand_*.sdf"
                            )
                        )[0]
                        if len(
                            list(
                                Path(
                                    str(unrelaxed_inference_dir) + f"_{x}_{cfg.repeat_index}"
                                ).rglob("*rank1_ligand_*.sdf")
                            )
                        )
                        else None
                    )
                )
            elif cfg.method in ["neuralplexer", "flowdock"]:
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(
                            os.path.join(
                                Path(str(inference_dir).replace("_relaxed", "")),
                                x,
                                "lig_rank1_aligned.sdf",
                            )
                        )[0]
                        if len(
                            glob.glob(
                                os.path.join(
                                    Path(str(inference_dir).replace("_relaxed", "")),
                                    x,
                                    "lig_rank1_aligned.sdf",
                                )
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "rfaa":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        list(
                            (Path(str(unrelaxed_inference_dir))).rglob(f"{x}_ligand_aligned.sdf")
                        )[0]
                        if len(
                            list(
                                (Path(str(unrelaxed_inference_dir))).rglob(
                                    f"{x}_ligand_aligned.sdf"
                                )
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "chai-lab":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(
                            os.path.join(
                                Path(str(inference_dir).replace("_relaxed", "")),
                                x,
                                "pred.model_idx_0_ligand_aligned.sdf",
                            )
                        )[0]
                        if len(
                            glob.glob(
                                os.path.join(
                                    Path(str(inference_dir).replace("_relaxed", "")),
                                    x,
                                    "pred.model_idx_0_ligand_aligned.sdf",
                                )
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "boltz":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(
                            os.path.join(
                                Path(str(inference_dir).replace("_relaxed", "")),
                                x,
                                "*_model_0_ligand_aligned.sdf",
                            )
                        )[0]
                        if len(
                            glob.glob(
                                os.path.join(
                                    Path(str(inference_dir).replace("_relaxed", "")),
                                    x,
                                    "*_model_0_ligand_aligned.sdf",
                                )
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "alphafold3":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(
                            os.path.join(
                                Path(str(inference_dir).replace("_relaxed", "")),
                                x,
                                "*_model_ligand_aligned.sdf",
                            )
                        )[0]
                        if len(
                            glob.glob(
                                os.path.join(
                                    Path(str(inference_dir).replace("_relaxed", "")),
                                    x,
                                    "*_model_ligand_aligned.sdf",
                                )
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "vina":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        list((Path(str(unrelaxed_inference_dir))).rglob(f"{x}.sdf"))[0]
                        if len(list((Path(str(unrelaxed_inference_dir))).rglob(f"{x}.sdf")))
                        else None
                    )
                )
            elif cfg.method == "tulip":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        list((Path(str(unrelaxed_inference_dir)) / x).rglob(f"{x}_relaxed.sdf"))[0]
                        if len(
                            list(
                                (Path(str(unrelaxed_inference_dir)) / x).rglob(f"{x}_relaxed.sdf")
                            )
                        )
                        else None
                    )
                )
            elif cfg.method == "ensemble":
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(os.path.join(unrelaxed_inference_dir, x, "*.sdf"))[0]
                        if len(glob.glob(os.path.join(unrelaxed_inference_dir, x, "*.sdf")))
                        else None
                    )
                )
            else:
                mol_table.loc[missing_pred_indices, "mol_pred"] = input_table.loc[
                    missing_pred_indices, "pdb_id"
                ].apply(
                    lambda x: (
                        glob.glob(os.path.join(unrelaxed_inference_dir, f"{x}_*.sdf"))[0]
                        if len(glob.glob(os.path.join(unrelaxed_inference_dir, f"{x}_*.sdf")))
                        else None
                    )
                )
            if mol_table["mol_pred"].isna().sum() > 0:
                logger.warning(
                    f"Skipping imputing missing (relaxed) predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
                )
                mol_table = mol_table.dropna(subset=["mol_pred"])
        else:
            logger.warning(
                f"Skipping missing predictions for {mol_table['mol_pred'].isna().sum()} molecules regarding the following conditioning inputs: {mol_table[mol_table['mol_pred'].isna()]['mol_cond'].tolist()}."
            )
            mol_table = mol_table.dropna(subset=["mol_pred"])

    mol_table.reset_index(drop=True, inplace=True)
    mol_table = select_primary_ligands_in_df(mol_table)

    return mol_table


@hydra.main(
    version_base="1.3",
    config_path="../../configs/analysis",
    config_name="inference_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the inference results of a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    with open_dict(cfg):
        # NOTE: besides their output directories, single-sequence baselines are treated like their multi-sequence counterparts
        output_dir = copy.deepcopy(cfg.output_dir)
        cfg.method = cfg.method.removesuffix("_ss")
        cfg.output_dir = output_dir

    for config in ["", "_relaxed"]:
        output_dir = cfg.output_dir + config
        bust_results_filepath = Path(output_dir) / "bust_results.csv"

        # differentiate relaxed and unrelaxed protein pose results
        if "relaxed" in config and cfg.relax_protein:
            bust_results_filepath = Path(
                str(bust_results_filepath).replace(".csv", "_protein_relaxed.csv")
            )

        os.makedirs(bust_results_filepath.parent, exist_ok=True)

        # collect test results
        if os.path.exists(bust_results_filepath) and not cfg.force_rescore:
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` already exist at `{bust_results_filepath}`. Directly analyzing..."
            )
            bust_results = pd.read_csv(bust_results_filepath)

            # if necessary, track the molecule ID of each row after initial scoring
            if "mol_id" not in bust_results.columns:
                if cfg.method == "dynamicbind":
                    output_dir = os.path.join(
                        Path(output_dir).parent,
                        Path(output_dir).stem.replace(f"_{cfg.repeat_index}", ""),
                    )
                mol_table = create_mol_table(
                    Path(cfg.input_csv_path),
                    Path(cfg.input_data_dir),
                    Path(output_dir),
                    cfg,
                    relaxed="relaxed" in config,
                    add_pdb_ids=True,
                )
                bust_results["mol_id"] = mol_table["pdb_id"]

                bust_results.to_csv(bust_results_filepath, index=False)
                logger.info(
                    f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` successfully amended at `{bust_results_filepath}`."
                )

        else:
            if cfg.method == "dynamicbind":
                output_dir = os.path.join(
                    Path(output_dir).parent,
                    Path(output_dir).stem.replace(f"_{cfg.repeat_index}", ""),
                )
            mol_table = create_mol_table(
                Path(cfg.input_csv_path),
                Path(cfg.input_data_dir),
                Path(output_dir),
                cfg,
                relaxed="relaxed" in config,
                add_pdb_ids=True,
            )
            mol_ids = copy.deepcopy(mol_table["pdb_id"])

            # NOTE: we use the `redock` mode here since with each method we implicitly perform cognate (e.g., apo or ab initio) docking,
            # and we have access to the ground-truth ligand structures in SDF format
            buster = PoseBusters(config="redock", top_n=None)
            buster.config["loading"]["mol_true"]["load_all"] = False
            bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)
            bust_results["mol_id"] = mol_ids.values

            bust_results.to_csv(bust_results_filepath, index=False)
            logger.info(
                f"{resolve_method_title(cfg.method)}{config} bust results for inference directory `{output_dir}` successfully saved to `{bust_results_filepath}`."
            )

        # report test results
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} rmsd_≤_2å: {bust_results['rmsd_≤_2å'].mean()}"
        )
        tests_table = copy.deepcopy(
            bust_results[
                DOCKGEN_BUST_TEST_COLUMNS if cfg.dataset == "dockgen" else BUST_TEST_COLUMNS
            ]
        )
        tests_table.loc[:, "pb_valid"] = tests_table.iloc[:, 1:].all(axis=1)
        logger.info(
            f"{resolve_method_title(cfg.method)}{config} rmsd_≤_2å and pb_valid: {tests_table[tests_table['pb_valid']]['rmsd_≤_2å'].sum() / len(tests_table)}"
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
