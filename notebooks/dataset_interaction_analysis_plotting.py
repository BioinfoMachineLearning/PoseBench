# %% [markdown]
# ## Dataset Interaction Analysis Plotting

# %% [markdown]
# #### Import packages

# %%
import os
import random
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Any, Literal
from Bio.PDB import PDBIO, PDBParser, Select
from posecheck import PoseCheck
from rdkit import Chem
from tqdm import tqdm

# %% [markdown]
# #### Configure packages

# %%
pd.options.mode.copy_on_write = True

# %% [markdown]
# #### Define constants

# %%
pdbbind_set_dir = os.path.join("..", "data", "pdbbind", "PDBBind_processed")
ad_set_dir = os.path.join("..", "data", "astex_diverse_set")
pb_set_dir = os.path.join("..", "data", "posebusters_benchmark_set")
dg_set_dir = os.path.join("..", "data", "dockgen_set")
casp15_set_dir = os.path.join(
    "..",
    "data",
    "casp15_set",
    "targets",
)
assert os.path.exists(
    ad_set_dir
), "Please download the Astex Diverse set from `https://zenodo.org/records/16791095` before proceeding."
assert os.path.exists(
    pb_set_dir
), "Please download the PoseBusters Benchmark set from `https://zenodo.org/records/16791095` before proceeding."
assert os.path.exists(
    dg_set_dir
), "Please download the DockGen set from `https://zenodo.org/records/16791095` before proceeding."
assert os.path.exists(
    casp15_set_dir
), "Please download the (public) CASP15 set from `https://zenodo.org/records/16791095` before proceeding."

CASP15_ANALYSIS_TARGETS_TO_SKIP = [
    "T1170"
]  # NOTE: these will be skipped since they were not scoreable
MAX_CASP15_ANALYSIS_PROTEIN_SEQUENCE_LENGTH = (
    2000  # Only CASP15 targets with protein sequences below this threshold can be analyzed
)

# %% [markdown]
# #### Define utility functions


# %%
class ProteinSelect(Select):
    """A class to select only protein residues from a PDB file."""

    def accept_residue(self, residue: Any):
        """Only accept residues that are part of a protein (e.g., standard
        amino acids).

        :param residue: The residue to check.
        :return: True if the residue is part of a protein, False
            otherwise.
        """
        return residue.id[0] == " "  # NOTE: `HETATM` flag must be a blank for protein residues


class LigandSelect(Select):
    """A class to select only ligand residues from a PDB file."""

    def accept_residue(self, residue: Any):
        """Only accept residues that are part of a ligand.

        :param residue: The residue to check.
        :return: True if the residue is part of a ligand, False
            otherwise.
        """
        return residue.id[0] != " "  # NOTE: `HETATM` flag must be a filled for ligand residues


@beartype
def create_temp_pdb_with_only_molecule_type_residues(
    input_pdb_filepath: str, molecule_type: Literal["protein", "ligand"]
) -> str:
    """Create a temporary PDB file with only residues of a chosen molecule
    type.

    :param input_pdb_filepath: The input PDB file path.
    :param molecule_type: The molecule type to keep in the temporary PDB
        file.
    :return: The temporary PDB file path.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(molecule_type, input_pdb_filepath)

    io = PDBIO()
    io.set_structure(structure)

    # create a temporary PDB file
    temp_pdb_filepath = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    io.save(
        temp_pdb_filepath.name, ProteinSelect() if molecule_type == "protein" else LigandSelect()
    )

    return temp_pdb_filepath.name


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


# %% [markdown]
# #### Compute interaction fingerprints for each dataset

# %% [markdown]
# ##### Analyze `PDBBind 2020` training subset interactions (as a generalization reference point for the following test sets)

# %%
pdbbind_train_subset_size = 1000

if os.path.exists(pdbbind_set_dir) and not os.path.exists(
    f"pdbbind_training_subset_{pdbbind_train_subset_size}_interaction_dataframes.h5"
):
    # NOTE: If PDBBind 2020 has not already been downloaded locally, the following commands will download and extract it:
    # cd ../data/pdbbind/
    # wget https://zenodo.org/record/6408497/files/PDBBind.zip
    # unzip PDBBind.zip
    # rm PDBBind.zip
    # cd ../../notebooks/
    # NOTE: Now, you should have a PDBBind data directory structure that looks like this: `../data/pdbbind/PDBBind_processed/`

    with open(
        os.path.join(os.path.dirname(pdbbind_set_dir), "timesplit_no_lig_overlap_train")
    ) as f:
        pdbbind_train_ids = set(f.read().splitlines())

    pdbbind_train_set_dirs = [
        item for item in os.listdir(pdbbind_set_dir) if item in pdbbind_train_ids
    ]
    random.shuffle(pdbbind_train_set_dirs)

    pdbbind_protein_ligand_filepath_pairs = []
    for item in pdbbind_train_set_dirs[:pdbbind_train_subset_size]:
        item_path = os.path.join(pdbbind_set_dir, item)
        if os.path.isdir(item_path):
            protein_filepath = os.path.join(item_path, f"{item}_protein_processed.pdb")
            ligand_filepath = os.path.join(item_path, f"{item}_ligand.mol2")
            if os.path.exists(protein_filepath) and os.path.exists(ligand_filepath):
                pdbbind_protein_ligand_filepath_pairs.append((protein_filepath, ligand_filepath))

    pc = PoseCheck()
    pdbbind_protein_ligand_interaction_dfs = []
    for protein_filepath, ligand_filepath in tqdm(
        pdbbind_protein_ligand_filepath_pairs, desc="Processing PDBBind 2020 set"
    ):
        try:
            temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_filepath, molecule_type="protein"
            )
            ligand_mol = None
            try:
                ligand_mol = Chem.MolFromMolFile(ligand_filepath)
            except Exception as e:
                ligand_mol = Chem.MolFromMolFile(ligand_filepath, sanitize=False)
            if ligand_mol is None:
                print(
                    f"Using the `.mol2` file for PDBBind 2020 target {ligand_filepath} failed. We found a `.sdf` file instead and are trying to use that. Be aware that the `.sdf` files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
                )
                try:
                    ligand_mol = Chem.MolFromMolFile(ligand_filepath.replace(".mol2", ".sdf"))
                except Exception as e:
                    ligand_mol = Chem.MolFromMolFile(
                        ligand_filepath.replace(".mol2", ".sdf"), sanitize=False
                    )
                try:
                    Chem.rdmolops.AssignAtomChiralTagsFromStructure(ligand_mol)
                except Exception as e:
                    print(
                        f"Could not assign chirality tags to the atoms in the PDBBind ligand molecule from {ligand_filepath}."
                    )
                if ligand_mol is None:
                    raise ValueError(f"Could not load PDBBind 2020 ligand from {ligand_filepath}.")
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_mols([ligand_mol])
            pdbbind_protein_ligand_interaction_df = pc.calculate_interactions(n_jobs=1)
            pdbbind_protein_ligand_interaction_df["target"] = os.path.basename(
                protein_filepath
            ).split("_protein")[0]
            pdbbind_protein_ligand_interaction_dfs.append(pdbbind_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing PDBBind filepaths {temp_protein_filepath} and {ligand_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore(
            f"pdbbind_training_subset_{pdbbind_train_subset_size}_interaction_dataframes.h5"
        ) as store:
            for i, df in enumerate(pdbbind_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# ##### Analyze `Astex Diverse` set interactions

# %%
if not os.path.exists("astex_diverse_interaction_dataframes.h5"):
    ad_protein_ligand_filepath_pairs = []
    for item in os.listdir(ad_set_dir):
        item_path = os.path.join(ad_set_dir, item)
        if os.path.isdir(item_path):
            protein_filepath = os.path.join(item_path, f"{item}_protein.pdb")
            ligand_filepath = os.path.join(item_path, f"{item}_ligand.sdf")
            if os.path.exists(protein_filepath) and os.path.exists(ligand_filepath):
                ad_protein_ligand_filepath_pairs.append((protein_filepath, ligand_filepath))

    pc = PoseCheck()
    ad_protein_ligand_interaction_dfs = []
    for protein_filepath, ligand_filepath in tqdm(
        ad_protein_ligand_filepath_pairs, desc="Processing Astex Diverse set"
    ):
        try:
            temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_filepath, molecule_type="protein"
            )
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_sdf(ligand_filepath)
            ad_protein_ligand_interaction_df = pc.calculate_interactions(n_jobs=1)
            ad_protein_ligand_interaction_df["target"] = os.path.basename(protein_filepath).split(
                "_protein"
            )[0]
            ad_protein_ligand_interaction_dfs.append(ad_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing Astex Diverse filepaths {temp_protein_filepath} and {ligand_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore("astex_diverse_interaction_dataframes.h5") as store:
            for i, df in enumerate(ad_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# ##### Analyze `PoseBusters Benchmark` set interactions

# %%
if not os.path.exists("posebusters_benchmark_interaction_dataframes.h5"):
    posebusters_ccd_ids_filepath = os.path.join(
        "..",
        "data",
        "posebusters_pdb_ccd_ids.txt",
    )
    assert os.path.exists(
        posebusters_ccd_ids_filepath
    ), f"Invalid CCD IDs file path for PoseBusters Benchmark: {os.path.exists(posebusters_ccd_ids_filepath)}."
    with open(posebusters_ccd_ids_filepath) as f:
        pdb_ids = set(f.read().splitlines())
    pb_protein_ligand_filepath_pairs = []
    for item in os.listdir(pb_set_dir):
        if item not in pdb_ids:
            continue
        item_path = os.path.join(pb_set_dir, item)
        if os.path.isdir(item_path):
            protein_filepath = os.path.join(item_path, f"{item}_protein.pdb")
            ligand_filepath = os.path.join(item_path, f"{item}_ligand.sdf")
            if os.path.exists(protein_filepath) and os.path.exists(ligand_filepath):
                pb_protein_ligand_filepath_pairs.append((protein_filepath, ligand_filepath))

    pc = PoseCheck()
    pb_protein_ligand_interaction_dfs = []
    for protein_filepath, ligand_filepath in tqdm(
        pb_protein_ligand_filepath_pairs, desc="Processing PoseBusters Benchmark set"
    ):
        try:
            temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_filepath, molecule_type="protein"
            )
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_sdf(ligand_filepath)
            pb_protein_ligand_interaction_df = pc.calculate_interactions(n_jobs=1)
            pb_protein_ligand_interaction_df["target"] = os.path.basename(protein_filepath).split(
                "_protein"
            )[0]
            pb_protein_ligand_interaction_dfs.append(pb_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing PoseBusters Benchmark filepaths {temp_protein_filepath} and {ligand_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore("posebusters_benchmark_interaction_dataframes.h5") as store:
            for i, df in enumerate(pb_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# ##### Analyze `DockGen` set interactions

# %%
if not os.path.exists("dockgen_interaction_dataframes.h5"):
    dockgen_test_ids_filepath = os.path.join(
        "..", "data", "dockgen_set", "split_test.txt"
    )  # NOTE: change as needed
    assert os.path.exists(
        dockgen_test_ids_filepath
    ), f"Invalid test IDs filepath for DockGen: {os.path.exists(dockgen_test_ids_filepath)}."
    with open(dockgen_test_ids_filepath) as f:
        pdb_ids = {line.replace(" ", "-") for line in f.read().splitlines()}
    dg_protein_ligand_filepath_pairs = []
    for item in os.listdir(dg_set_dir):
        if item not in pdb_ids:
            continue
        item_path = os.path.join(dg_set_dir, item)
        if os.path.isdir(item_path):
            protein_filepath = os.path.join(item_path, f"{item}_protein_processed.pdb")
            ligand_filepath = os.path.join(item_path, f"{item}_ligand.pdb")
            if os.path.exists(protein_filepath) and os.path.exists(ligand_filepath):
                dg_protein_ligand_filepath_pairs.append((protein_filepath, ligand_filepath))

    pc = PoseCheck()
    dg_protein_ligand_interaction_dfs = []
    for protein_filepath, ligand_filepath in tqdm(
        dg_protein_ligand_filepath_pairs, desc="Processing DockGen set"
    ):
        try:
            temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_filepath, molecule_type="protein"
            )
            # NOTE: due to a bug in RDKit up until version 2024.09.2 (see https://github.com/rdkit/rdkit/issues/5599),
            # to cache DockGen's PLIs, one may need to temporarily install the latest RDKit within a temporary new environment
            # via `mamba env create -f environments/posebench_rd_environment.yaml` to avoid a segmentation fault (i.e., core dumped)
            ligand_mol = Chem.MolFromPDBFile(ligand_filepath)
            if ligand_mol is None:
                ligand_mol = Chem.MolFromPDFile(ligand_filepath, sanitize=False)
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_mols([ligand_mol])
            dg_protein_ligand_interaction_df = pc.calculate_interactions(n_jobs=1)
            dg_protein_ligand_interaction_df["target"] = os.path.basename(protein_filepath).split(
                "_protein"
            )[0]
            dg_protein_ligand_interaction_dfs.append(dg_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing Dockgen filepaths {temp_protein_filepath} and {ligand_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore("dockgen_interaction_dataframes.h5") as store:
            for i, df in enumerate(dg_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# ##### Analyze `CASP15` set interactions

# %%
if not os.path.exists("casp15_interaction_dataframes.h5"):
    casp15_protein_ligand_complex_filepaths = []
    for item in os.listdir(casp15_set_dir):
        item_path = os.path.join(casp15_set_dir, item)
        if item.endswith("_lig.pdb") and item.split("_")[0] not in CASP15_ANALYSIS_TARGETS_TO_SKIP:
            casp15_protein_ligand_complex_filepaths.append(item_path)

    pc = PoseCheck()
    casp15_protein_ligand_interaction_dfs = []
    for protein_ligand_complex_filepath in tqdm(
        casp15_protein_ligand_complex_filepaths, desc="Processing CASP15 set"
    ):
        try:
            temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_ligand_complex_filepath, molecule_type="protein"
            )
            num_residues_in_target_protein = count_num_residues_in_pdb_file(temp_protein_filepath)
            if num_residues_in_target_protein > MAX_CASP15_ANALYSIS_PROTEIN_SEQUENCE_LENGTH:
                print(
                    f"CASP15 target {protein_ligand_complex_filepath} has too many protein residues ({num_residues_in_target_protein} > {MAX_CASP15_ANALYSIS_PROTEIN_SEQUENCE_LENGTH}) for `MDAnalysis` to fit into CPU memory. Skipping..."
                )
                continue
            temp_ligand_filepath = create_temp_pdb_with_only_molecule_type_residues(
                protein_ligand_complex_filepath, molecule_type="ligand"
            )
            # NOTE: due to a bug in RDKit up until version 2024.09.2 (see https://github.com/rdkit/rdkit/issues/5599),
            # to cache CASP15's PLIs, one may need to temporarily install the latest RDKit within a temporary new environment
            # via `mamba env create -f environments/posebench_rd_environment.yaml` to avoid a segmentation fault (i.e., core dumped)
            ligand_mol = Chem.MolFromPDBFile(temp_ligand_filepath)
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_mols(
                Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
            )
            casp15_protein_ligand_interaction_df = pc.calculate_interactions(n_jobs=1)
            casp15_protein_ligand_interaction_df["target"] = os.path.basename(
                protein_ligand_complex_filepath
            ).split("_lig")[0]
            casp15_protein_ligand_interaction_dfs.append(casp15_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing CASP15 target {protein_ligand_complex_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore("casp15_interaction_dataframes.h5") as store:
            for i, df in enumerate(casp15_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# #### Plot interaction statistics for each dataset

# %%
dfs = []


# define a function to process each dataset
def process_dataset(file_path, category):
    interactions = []
    with pd.HDFStore(file_path) as store:
        for key in store.keys():
            for row_index in range(len(store[key])):
                interaction_types = [
                    interaction[2]
                    for interaction in store[key].iloc[row_index].keys().tolist()
                    if interaction[2]  # NOTE: this excludes the `target` column's singular value
                ]
                num_hb_acceptors = interaction_types.count("HBAcceptor")
                num_hb_donors = interaction_types.count("HBDonor")
                num_vdw_contacts = interaction_types.count("VdWContact")
                num_hydrophobic = interaction_types.count("Hydrophobic")
                interactions.append(
                    {
                        "Hydrogen Bond Acceptors": num_hb_acceptors,
                        "Hydrogen Bond Donors": num_hb_donors,
                        "Van der Waals Contacts": num_vdw_contacts,
                        "Hydrophobic Interactions": num_hydrophobic,
                    }
                )
    df_rows = []
    for interaction in interactions:
        for interaction_type, num_interactions in interaction.items():
            df_rows.append(
                {
                    "Category": category,
                    "InteractionType": interaction_type,
                    "NumInteractions": num_interactions,
                }
            )
    return pd.DataFrame(df_rows)


# load data from files
if os.path.exists(
    f"pdbbind_training_subset_{pdbbind_train_subset_size}_interaction_dataframes.h5"
):
    dfs.append(
        process_dataset(
            f"pdbbind_training_subset_{pdbbind_train_subset_size}_interaction_dataframes.h5",
            "PDBBind 2020 (1000)",
        )
    )

if os.path.exists("astex_diverse_interaction_dataframes.h5"):
    dfs.append(process_dataset("astex_diverse_interaction_dataframes.h5", "Astex Diverse"))

if os.path.exists("posebusters_benchmark_interaction_dataframes.h5"):
    dfs.append(
        process_dataset("posebusters_benchmark_interaction_dataframes.h5", "PoseBusters Benchmark")
    )

if os.path.exists("dockgen_interaction_dataframes.h5"):
    dfs.append(process_dataset("dockgen_interaction_dataframes.h5", "DockGen"))

if os.path.exists("casp15_interaction_dataframes.h5"):
    dfs.append(process_dataset("casp15_interaction_dataframes.h5", "CASP15"))

# combine statistics
assert len(dfs) > 0, "No interaction dataframes found."
df = pd.concat(dfs)

# plot statistics
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)

interaction_types = [
    "Hydrogen Bond Acceptors",
    "Hydrogen Bond Donors",
    "Van der Waals Contacts",
    "Hydrophobic Interactions",
]
plot_types = ["box", "box", "violin", "violin"]

for ax, interaction, plot_type in zip(axes.flatten(), interaction_types, plot_types):
    data = df[df["InteractionType"] == interaction]

    if plot_type == "box":
        sns.boxplot(data=data, x="Category", y="NumInteractions", ax=ax, showfliers=True)
        sns.stripplot(
            data=data,
            x="Category",
            y="NumInteractions",
            ax=ax,
            color="black",
            alpha=0.3,
            jitter=True,
        )
    elif plot_type == "violin":
        sns.violinplot(data=data, x="Category", y="NumInteractions", ax=ax)
        sns.stripplot(
            data=data,
            x="Category",
            y="NumInteractions",
            ax=ax,
            color="black",
            alpha=0.3,
            jitter=True,
        )

    ax.set_title(interaction)
    ax.set_ylabel("No. Interactions")
    ax.set_xlabel("")
    ax.grid(True)

plt.tight_layout()
plt.savefig("dataset_interaction_analysis.png", dpi=300)
plt.show()
