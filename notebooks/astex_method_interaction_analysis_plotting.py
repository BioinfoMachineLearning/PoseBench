# %% [markdown]
# ## Astex Diverse Method Interaction Analysis Plotting

# %% [markdown]
# #### Import packages

# %%
import os
import re
import shutil
import subprocess  # nosec
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Any, Literal
from Bio.PDB import PDBIO, PDBParser, Select
from omegaconf import DictConfig, open_dict
from posecheck import PoseCheck
from rdkit import Chem
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from posebench import resolve_method_input_csv_path, resolve_method_output_dir
from posebench.analysis.inference_analysis import create_mol_table
from posebench.utils.data_utils import count_num_residues_in_pdb_file

# %% [markdown]
# #### Configure packages

# %%
pd.options.mode.copy_on_write = True

# %% [markdown]
# #### Define constants

# %%
# General variables
baseline_methods = [
    "diffdock",
    "dynamicbind",
    "neuralplexer",
    "rfaa",
    "chai-lab",
    # "vina_p2rank",
]
max_num_repeats_per_method = (
    1  # NOTE: Here, to simplify the analysis, we only consider the first run of each method
)

ad_set_dir = os.path.join(
    "..",
    "data",
    "astex_diverse_set",
)
assert os.path.exists(
    ad_set_dir
), "Please download the Astex Diverse set from `https://zenodo.org/records/13858866` before proceeding."

# Mappings
method_mapping = {
    "diffdock": "DiffDock-L",
    "dynamicbind": "DynamicBind",
    "neuralplexer": "NeuralPLexer",
    "rfaa": "RoseTTAFold-AA",
    "chai-lab": "Chai-1",
    "vina_p2rank": "P2Rank-Vina",
}

MAX_ASTEX_DIVERSE_ANALYSIS_PROTEIN_SEQUENCE_LENGTH = (
    2000  # Only Astex Diverse targets with protein sequences below this threshold can be analyzed
)

# %% [markdown]
# #### Define utility functions


# %%
class ProteinSelect(Select):
    """A class to select only protein residues from a PDB file."""

    def accept_residue(self, residue: Any):
        """Only accept residues that are part of a protein (e.g., standard amino acids).

        :param residue: The residue to check.
        :return: True if the residue is part of a protein, False otherwise.
        """
        return residue.id[0] == " "  # NOTE: `HETATM` flag must be a blank for protein residues


class LigandSelect(Select):
    """A class to select only ligand residues from a PDB file."""

    def accept_residue(self, residue: Any):
        """Only accept residues that are part of a ligand.

        :param residue: The residue to check.
        :return: True if the residue is part of a ligand, False otherwise.
        """
        return residue.id[0] != " "  # NOTE: `HETATM` flag must be a filled for ligand residues


@beartype
def create_temp_pdb_with_only_molecule_type_residues(
    input_pdb_filepath: str,
    molecule_type: Literal["protein", "ligand"],
    add_element_types: bool = False,
) -> str:
    """Create a temporary PDB file with only residues of a chosen molecule type.

    :param input_pdb_filepath: The input PDB file path.
    :param molecule_type: The molecule type to keep (either "protein" or "ligand").
    :param add_element_types: Whether to add element types to the atoms.
    :return: The temporary PDB file path.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(molecule_type, input_pdb_filepath)

    io = PDBIO()
    io.set_structure(structure)

    # create a temporary PDB filepdb_name
    temp_pdb_filepath = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    io.save(
        temp_pdb_filepath.name, ProteinSelect() if molecule_type == "protein" else LigandSelect()
    )

    if add_element_types:
        with open(temp_pdb_filepath.name.replace(".pdb", "_elem.pdb"), "w") as f:
            subprocess.run(  # nosec
                f"pdb_element {temp_pdb_filepath.name}",
                shell=True,
                check=True,
                stdout=f,
            )
        shutil.move(temp_pdb_filepath.name.replace(".pdb", "_elem.pdb"), temp_pdb_filepath.name)

    return temp_pdb_filepath.name


# %% [markdown]
# #### Compute interaction fingerprints

# %% [markdown]
# ##### Analyze `Astex Diverse` set interactions as a baseline

# %%
if not os.path.exists("astex_diverse_interaction_dataframes.h5"):
    ad_protein_ligand_filepath_pairs = []
    for item in os.listdir(ad_set_dir):
        ligand_item_path = os.path.join(ad_set_dir, item)
        if os.path.isdir(ligand_item_path):
            protein_filepath = os.path.join(ligand_item_path, f"{item}_protein.pdb")
            ligand_filepath = os.path.join(ligand_item_path, f"{item}_ligand.sdf")
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
            ad_protein_ligand_interaction_df = pc.calculate_interactions()
            ad_protein_ligand_interaction_df["target"] = Path(protein_filepath).stem.split(
                "_protein"
            )[0]
            ad_protein_ligand_interaction_dfs.append(ad_protein_ligand_interaction_df)
        except Exception as e:
            print(
                f"Error processing Astex Diverse target {protein_filepath, ligand_filepath} due to: {e}. Skipping..."
            )
            continue

        # NOTE: we iteratively save the interaction dataframes to an HDF5 file
        with pd.HDFStore("astex_diverse_interaction_dataframes.h5") as store:
            for i, df in enumerate(ad_protein_ligand_interaction_dfs):
                store.put(f"df_{i}", df)

# %% [markdown]
# ##### Analyze interactions of each method

# %%
# calculate and cache Astex Diverse interaction statistics for each baseline method
config = ""  # NOTE: we do not calculate interactions for relaxed predictions currently
dataset = "astex_diverse"
ensemble_ranking_method = "consensus"
relax_protein = False
pocket_only_baseline = False

cfg = DictConfig(
    {
        "dataset": dataset,
        "relax_protein": relax_protein,
        "pocket_only_baseline": pocket_only_baseline,
        "input_data_dir": os.path.join("..", "data", f"{dataset}_set"),
        "posebusters_ccd_ids_filepath": os.path.join("..", "data", "posebusters_pdb_ccd_ids.txt"),
        "dockgen_test_ids_filepath": os.path.join("..", "data", "dockgen_set", "split_test.txt"),
    }
)

for method in baseline_methods:
    for repeat_index in range(1, max_num_repeats_per_method + 1):
        method_title = method_mapping[method]

        if not os.path.exists(f"{method}_{dataset}_interaction_dataframes_{repeat_index}.h5"):
            v1_baseline = method == "diffdockv1"
            vina_binding_site_method = method.split("_")[-1] if "_" in method else "p2rank"

            with open_dict(cfg):
                cfg.method = method
                cfg.repeat_index = repeat_index
                cfg.input_csv_path = str(
                    ".."
                    / Path(resolve_method_input_csv_path(method, dataset, pocket_only_baseline))
                )
                cfg.output_dir = str(
                    ".."
                    / Path(
                        resolve_method_output_dir(
                            method,
                            dataset,
                            vina_binding_site_method,
                            ensemble_ranking_method,
                            repeat_index,
                            pocket_only_baseline,
                            v1_baseline,
                        )
                    )
                )

            output_dir = cfg.output_dir + config
            if method == "dynamicbind":
                output_dir = os.path.join(
                    Path(output_dir).parent,
                    Path(output_dir).stem.replace(f"_{repeat_index}", ""),
                )

            mol_table = create_mol_table(
                Path(cfg.input_csv_path),
                Path(cfg.input_data_dir),
                Path(output_dir),
                cfg,
                relaxed="relaxed" in config,
                add_pdb_ids=True,
            )

            pc = PoseCheck()
            astex_protein_ligand_interaction_dfs = []
            for row in tqdm(
                mol_table.itertuples(index=False),
                desc=f"Processing interactions for {method_title}",
            ):
                try:
                    protein_filepath, ligand_filepath = str(row.mol_cond), str(row.mol_pred)
                    num_residues_in_target_protein = count_num_residues_in_pdb_file(
                        protein_filepath
                    )
                    if (
                        num_residues_in_target_protein
                        > MAX_ASTEX_DIVERSE_ANALYSIS_PROTEIN_SEQUENCE_LENGTH
                    ):
                        print(
                            f"{method_title} target {row} has too many protein residues ({num_residues_in_target_protein} > {MAX_ASTEX_DIVERSE_ANALYSIS_PROTEIN_SEQUENCE_LENGTH}) for `MDAnalysis` to fit into CPU memory. Skipping..."
                        )
                        continue
                    ligand_mol = Chem.MolFromMolFile(ligand_filepath)
                    pc.load_protein_from_pdb(protein_filepath)
                    pc.load_ligands_from_mols(
                        Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
                    )
                    protein_ligand_interaction_df = pc.calculate_interactions()
                    protein_ligand_interaction_df["target"] = row.pdb_id
                    astex_protein_ligand_interaction_dfs.append(protein_ligand_interaction_df)
                except Exception as e:
                    print(f"Error processing {method_title} target {row} due to: {e}. Skipping...")
                    continue

                # NOTE: we iteratively save the interaction dataframes to an HDF5 file
                with pd.HDFStore(
                    f"{method}_astex_diverse_interaction_dataframes_{repeat_index}.h5"
                ) as store:
                    for i, df in enumerate(astex_protein_ligand_interaction_dfs):
                        store.put(f"df_{i}", df)

# %% [markdown]
# #### Plot interaction statistics for each method

# %%
dfs = []


# define a function to process each method
def process_method(file_path, category):
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
for method in baseline_methods:
    for repeat_index in range(1, max_num_repeats_per_method + 1):
        method_title = method_mapping[method]
        file_path = f"{method}_astex_diverse_interaction_dataframes_{repeat_index}.h5"
        if os.path.exists(file_path):
            dfs.append(process_method(file_path, method_title))

if os.path.exists("astex_diverse_interaction_dataframes.h5"):
    dfs.append(process_method("astex_diverse_interaction_dataframes.h5", "Reference"))

# combine statistics
assert len(dfs) > 0, "No interaction dataframes found."
df = pd.concat(dfs)

# define font properties
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16

# plot statistics
fig, axes = plt.subplots(2, 2, figsize=(34, 14), sharey=False)

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
plt.savefig("astex_diverse_method_interaction_analysis.png", dpi=300)
plt.show()

# %% [markdown]
# #### Plot interaction metrics for each method

# %%
dfs = []


# define helper functions
def split_string_at_numeric(s: str) -> list:
    """Split a string at numeric characters."""
    return re.split(r"\d+", s)


def bin_interactions(file_path, category):
    """Bin interactions for each target."""
    interactions = defaultdict(list)
    with pd.HDFStore(file_path) as store:
        for key in store.keys():
            for row_index in range(len(store[key])):
                try:
                    interactions[store[key].iloc[row_index]["target"].values[0]].extend(
                        [
                            f"{split_string_at_numeric(row[0])[0]}:{split_string_at_numeric(row[1])[0]}:{row[2]}"
                            for row in store[key].iloc[row_index].index.values[:-1]
                        ]
                    )
                except Exception as e:
                    print(f"Error processing {key} row {row_index} due to: {e}. Skipping...")
                    continue
    df_rows = []
    for target in interactions:
        target_interactions = interactions[target]
        target_interactions_histogram = defaultdict(int)
        for target_interaction in target_interactions:
            target_interactions_histogram[target_interaction] += 1

        df_rows.append(
            {
                "Category": category,
                "Target": target,
                "Interactions_Histogram": target_interactions_histogram,
            }
        )
    return pd.DataFrame(df_rows)


def histogram_to_vector(histogram, bins):
    """Convert a histogram dictionary to a vector aligned with bins."""
    return np.array([histogram.get(bin, 0) for bin in bins])


# load data from files
for method in baseline_methods:
    for repeat_index in range(1, max_num_repeats_per_method + 1):
        method_title = method_mapping[method]
        file_path = f"{method}_astex_diverse_interaction_dataframes_{repeat_index}.h5"
        if os.path.exists(file_path):
            dfs.append(bin_interactions(file_path, method_title))

assert os.path.exists(
    "astex_diverse_interaction_dataframes.h5"
), "No reference Astex Diverse interaction dataframe found."
reference_df = bin_interactions("astex_diverse_interaction_dataframes.h5", "Reference")

# combine bins from all method dataframes
assert len(dfs) > 0, "No interaction dataframes found."
df = pd.concat(dfs)

emd_values = []
for method in df["Category"].unique():
    for target in reference_df["Target"]:
        # step 1: extract unique bins for each pair of method and reference histograms
        method_histogram = df[(df["Category"] == method) & (df["Target"] == target)][
            "Interactions_Histogram"
        ]
        reference_histogram = reference_df[reference_df["Target"] == target][
            "Interactions_Histogram"
        ]
        if method_histogram.empty:
            # NOTE: if a method does not have any ProLIF-parseable interactions
            # for a target, we skip-penalize it with a maximum EMD value
            emd_values.append({"Category": method, "Target": target, "EMD": 1.0})
            continue

        # NOTE: collecting bins from both histograms allows us to penalize "hallucinated" interactions
        all_bins = set(method_histogram.values[0].keys()) | set(
            reference_histogram.values[0].keys()
        )
        all_bins = sorted(all_bins)  # keep bins in a fixed order for consistency

        # step 2: convert histograms to aligned vectors
        method_histogram_vector = method_histogram.apply(
            lambda h: histogram_to_vector(h, all_bins)
        ).squeeze()
        reference_histogram_vector = reference_histogram.apply(
            lambda h: histogram_to_vector(h, all_bins)
        ).squeeze()

        # step 3: compute the EMD values of each method's PLIF histograms
        emd_values.append(
            {
                "Category": method,
                "Target": target,
                "EMD": wasserstein_distance(method_histogram_vector, reference_histogram_vector),
            }
        )


# plot the EMD values for each method
emd_values_df = pd.DataFrame(emd_values, columns=["Category", "Target", "EMD"])
emd_values_df.to_csv("astex_diverse_plif_emd_values.csv")

plt.figure(figsize=(10, 5))
sns.boxplot(data=emd_values_df, x="Category", y="EMD")
plt.xlabel("")
plt.ylabel("PLIF-EMD")
plt.savefig("astex_diverse_plif_emd_values.png")
plt.show()