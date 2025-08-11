# %% [markdown]
# ## CASP15 Method Interaction Analysis Plotting

# %% [markdown]
# #### Import packages

# %%
import gc
import glob
import os
import re
import shutil
import signal
import subprocess  # nosec
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Any, Literal
from Bio.PDB import PDBIO, PDBParser, Select
from posecheck import PoseCheck
from rdkit import Chem
from scipy.stats import ttest_rel, wasserstein_distance
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

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
    "vina_p2rank",
    "diffdock",
    "dynamicbind",
    "neuralplexer",
    "rfaa",
    "chai-lab_ss",
    "chai-lab",
    "boltz_ss",
    "boltz",
    "alphafold3_ss",
    "alphafold3",
]
max_num_repeats_per_method = (
    1  # NOTE: Here, to simplify the analysis, we only consider the first run of each method
)

casp15_set_dir = os.path.join(
    "..",
    "data",
    "casp15_set",
    "targets",
)
assert os.path.exists(
    casp15_set_dir
), "Please download the (public) CASP15 set from `https://zenodo.org/records/16791095` before proceeding."

# Mappings
method_mapping = {
    "vina_p2rank": "P2Rank-Vina",
    "diffdock": "DiffDock-L",
    "dynamicbind": "DynamicBind",
    "neuralplexer": "NeuralPLexer",
    "rfaa": "RFAA",
    "chai-lab_ss": "Chai-1-Single-Seq",
    "chai-lab": "Chai-1",
    "boltz_ss": "Boltz-1-Single-Seq",
    "boltz": "Boltz-1",
    "alphafold3_ss": "AF3-Single-Seq",
    "alphafold3": "AF3",
}

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
    input_pdb_filepath: str,
    molecule_type: Literal["protein", "ligand"],
    add_element_types: bool = False,
) -> str:
    """Create a temporary PDB file with only residues of a chosen molecule
    type.

    :param input_pdb_filepath: The input PDB file path.
    :param molecule_type: The molecule type to keep (either "protein" or
        "ligand").
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


def signal_handler(signum, frame):
    """Raise a runtime error when receiving a signal.

    :param signum: The signal number.
    :param frame: The frame.
    """
    raise RuntimeError("Received external interrupt (SIGUSR1)")


signal.signal(signal.SIGUSR1, signal_handler)

# %% [markdown]
# #### Compute interaction fingerprints

# %% [markdown]
# ##### Analyze `CASP15` set interactions as a baseline

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
            ligand_mol = Chem.MolFromPDBFile(temp_ligand_filepath)
            pc.load_protein_from_pdb(temp_protein_filepath)
            pc.load_ligands_from_mols(
                Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
            )
            casp15_protein_ligand_interaction_df = timeout(dec_timeout=600)(
                pc.calculate_interactions
            )(n_jobs=1)
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
# ##### Analyze interactions of each method

# %%
# calculate and cache CASP15 interaction statistics for each baseline method
dataset = "casp15"

for method in baseline_methods:
    for repeat_index in range(1, max_num_repeats_per_method + 1):
        method_title = method_mapping[method]

        if not os.path.exists(f"{method}_{dataset}_interaction_dataframes_{repeat_index}.h5"):
            method_casp15_set_dir = os.path.join(
                "..",
                "data",
                "test_cases",
                "casp15",
                f"top_{method}{'' if 'ensemble' in method else '_ensemble'}_predictions_{repeat_index}",
            )

            casp15_protein_ligand_complex_filepaths = []
            for item in os.listdir(method_casp15_set_dir):
                item_path = os.path.join(method_casp15_set_dir, item)
                if (
                    item.split("_")[0] not in CASP15_ANALYSIS_TARGETS_TO_SKIP
                    and os.path.isdir(item_path)
                    and "_relaxed" not in item
                ):
                    protein_pdb_filepath, ligand_sdf_filepath = None, None
                    complex_filepaths = glob.glob(
                        os.path.join(item_path, "*rank1*.pdb")
                    ) + glob.glob(os.path.join(item_path, "*rank1*.sdf"))
                    for file in complex_filepaths:
                        if file.endswith(".pdb"):
                            protein_pdb_filepath = file
                        elif file.endswith(".sdf"):
                            ligand_sdf_filepath = file
                    if protein_pdb_filepath is not None and ligand_sdf_filepath is not None:
                        casp15_protein_ligand_complex_filepaths.append(
                            (protein_pdb_filepath, ligand_sdf_filepath)
                        )
                    else:
                        raise FileNotFoundError(
                            f"Could not find `rank1` protein-ligand complex files for {item}"
                        )

            pc = PoseCheck()
            casp15_protein_ligand_interaction_dfs = []
            for protein_ligand_complex_filepath in tqdm(
                casp15_protein_ligand_complex_filepaths,
                desc=f"Processing interactions for {method_title}",
            ):
                protein_filepath, ligand_filepath = protein_ligand_complex_filepath
                casp15_target = os.path.basename(os.path.dirname(protein_filepath))
                print(f"Processing {method_title} target {casp15_target}...")
                try:
                    temp_protein_filepath = create_temp_pdb_with_only_molecule_type_residues(
                        protein_filepath, molecule_type="protein", add_element_types=True
                    )
                    num_residues_in_target_protein = count_num_residues_in_pdb_file(
                        temp_protein_filepath
                    )
                    if (
                        num_residues_in_target_protein
                        > MAX_CASP15_ANALYSIS_PROTEIN_SEQUENCE_LENGTH
                    ):
                        print(
                            f"{method_title} target {casp15_target} has too many protein residues ({num_residues_in_target_protein} > {MAX_CASP15_ANALYSIS_PROTEIN_SEQUENCE_LENGTH}) for `MDAnalysis` to fit into CPU memory. Skipping..."
                        )
                        continue
                    ligand_mol = Chem.MolFromMolFile(ligand_filepath)
                    pc.load_protein_from_pdb(temp_protein_filepath)
                    pc.load_ligands_from_mols(
                        Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
                    )
                    casp15_protein_ligand_interaction_df = timeout(
                        dec_timeout=600, use_signals=False
                    )(pc.calculate_interactions)(n_jobs=1)
                    casp15_protein_ligand_interaction_df["target"] = casp15_target
                    casp15_protein_ligand_interaction_dfs.append(
                        casp15_protein_ligand_interaction_df
                    )
                    gc.collect()
                except Exception as e:
                    print(
                        f"Error processing {method_title} target {casp15_target} due to: {e}. Skipping..."
                    )
                    continue

                # NOTE: we iteratively save the interaction dataframes to an HDF5 file
                with pd.HDFStore(
                    f"{method}_{dataset}_interaction_dataframes_{repeat_index}.h5"
                ) as store:
                    for i, df in enumerate(casp15_protein_ligand_interaction_dfs):
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
        file_path = f"{method}_casp15_interaction_dataframes_{repeat_index}.h5"
        if os.path.exists(file_path):
            dfs.append(process_method(file_path, method_title))

if os.path.exists("casp15_interaction_dataframes.h5"):
    dfs.append(process_method("casp15_interaction_dataframes.h5", "Reference"))

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
plt.savefig("casp15_method_interaction_analysis.png", dpi=300)
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
                target = store[key].iloc[row_index]["target"]
                if not isinstance(target, str):
                    target = target.values[0]

                try:
                    interactions[target].extend(
                        [
                            # NOTE: we use the `UNL` prefix to denote "unspecified" ligand types,
                            # since ProLIF cannot differentiate between ligand types for method predictions
                            f"UNL:{split_string_at_numeric(row[1])[0]}:{row[2]}"
                            # f"{split_string_at_numeric(row[0])[0]}:{split_string_at_numeric(row[1])[0]}:{row[2]}"
                            for row in store[key].iloc[row_index].index.values[:-1]
                        ]
                    )
                except Exception as e:
                    print(
                        f"Error processing {key} row {row_index} for target {target} due to: {e}. Skipping..."
                    )
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
        file_path = f"{method}_casp15_interaction_dataframes_{repeat_index}.h5"
        if os.path.exists(file_path):
            dfs.append(bin_interactions(file_path, method_title))

assert os.path.exists(
    "casp15_interaction_dataframes.h5"
), "No reference interaction dataframe found."
reference_df = bin_interactions("casp15_interaction_dataframes.h5", "Reference")

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
            # for a target, we skip-penalize it with a null EMD value
            emd_values.append({"Category": method, "Target": target, "EMD": np.nan})
            continue
        if reference_histogram.empty:
            # NOTE: if a target does not have any ProLIF-parseable interactions
            # in the reference data, we skip this target
            print(
                f"Skipping target {target} for method {method} due to missing reference interaction data."
            )
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
        try:
            emd = wasserstein_distance(method_histogram_vector, reference_histogram_vector)
        except Exception as e:
            emd = np.nan
            print(f"Skipping EMD computation for {method} target {target} due to: {e}")

        emd_values.append(
            {
                "Category": method,
                "Target": target,
                "EMD": emd,
                "Method_Histogram": dict(sorted(method_histogram.values[0].items())),
                "Reference_Histogram": dict(sorted(reference_histogram.values[0].items())),
            }
        )

# plot the EMD and WM values for each method
all_emd_values = [
    min(50.0, entry["EMD"]) for entry in emd_values
]  # clip EMD values to 50.0 when constructing WM values
min_emd = np.nanmin(all_emd_values)
max_emd = np.nanmax(all_emd_values)
for entry in emd_values:
    # NOTE: we normalize the EMD values to the range `[0, 1]`
    # to compute the Wasserstein Matching (WM) metric while
    # ensuring missing predictions are maximally skip-penalized
    emd = max_emd if np.isnan(entry["EMD"]).item() else min(50.0, entry["EMD"])
    normalized_score = 1 - (emd - min_emd) / (max_emd - min_emd)
    entry["WM"] = normalized_score

emd_values_df = pd.DataFrame(
    emd_values,
    columns=["Category", "Target", "EMD", "WM", "Method_Histogram", "Reference_Histogram"],
)
emd_values_df.to_csv("casp15_plif_metrics.csv")

plt.figure(figsize=(20, 8))
sns.boxplot(data=emd_values_df, x="Category", y="EMD")
plt.xlabel("")
plt.ylabel("PLIF-EMD")
plt.savefig("casp15_plif_emd_values.png")
plt.show()

plt.close("all")

plt.figure(figsize=(20, 8))
sns.boxplot(data=emd_values_df, x="Category", y="WM")
plt.xlabel("")
plt.ylabel("PLIF-WM")
plt.savefig("casp15_plif_wm_values.png")
plt.show()

plt.close("all")

# %% [markdown]
# #### Identify which types of interactions are most difficult to reproduce

# %%
# hypothesis: the most structured types of interactions (e.g., HBAcceptor, HBDonor) are more difficult to reproduce than unstructured types (e.g., VdWContact, Hydrophobic)
struct_emd_values = []
for _, row in emd_values_df.iterrows():
    method = row["Category"]
    target = row["Target"]

    method_histogram = row["Method_Histogram"]
    reference_histogram = row["Reference_Histogram"]

    if method_histogram is np.nan or reference_histogram is np.nan:
        continue

    # NOTE: collecting bins from both histograms allows us to penalize "hallucinated" interactions
    all_bins = set(method_histogram.keys()) | set(reference_histogram.keys())
    all_bins = sorted(all_bins)  # keep bins in a fixed order for consistency

    structured_interaction_bins = [
        bin for bin in all_bins if bin.split(":")[-1] in ("HBAcceptor", "HBDonor")
    ]
    unstructured_interaction_bins = [
        bin for bin in all_bins if bin.split(":")[-1] in ("VdWContact", "Hydrophobic")
    ]

    if not structured_interaction_bins or not unstructured_interaction_bins:
        continue

    # convert histograms to aligned vectors
    structured_method_histogram_vector = np.array(
        np.array([method_histogram.get(bin, 0) for bin in structured_interaction_bins])
    )
    structured_reference_histogram_vector = np.array(
        np.array([reference_histogram.get(bin, 0) for bin in structured_interaction_bins])
    )

    unstructured_method_histogram_vector = np.array(
        np.array([method_histogram.get(bin, 0) for bin in unstructured_interaction_bins])
    )
    unstructured_reference_histogram_vector = np.array(
        np.array([reference_histogram.get(bin, 0) for bin in unstructured_interaction_bins])
    )

    if (
        not structured_method_histogram_vector.any()
        or not structured_reference_histogram_vector.any()
    ):
        continue

    # compute the EMD values of each method's PLIF histograms
    try:
        structured_emd = wasserstein_distance(
            structured_method_histogram_vector, structured_reference_histogram_vector
        )
    except Exception as e:
        structured_emd = np.nan
        print(f"Skipping structured EMD computation for {method} target {target} due to: {e}")

    try:
        unstructured_emd = wasserstein_distance(
            unstructured_method_histogram_vector,
            unstructured_reference_histogram_vector,
        )
    except Exception as e:
        unstructured_emd = np.nan
        print(f"Skipping unstructured EMD computation for {method} target {target} due to: {e}")

    if structured_emd is np.nan or unstructured_emd is np.nan:
        continue

    struct_emd_values.append(
        {
            "Category": method,
            "Target": target,
            "Structured_EMD": structured_emd,
            "Unstructured_EMD": unstructured_emd,
            "Method_Histogram": dict(sorted(method_histogram.items())),
            "Reference_Histogram": dict(sorted(reference_histogram.items())),
        }
    )

struct_emd_values_df = pd.DataFrame(
    struct_emd_values,
    columns=[
        "Category",
        "Target",
        "Structured_EMD",
        "Unstructured_EMD",
        "Method_Histogram",
        "Reference_Histogram",
    ],
)
struct_emd_values_df.to_csv("casp15_structured_plif_metrics.csv")

# get unique categories
categories = struct_emd_values_df["Category"].unique()

# define colors and markers for each category
colors = plt.cm.get_cmap("tab10", len(categories))
markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

# create a scatter plot with lines connecting the paired values for each category
plt.figure(figsize=(20, 8))

for i, category in enumerate(categories):
    category_df = struct_emd_values_df[struct_emd_values_df["Category"] == category]

    # perform a paired t-test
    t_stat, p_value = ttest_rel(category_df["Structured_EMD"], category_df["Unstructured_EMD"])
    print(f"Category: {category} - T-statistic: {t_stat}, P-value: {p_value}")

    # plot the values
    plt.plot(
        category_df.index,
        category_df["Structured_EMD"],
        marker=markers[i % len(markers)],
        linestyle="-",
        color=colors(i),
        label=f"{category} Structured_EMD",
    )
    plt.plot(
        category_df.index,
        category_df["Unstructured_EMD"],
        marker=markers[i % len(markers)],
        linestyle="--",
        color=colors(i),
        label=f"{category} Unstructured_EMD",
    )

plt.xlabel("Index")
plt.ylabel("EMD Value")
plt.title("Comparison of Structured_EMD and Unstructured_EMD by Method")
plt.legend()
plt.ylim(0, 50)
plt.savefig("casp15_structured_vs_unstructured_emd_values.png")
plt.show()

plt.close("all")
