# Adapted from: https://github.com/gcorso/DiffDock

import ast
import copy
import glob
import os
import pickle  # nosec
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Optional
from prody import confProDy
from rdkit import Chem
from scipy.spatial import distance
from torch_geometric.data import Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import process_molecule
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    centralize_complex_graph,
    combine_molecules,
    merge_protein_and_ligands,
    pdb_filepath_to_protein,
    process_protein,
)
from flowdock.utils.model_utils import sample_inplace_to_torch
from flowdock.utils.utils import fasta_to_dict, read_strings_from_txt

log = RankedLogger(__name__, rank_zero_only=True)

confProDy(verbosity="none")


class BindingMOADDataset(Dataset):
    """PyTorch Geometric Dataset for BindingMOAD dataset."""

    def __init__(
        self,
        root,
        dockgen_dir,
        clusters_filepath,
        transform=None,
        cache_path=os.path.join("data", "cache"),
        split="train",
        limit_complexes=0,
        num_workers=0,
        max_lig_size=None,
        min_multi_lig_distance=None,
        remove_hs=False,
        esm_embeddings_path=None,
        dockgen_esm_embeddings_path=None,
        esm_embeddings_sequences_path=None,
        dockgen_esm_embeddings_sequences_path=None,
        apo_protein_structure_dir=None,
        dockgen_apo_protein_structure_dir=None,
        require_ligand=False,
        include_miscellaneous_atoms=False,
        min_ligand_size=0,
        multiplicity=1,
        max_receptor_size=None,
        remove_promiscuous_targets=None,
        unroll_clusters=False,
        remove_pdbbind=False,
        enforce_timesplit=False,
        no_randomness=False,
        single_cluster_name=None,
        total_dataset_size=None,
        pdbbind_split_train=None,
        pdbbind_split_val=None,
        split_time=None,
        min_protein_length: Optional[int] = 10,
        max_protein_length: Optional[int] = 4000,
        is_test_dataset=False,
        a2h_assessment_csv_filepath=None,
        dockgen_a2h_assessment_csv_filepath=None,
        filter_using_a2h_assessment=False,
        a2h_min_tmscore=None,
        a2h_max_rmsd=None,
        a2h_min_protein_length=None,
        a2h_max_protein_length=None,
        a2h_min_ligand_length=None,
        a2h_max_ligand_length=None,
        binding_affinity_values_dict=None,
        n_lig_patches=32,
    ):
        """Initializes the BindingMOADDataset."""
        if dockgen_a2h_assessment_csv_filepath is not None:
            assert (
                split != "train"
            ), "DockGen `a2h` filtering is only for validation and test datasets."

        super().__init__(root, transform)
        self.moad_dir = root
        self.dockgen_dir = dockgen_dir
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.min_multi_lig_distance = min_multi_lig_distance
        self.split = split
        self.limit_complexes = limit_complexes
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.require_ligand = require_ligand
        self.esm_embeddings_path = esm_embeddings_path
        self.dockgen_esm_embeddings_path = dockgen_esm_embeddings_path
        self.esm_embeddings_sequences_path = esm_embeddings_sequences_path
        self.dockgen_esm_embeddings_sequences_path = dockgen_esm_embeddings_sequences_path
        self.apo_protein_structure_dir = apo_protein_structure_dir
        self.dockgen_apo_protein_structure_dir = dockgen_apo_protein_structure_dir
        self.multiplicity = multiplicity
        self.no_randomness = no_randomness
        self.total_dataset_size = total_dataset_size
        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length
        self.is_test_dataset = is_test_dataset
        self.binding_affinity_values_dict = binding_affinity_values_dict
        self.n_lig_patches = n_lig_patches

        self.prot_cache_path = os.path.join(
            cache_path,
            f"MOAD_limit{self.limit_complexes}_INDEX{self.split}"
            + (
                ""
                if (
                    self.dockgen_esm_embeddings_path
                    if self.split != "train"
                    else self.esm_embeddings_path
                )
                is None
                else "_esmEmbeddings"
            )
            + ("" if not self.include_miscellaneous_atoms else "_miscAtoms")
            + ("" if min_protein_length is None else f"_minProteinLength{min_protein_length}")
            + ("" if max_protein_length is None else f"_maxProteinLength{max_protein_length}"),
        )

        self.lig_cache_path = os.path.join(
            cache_path,
            f"MOAD_limit{self.limit_complexes}_INDEX{self.split}"
            + f"_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}"
            + (
                ""
                if self.min_multi_lig_distance is None
                else f"_minMultiLigDist{self.min_multi_lig_distance}"
            ),
        )

        self.single_cluster_name = single_cluster_name
        if split == "train":
            split = "PDBBind"

        with open(clusters_filepath, "rb") as f:
            self.split_clusters = pickle.load(f)[split]  # nosec

        clusters_path = os.path.join(self.moad_dir, "new_cluster_to_ligands.pkl")
        with open(clusters_path, "rb") as f:
            self.cluster_to_ligands = pickle.load(f)  # nosec

        if not self.check_all_receptors():
            os.makedirs(self.prot_cache_path, exist_ok=True)
            self.preprocessing_receptors()

        if not os.path.exists(os.path.join(self.lig_cache_path, "ligands.pkl")):
            os.makedirs(self.lig_cache_path, exist_ok=True)
            self.preprocessing_ligands()

        log.info("Loading ligands from memory: ", os.path.join(self.lig_cache_path, "ligands.pkl"))
        with open(os.path.join(self.lig_cache_path, "ligands.pkl"), "rb") as f:
            self.ligands = pickle.load(f)  # nosec

        if require_ligand:
            with open(os.path.join(self.lig_cache_path, "rdkit_ligands.pkl"), "rb") as f:
                self.rdkit_ligands = pickle.load(f)  # nosec
                self.rdkit_ligands = {
                    ligs[0]["metadata"]["sample_ID"]: mol
                    for mol, ligs in zip(self.rdkit_ligands, self.ligands)
                }

        len_before = len(self.ligands)
        if self.single_cluster_name is not None:
            self.ligands = [
                ligs
                for ligs in self.ligands
                if ligs[0]["metadata"]["sample_ID"]
                in self.cluster_to_ligands[self.single_cluster_name]
            ]
        log.info(
            f"Kept {len(self.ligands)} ligands in {self.single_cluster_name} out of {len_before}"
        )

        len_before = len(self.ligands)
        self.ligands = {
            ligs[0]["metadata"]["sample_ID"]: ligs
            for ligs in self.ligands
            if min_ligand_size == 0
            or all(lig["features"]["sdf_coordinates"].shape[0] >= min_ligand_size for lig in ligs)
        }
        log.info(
            f"Removed {len_before - len(self.ligands)} ligands below minimum size out of {len_before}"
        )

        receptors_names = {
            (
                ligs[0]["metadata"]["sample_ID"]
                if self.split != "train"
                else ligs[0]["metadata"]["sample_ID"][:6]
            )
            for ligs in self.ligands.values()
        }
        self.collect_receptors(receptors_names, max_receptor_size, remove_promiscuous_targets)

        # filter ligands for which the receptor failed
        tot_before = len(self.ligands)
        self.ligands = {
            k: v
            for k, v in self.ligands.items()
            if (k if self.split != "train" else k[:6]) in self.receptors
        }
        log.info(
            f"Removed {tot_before - len(self.ligands)} ligands with no receptor out of {tot_before}"
        )

        if remove_pdbbind:
            assert (
                pdbbind_split_train is not None and pdbbind_split_val is not None
            ), "PDBBind splits must be provided"
            complexes_pdbbind = read_strings_from_txt(pdbbind_split_train) + read_strings_from_txt(
                pdbbind_split_val
            )
            with open(os.path.join(self.moad_dir, "pdbbind_clusters_ecod.pkl"), "rb") as f:
                pdbbind_to_cluster = pickle.load(f)  # nosec
            clusters_pdbbind = {pdbbind_to_cluster[c] for c in complexes_pdbbind}
            self.split_clusters = [c for c in self.split_clusters if c not in clusters_pdbbind]
            self.cluster_to_ligands = {
                k: v for k, v in self.cluster_to_ligands.items() if k not in clusters_pdbbind
            }
            ligand_accepted = []
            for c, ligands in self.cluster_to_ligands.items():
                ligand_accepted += [(lig if self.split != "train" else lig[:6]) for lig in ligands]
            ligand_accepted = set(ligand_accepted)
            tot_before = len(self.ligands)
            self.ligands = {k: v for k, v in self.ligands.items() if k in ligand_accepted}
            log.info(
                f"Removed {tot_before - len(self.ligands)} ligands in overlap with PDBBind out of {tot_before}"
            )

        if enforce_timesplit:
            assert split_time is not None, "Time split must be provided"
            with open(split_time) as f:
                lines = f.readlines()
            pdbids_from2019 = []
            for i in range(6, len(lines), 4):
                pdbids_from2019.append(lines[i][18:22])

            pdbids_from2019 = set(pdbids_from2019)
            len_before = len(self.ligands)
            self.ligands = {
                k: v for k, v in self.ligands.items() if k[:4].upper() not in pdbids_from2019
            }
            log.info(
                f"Removed {len_before - len(self.ligands)} ligands from 2019 out of {len_before}"
            )

        if unroll_clusters:
            rec_keys = {(k if self.split != "train" else k[:6]) for k in self.ligands.keys()}
            self.cluster_to_ligands = {
                k: [
                    k2
                    for k2 in self.ligands.keys()
                    if (k2 if self.split != "train" else k2[:6]) == k
                ]
                for k in rec_keys
            }
            self.split_clusters = list(rec_keys)
        else:
            for c in self.cluster_to_ligands.keys():
                self.cluster_to_ligands[c] = [
                    v
                    for v in self.cluster_to_ligands[c]
                    if (v if self.split != "train" else v[:6]) in self.ligands
                ]
            self.split_clusters = [
                c for c in self.split_clusters if len(self.cluster_to_ligands[c]) > 0
            ]

        # analyze and potentially filter the Binding MOAD dataset based on its apo-to-holo (a2h) structural assessment
        if a2h_assessment_csv_filepath is not None and os.path.exists(a2h_assessment_csv_filepath):
            a2h_assessment_df = pd.read_csv(a2h_assessment_csv_filepath)
            a2h_assessment_df["ID"] = [
                "_".join(Path(paths[0]).stem.split("_")[:2])
                for paths in a2h_assessment_df["Filepath"].apply(ast.literal_eval).tolist()
            ]
            ligand_num_atoms = [
                [int(num_atoms) for num_atoms in num_atoms_str.split(",")]
                for num_atoms_str in a2h_assessment_df["Ligand_Num_Atoms"].tolist()
            ]
            a2h_assessment_df["Ligand_Total_Num_Atoms"] = np.array(
                [np.array(num_atoms).sum() for num_atoms in ligand_num_atoms]
            )
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # plot_dir = Path(a2h_assessment_csv_filepath).parent / "plots"
            # plot_dir.mkdir(exist_ok=True)
            # plt.clf()
            # sns.histplot(a2h_assessment_df["TM-score"])
            # plt.title("Apo-To-Holo Protein TM-score")
            # plt.savefig(plot_dir / "a2h_TM-score_hist.png")
            # plt.clf()
            # sns.histplot(a2h_assessment_df["RMSD"])
            # plt.title("Apo-To-Holo Protein RMSD")
            # plt.savefig(plot_dir / "a2h_RMSD_hist.png")
            # plt.clf()
            # plt.xlim(0, 1500)
            # sns.histplot(a2h_assessment_df["Apo_Length"])
            # plt.title("Apo Protein Length")
            # plt.savefig(plot_dir / "apo_length_hist.png")
            # plt.clf()
            # plt.xlim(0, 500)
            # sns.histplot(a2h_assessment_df["Ligand_Total_Num_Atoms"])
            # plt.title("Ligand Total Number of Atoms")
            # plt.savefig(plot_dir / "ligand_total_num_atoms_hist.png")
            if filter_using_a2h_assessment and not is_test_dataset:
                log.info(
                    f"Filtering the Binding MOAD {self.split} dataset based on its apo-to-holo (a2h) structural assessment"
                )
                a2h_assessment_df = a2h_assessment_df[
                    (a2h_assessment_df["TM-score"] >= a2h_min_tmscore)
                    & (a2h_assessment_df["RMSD"] <= a2h_max_rmsd)
                    & (a2h_assessment_df["Apo_Length"] >= a2h_min_protein_length)
                    & (a2h_assessment_df["Apo_Length"] <= a2h_max_protein_length)
                    & (a2h_assessment_df["Ligand_Total_Num_Atoms"] >= a2h_min_ligand_length)
                    & (a2h_assessment_df["Ligand_Total_Num_Atoms"] <= a2h_max_ligand_length)
                ]
                a2h_filtered_ids = {id: None for id in a2h_assessment_df["ID"].tolist()}
                self.ligands = {
                    ligand: self.ligands[ligand]
                    for ligand in self.ligands
                    if ligand.lower() in a2h_filtered_ids
                }
                self.receptors = {
                    receptor: self.receptors[receptor]
                    for receptor in self.receptors
                    if receptor.lower() in a2h_filtered_ids
                }
                new_split_clusters = []
                for cluster in self.split_clusters:
                    self.cluster_to_ligands[cluster] = [
                        ligand
                        for ligand in self.cluster_to_ligands[cluster]
                        if "_".join(ligand.lower().split("_")[:2]) in a2h_filtered_ids
                    ]
                    if len(self.cluster_to_ligands[cluster]) > 0:
                        new_split_clusters.append(cluster)
                self.split_clusters = new_split_clusters

        # analyze the DockGen dataset based on its apo-to-holo (a2h) structural assessment
        if dockgen_a2h_assessment_csv_filepath is not None and os.path.exists(
            dockgen_a2h_assessment_csv_filepath
        ):
            dockgen_a2h_assessment_df = pd.read_csv(dockgen_a2h_assessment_csv_filepath)
            dockgen_a2h_assessment_df["ID"] = [
                Path(paths[0]).stem.split("_holo_aligned")[0]
                for paths in dockgen_a2h_assessment_df["Filepath"].apply(ast.literal_eval).tolist()
            ]
            dockgen_ligand_num_atoms = [
                [num_atoms] for num_atoms in dockgen_a2h_assessment_df["Ligand_Num_Atoms"].tolist()
            ]
            dockgen_a2h_assessment_df["Ligand_Total_Num_Atoms"] = np.array(
                [np.array(num_atoms).sum() for num_atoms in dockgen_ligand_num_atoms]
            )
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # dockgen_plot_dir = Path(dockgen_a2h_assessment_csv_filepath).parent / "plots"
            # dockgen_plot_dir.mkdir(exist_ok=True)
            # plt.clf()
            # sns.histplot(dockgen_a2h_assessment_df["TM-score"])
            # plt.title("Apo-To-Holo Protein TM-score")
            # plt.savefig(dockgen_plot_dir / "a2h_TM-score_hist.png")
            # plt.clf()
            # sns.histplot(dockgen_a2h_assessment_df["RMSD"])
            # plt.title("Apo-To-Holo Protein RMSD")
            # plt.savefig(dockgen_plot_dir / "a2h_RMSD_hist.png")
            # plt.clf()
            # plt.xlim(0, 2000)
            # sns.histplot(dockgen_a2h_assessment_df["Apo_Length"])
            # plt.title("Apo Protein Length")
            # plt.savefig(dockgen_plot_dir / "apo_length_hist.png")
            # plt.clf()
            # plt.xlim(0, 150)
            # sns.histplot(dockgen_a2h_assessment_df["Ligand_Total_Num_Atoms"])
            # plt.title("Ligand Total Number of Atoms")
            # plt.savefig(dockgen_plot_dir / "ligand_total_num_atoms_hist.png")
            if filter_using_a2h_assessment and not is_test_dataset:
                log.info(
                    f"Filtering the DockGen {self.split} dataset based on its apo-to-holo (a2h) structural assessment"
                )
                dockgen_a2h_assessment_df = dockgen_a2h_assessment_df[
                    (dockgen_a2h_assessment_df["TM-score"] >= a2h_min_tmscore)
                    & (dockgen_a2h_assessment_df["RMSD"] <= a2h_max_rmsd)
                    & (dockgen_a2h_assessment_df["Apo_Length"] >= a2h_min_protein_length)
                    & (dockgen_a2h_assessment_df["Apo_Length"] <= a2h_max_protein_length)
                    & (
                        dockgen_a2h_assessment_df["Ligand_Total_Num_Atoms"]
                        >= a2h_min_ligand_length
                    )
                    & (
                        dockgen_a2h_assessment_df["Ligand_Total_Num_Atoms"]
                        <= a2h_max_ligand_length
                    )
                ]
                dockgen_a2h_filtered_ids = {
                    id: None for id in dockgen_a2h_assessment_df["ID"].tolist()
                }
                self.ligands = {
                    ligand: self.ligands[ligand]
                    for ligand in self.ligands
                    if ligand.replace(" ", "-") in dockgen_a2h_filtered_ids
                }
                self.receptors = {
                    receptor: self.receptors[receptor]
                    for receptor in self.receptors
                    if receptor.replace(" ", "-") in dockgen_a2h_filtered_ids
                }
                new_split_clusters = []
                for cluster in self.split_clusters:
                    self.cluster_to_ligands[cluster] = [
                        ligand
                        for ligand in self.cluster_to_ligands[cluster]
                        if ligand.replace(" ", "-") in dockgen_a2h_filtered_ids
                    ]
                    if len(self.cluster_to_ligands[cluster]) > 0:
                        new_split_clusters.append(cluster)
                self.split_clusters = new_split_clusters

        list_names = [
            name for cluster in self.split_clusters for name in self.cluster_to_ligands[cluster]
        ]
        log.info(
            f"{len(list_names)} total complexes available from {len(self.split_clusters)} clusters after all {self.split} filtering"
        )
        with open(os.path.join(self.prot_cache_path, f"moad_{self.split}_names.txt"), "w") as f:
            f.write("\n".join(list_names))

    def len(self):
        """Returns the number of complexes in the dataset."""
        return (
            len(self.split_clusters) * self.multiplicity
            if self.total_dataset_size is None
            else self.total_dataset_size
        )

    def get_by_name(self, ligand_name):
        """Returns the complex graph for a given ligand name."""
        lig_samples = copy.deepcopy(self.ligands[ligand_name])
        protein_samples = copy.deepcopy(
            self.receptors[(ligand_name if self.split != "train" else ligand_name[:6])]
        )
        complex_graph = merge_protein_and_ligands(
            lig_samples,
            protein_samples,
            n_lig_patches=self.n_lig_patches,
        )
        if self.require_ligand:
            complex_graph["metadata"]["mol"] = copy.deepcopy(self.rdkit_ligands[ligand_name])
        return complex_graph

    def get(self, idx):
        """Returns the complex graph for a given index."""
        if self.total_dataset_size is not None:
            raise NotImplementedError(
                "Random sampling not supported for total_dataset_size currently"
            )
            idx = random.randint(0, len(self.split_clusters) - 1)  # nosec

        idx = idx % len(self.split_clusters)
        cluster = self.split_clusters[idx]

        if self.no_randomness:
            ligand_name = sorted(self.cluster_to_ligands[cluster])[0]
        else:
            ligand_name = random.choice(self.cluster_to_ligands[cluster])  # nosec

        complex_graph = sample_inplace_to_torch(
            self.get_by_name(ligand_name if self.split != "train" else ligand_name[:6])
        )

        if self.binding_affinity_values_dict is not None:
            try:
                # associate (super)ligands with their binding affinities;
                # NOTE: this assumes that the number of available binding
                # affinities for a given complex is equal to the number of
                # ligands in the complex, which may not always be the case
                ligand_code = ligand_name[:4].upper()
                ligand_binding_affinities = list(
                    self.binding_affinity_values_dict[ligand_code].values()
                )
                assert (
                    len(ligand_binding_affinities) == complex_graph["metadata"]["num_molid"]
                ), f"Mismatch between the number of ligands and available binding affinities for {ligand_name}."
                complex_graph["features"]["affinity"] = torch.tensor(
                    ligand_binding_affinities, dtype=torch.float32
                )
            except Exception as e:
                log.info(
                    f"Failed to get binding affinity for {ligand_name} due to: {e}. Substituting binding affinity value(s) with `NaN`."
                )
                complex_graph["features"]["affinity"] = torch.tensor(
                    [torch.nan for _ in range(complex_graph["metadata"]["num_molid"])],
                    dtype=torch.float32,
                )
        else:
            complex_graph["features"]["affinity"] = torch.tensor(
                [torch.nan for _ in range(complex_graph["metadata"]["num_molid"])],
                dtype=torch.float32,
            )

        return centralize_complex_graph(complex_graph)

    def get_all_complexes(self):
        """Returns all the complexes in the dataset."""
        complexes = {}
        for cluster in self.split_clusters:
            for ligand_name in self.cluster_to_ligands[cluster]:
                complexes[ligand_name] = self.get_by_name(ligand_name)
        return complexes

    def preprocessing_receptors(self):
        """Preprocesses the receptors and saves them to the cache path."""
        log.info(
            f"Processing receptors from [{self.split}] and saving them to [{self.prot_cache_path}]"
        )

        complex_names_all = sorted(
            [item for cluster in self.split_clusters for item in self.cluster_to_ligands[cluster]]
        )
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]

        receptor_names_all = [
            (item if self.split != "train" else item[:6]) for item in complex_names_all
        ]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))
        log.info(f"Loading {len(receptor_names_all)} receptors.")

        esm_embeddings_path = (
            self.dockgen_esm_embeddings_path if self.split != "train" else self.esm_embeddings_path
        )
        esm_embeddings_sequences_path = (
            self.dockgen_esm_embeddings_sequences_path
            if self.split != "train"
            else self.esm_embeddings_sequences_path
        )

        if esm_embeddings_path is not None:
            log.info("Loading ESM embeddings")
            sequences_to_embeddings = {}
            receptor_names_all_dict = {name: None for name in receptor_names_all}
            sequences_dict = fasta_to_dict(esm_embeddings_sequences_path)
            for embedding_filepath in os.listdir(esm_embeddings_path):
                key = Path(embedding_filepath).stem
                if key.split("_chain_")[0].replace("-", " ") in receptor_names_all_dict:
                    embedding = torch.load(os.path.join(esm_embeddings_path, embedding_filepath))[
                        "representations"
                    ][33]
                    seq = sequences_dict[key]
                    id = int(key.split("_chain_")[1])
                    sequences_to_embeddings[seq + f":{id}"] = embedding
        else:
            sequences_to_embeddings = None

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(receptor_names_all) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.prot_cache_path, f"receptors{i}.pkl")):
                continue
            receptor_names = receptor_names_all[1000 * i : 1000 * (i + 1)]
            receptor_graphs = []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(receptor_names),
                desc=f"Loading receptors {i}/{len(receptor_names_all)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.get_receptor,
                    zip(receptor_names, [sequences_to_embeddings] * len(receptor_names)),
                ):
                    if t is not None:
                        log.info(len(receptor_graphs))
                        receptor_graphs.append(t)
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            log.info(f"Number of receptors: {len(receptor_graphs)}")
            with open(os.path.join(self.prot_cache_path, f"receptors{i}.pkl"), "wb") as f:
                pickle.dump((receptor_graphs), f)
        return receptor_names_all

    def check_all_receptors(self):
        """Checks if all the receptors are preprocessed."""
        complex_names_all = sorted(
            [item for c in self.split_clusters for item in self.cluster_to_ligands[c]]
        )
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        receptor_names_all = [item[:6] for item in complex_names_all]
        receptor_names_all = list(dict.fromkeys(receptor_names_all))
        for i in range(len(receptor_names_all) // 1000 + 1):
            if not os.path.exists(os.path.join(self.prot_cache_path, f"receptors{i}.pkl")):
                return False
        return True

    def collect_receptors(
        self, receptors_to_keep=None, max_receptor_size=None, remove_promiscuous_targets=None
    ):
        """Collects the receptors to keep and saves them to the receptors dictionary."""
        complex_names_all = sorted(
            [item for c in self.split_clusters for item in self.cluster_to_ligands[c]]
        )
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        receptor_names_all = [
            (item if self.split != "train" else item[:6]) for item in complex_names_all
        ]
        receptor_names_all = sorted(list(dict.fromkeys(receptor_names_all)))

        receptor_graphs_all = []
        total_recovered = 0
        log.info(f"Loading {len(receptor_names_all)} receptors to keep {len(receptors_to_keep)}.")
        for i in range(len(receptor_names_all) // 1000 + 1):
            log.info(f'Prot path: {os.path.join(self.prot_cache_path, f"receptors{i}.pkl")}')
            with open(os.path.join(self.prot_cache_path, f"receptors{i}.pkl"), "rb") as f:
                item = pickle.load(f)  # nosec
                total_recovered += len(item)
                if receptors_to_keep is not None:
                    item = [t for t in item if t["metadata"]["sample_ID"] in receptors_to_keep]
                receptor_graphs_all.extend(item)

        cur_len = len(receptor_graphs_all)
        log.info(
            f"Kept {len(receptor_graphs_all)} receptors out of {len(receptor_names_all)} total and recovered {total_recovered}"
        )

        if max_receptor_size is not None:
            receptor_graphs_all = [
                rec
                for rec in receptor_graphs_all
                if rec["features"]["res_atom_positions"].shape[0] <= max_receptor_size
            ]
            log.info(
                f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after filtering by size"
            )
            cur_len = len(receptor_graphs_all)

        if remove_promiscuous_targets is not None:
            promiscuous_targets = set()
            for name in complex_names_all:
                item = name.split("_")
                if int(item[3]) > remove_promiscuous_targets:
                    promiscuous_targets.add(name[:6])
            receptor_graphs_all = [
                rec
                for rec in receptor_graphs_all
                if rec["metadata"]["sample_ID"] not in promiscuous_targets
            ]
            log.info(
                f"Kept {len(receptor_graphs_all)} receptors out of {cur_len} after removing promiscuous targets"
            )

        self.receptors = {}
        for r in receptor_graphs_all:
            self.receptors[r["metadata"]["sample_ID"]] = r

    def get_receptor(self, par):
        """Returns the receptor graph for a given receptor name."""
        name, sequences_to_embeddings = par
        if self.split == "train":
            holo_protein_filepath = os.path.join(
                self.moad_dir, "pdb_protein", f"{name}_protein.pdb"
            )
        else:
            holo_protein_filepath = os.path.join(
                self.dockgen_dir,
                name.replace(" ", "-"),
                f"{name.replace(' ', '-')}_protein_processed.pdb",
            )
        if not os.path.exists(holo_protein_filepath):
            log.error(f"Holo receptor not found for {name}: {holo_protein_filepath}")
            return None

        try:
            holo_af_protein = pdb_filepath_to_protein(holo_protein_filepath)
            holo_protein_sample = process_protein(
                holo_af_protein,
                sample_name=f"{name}_",
            )
            complex_graph = holo_protein_sample

        except Exception as e:
            log.error(f"Skipping holo {name} because of the error: {e}")
            return None

        if np.isnan(complex_graph["features"]["res_atom_positions"]).any():
            log.error(
                f"NaN in holo receptor pos for {name}. Skipping preprocessing for this example..."
            )
            return None

        try:
            apo_protein_structure_dir = (
                self.dockgen_apo_protein_structure_dir
                if self.split != "train"
                else self.apo_protein_structure_dir
            )
            if apo_protein_structure_dir is not None:
                apo_protein_filepath = os.path.join(
                    apo_protein_structure_dir,
                    f"{name.replace(' ', '-')}_holo_aligned_esmfold_protein.pdb",
                )
                apo_af_protein = pdb_filepath_to_protein(apo_protein_filepath)
                apo_protein_sample = process_protein(
                    apo_af_protein,
                    sample_name=f"{name}_",
                    sequences_to_embeddings=sequences_to_embeddings,
                )
                for key in complex_graph.keys():
                    for subkey, value in apo_protein_sample[key].items():
                        complex_graph[key]["apo_" + subkey] = value
                if not np.array_equal(
                    complex_graph["features"]["res_type"],
                    complex_graph["features"]["apo_res_type"],
                ):
                    log.error(
                        f"Residue type mismatch between holo protein and apo protein for {name}. Skipping preprocessing for this example..."
                    )
                    return None
                if np.isnan(complex_graph["features"]["apo_res_atom_positions"]).any():
                    log.error(
                        f"NaN in apo receptor pos for {name}. Skipping preprocessing for this example..."
                    )
                    return None

        except Exception as e:
            log.error(f"Skipping apo {name} because of the error: {e}")
            return None

        if (
            self.min_protein_length is not None
            and complex_graph["metadata"]["num_a"] < self.min_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return None
        if (
            self.max_protein_length is not None
            and complex_graph["metadata"]["num_a"] > self.max_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return None

        complex_graph["metadata"]["sample_ID"] = name
        return complex_graph

    def preprocessing_ligands(self):
        """Preprocesses the ligands and saves them to the cache path."""
        log.info(
            f"Processing complexes from [{self.split}] and saving them to [{self.lig_cache_path}]"
        )

        complex_names_all = sorted(
            [item for c in self.split_clusters for item in self.cluster_to_ligands[c]]
        )
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]

        complex_names_all = [
            (item if self.split != "train" else item[:6]) for item in complex_names_all
        ]
        complex_names_all = sorted(list(dict.fromkeys(complex_names_all)))
        log.info(f"Loading {len(complex_names_all)} (multi-)ligands.")

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(complex_names_all) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.lig_cache_path, f"ligands{i}.pkl")):
                continue
            complex_names = complex_names_all[1000 * i : 1000 * (i + 1)]
            ligand_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(complex_names),
                desc=f"Loading complexes {i}/{len(complex_names_all)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(self.get_ligand, complex_names):
                    if t is not None:
                        ligand_graphs.append(t[0])
                        rdkit_ligands.append(t[1])
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.lig_cache_path, f"ligands{i}.pkl"), "wb") as f:
                pickle.dump((ligand_graphs), f)
            with open(os.path.join(self.lig_cache_path, f"rdkit_ligands{i}.pkl"), "wb") as f:
                pickle.dump((rdkit_ligands), f)

        ligand_graphs_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.lig_cache_path, f"ligands{i}.pkl"), "rb") as f:
                item = pickle.load(f)  # nosec
                ligand_graphs_all.extend(item)
        with open(os.path.join(self.lig_cache_path, "ligands.pkl"), "wb") as f:
            pickle.dump((ligand_graphs_all), f)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.lig_cache_path, f"rdkit_ligands{i}.pkl"), "rb") as f:
                item = pickle.load(f)  # nosec
                rdkit_ligands_all.extend(item)
        with open(os.path.join(self.lig_cache_path, "rdkit_ligands.pkl"), "wb") as f:
            pickle.dump((rdkit_ligands_all), f)

    def get_ligand(self, name):
        """Returns the ligand graph(s) for a given ligand name."""
        if self.split == "train":
            lig_paths = glob.glob(os.path.join(self.moad_dir, "pdb_superligand", f"{name}_*.pdb"))
        else:
            # NOTE: we refer to the DockGen data directory for all validation and test data
            lig_paths = glob.glob(
                os.path.join(
                    self.dockgen_dir,
                    name.replace(" ", "-"),
                    f"{name.replace(' ', '-')}_ligand*.pdb",
                )
            )

        if not lig_paths:
            log.error(f"No ligands found for {name}")
            return None

        try:
            # NOTE: for `6nco_1_KQP_0` in the DockGen test dataset, we ignore sanitization to avoid an explicit valence error
            is_unsanitizeable_ligand = self.split != "train" and name in [
                "6wjy_2_U41_0",
                "6nco_1_KQP_0",
            ]
            ligands = [
                Chem.MolFromPDBFile(lig_path, sanitize=not is_unsanitizeable_ligand)
                for lig_path in lig_paths
            ]
            ligands_pos_list = [ligand.GetConformer().GetPositions() for ligand in ligands]

            lig = combine_molecules(ligands)
            if not is_unsanitizeable_ligand:
                Chem.SanitizeMol(lig)
        except Exception as e:
            log.error(
                f"Failed to load, combine, or sanitize Binding MOAD ligand(s) for {name} due to: {e}. Skipping preprocessing for this example..."
            )
            return None

        if self.min_multi_lig_distance is not None and len(ligands_pos_list) > 1:
            # calculate pairwise distances between all ligands
            for i in range(len(ligands_pos_list)):
                for j in range(i + 1, len(ligands_pos_list)):
                    ligand_i_pos = ligands_pos_list[i]
                    ligand_j_pos = ligands_pos_list[j]
                    distances = distance.cdist(ligand_i_pos, ligand_j_pos, metric="euclidean")
                    min_dist = np.min(distances)
                    if min_dist < self.min_multi_lig_distance:
                        log.error(
                            f"At a minimum distance of {min_dist}, ligands {i + 1} and {j + 1} were too close to each other for {name}. Skipping preprocessing for this example"
                        )
                        return None

        if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            log.error(
                f"(Multi-)ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Skipping preprocessing for this example..."
            )
            return None

        try:
            lig_samples = [
                process_molecule(
                    ligand,
                    ref_conf_xyz=np.array(ligand.GetConformer().GetPositions()),
                    return_as_dict=True,
                )
                for ligand in ligands
            ]

        except Exception as e:
            log.error(f"Skipping {name} because of the error: {e}")
            return None

        for lig_sample in lig_samples:
            lig_sample["metadata"]["sample_ID"] = name
        return lig_samples, lig
