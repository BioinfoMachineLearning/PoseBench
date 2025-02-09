# Adapted from: https://github.com/gcorso/DiffDock

import ast
import binascii
import copy
import glob
import os
import pickle  # nosec
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Optional
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles, RemoveAllHs, RemoveHs
from torch_geometric.data import Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components.mol_features import process_molecule
from flowdock.data.components.process_mols import generate_conformer, read_molecule
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    centralize_complex_graph,
    merge_protein_and_ligands,
    pdb_filepath_to_protein,
    process_protein,
)
from flowdock.utils.model_utils import sample_inplace_to_torch
from flowdock.utils.utils import read_strings_from_txt

log = RankedLogger(__name__, rank_zero_only=True)


class PDBBindDataset(Dataset):
    """A PyTorch Geometric Dataset for PDBBind dataset."""

    def __init__(
        self,
        root,
        transform=None,
        cache_path=os.path.join("data", "cache"),
        split_path="data" + os.sep,
        limit_complexes=0,
        num_workers=0,
        max_lig_size=None,
        remove_hs=False,
        num_conformers=1,
        esm_embeddings_path=None,
        apo_protein_structure_dir=None,
        require_ligand=False,
        include_miscellaneous_atoms=False,
        protein_path_list=None,
        ligand_descriptions=None,
        keep_local_structures=False,
        protein_file="protein_processed",
        ligand_file="ligand",
        min_protein_length: Optional[int] = 10,
        max_protein_length: Optional[int] = 4000,
        is_test_dataset=False,
        a2h_assessment_csv_filepath=None,
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
        """Initializes the dataset."""

        super().__init__(root, transform)
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.apo_protein_structure_dir = apo_protein_structure_dir
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.ligand_file = ligand_file
        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length
        self.is_test_dataset = is_test_dataset
        self.binding_affinity_values_dict = binding_affinity_values_dict
        self.n_lig_patches = n_lig_patches

        split = os.path.splitext(os.path.basename(self.split_path))[0]
        self.full_cache_path = os.path.join(
            cache_path,
            f"PDBBind_limit{self.limit_complexes}"
            f"_INDEX{split}"
            f"_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}"
            + ("" if self.esm_embeddings_path is None else "_esmEmbeddings")
            + "_full"
            + ("" if not keep_local_structures else "_keptLocalStruct")
            + (
                ""
                if protein_path_list is None or ligand_descriptions is None
                else str(binascii.crc32("".join(ligand_descriptions + protein_path_list).encode()))
            )
            + ("" if protein_file == "protein_processed" else "_" + protein_file)
            + ("" if not self.include_miscellaneous_atoms else "_miscAtoms")
            + ("" if self.use_old_wrong_embedding_order else "_chainOrd")
            + ("" if min_protein_length is None else f"_minProteinLength{min_protein_length}")
            + ("" if max_protein_length is None else f"_maxProteinLength{max_protein_length}"),
        )
        self.num_conformers = num_conformers

        if not self.check_all_complexes():
            os.makedirs(self.full_cache_path, exist_ok=True)
            if protein_path_list is None or ligand_descriptions is None:
                self.preprocessing()
            else:
                self.inference_preprocessing()

        self.complex_graphs, self.rdkit_ligands = self.collect_all_complexes()

        # analyze and potentially filter the PDBBind dataset based on its apo-to-holo (a2h) structural assessment
        if a2h_assessment_csv_filepath is not None and os.path.exists(a2h_assessment_csv_filepath):
            a2h_assessment_df = pd.read_csv(a2h_assessment_csv_filepath)
            a2h_assessment_df["ID"] = [
                Path(paths[0]).stem[:4]
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
                    f"Filtering the PDBBind {split} dataset based on its apo-to-holo (a2h) structural assessment"
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
                new_complex_graphs, new_rdkit_ligands = [], []
                for complex_id, complex_obj in enumerate(self.complex_graphs):
                    if complex_obj["metadata"]["sample_ID"].lower() in a2h_filtered_ids:
                        new_complex_graphs.append(complex_obj)
                        new_rdkit_ligands.append(self.rdkit_ligands[complex_id])
                self.complex_graphs = new_complex_graphs
                self.rdkit_ligands = new_rdkit_ligands

        list_names = [complex_obj["metadata"]["sample_ID"] for complex_obj in self.complex_graphs]
        log.info(
            f"{len(list_names)} total complexes available from {self.full_cache_path} after all {split} filtering"
        )
        with open(
            os.path.join(
                self.full_cache_path,
                f"pdbbind_{os.path.splitext(os.path.basename(self.split_path))[0][:3]}_names.txt",
            ),
            "w",
        ) as f:
            f.write("\n".join(list_names))

    def len(self):
        """Returns the number of complexes in the dataset."""
        return len(self.complex_graphs)

    def get(self, idx, default_ligand_ccd_id: str = "XXX"):
        """Returns a HeteroData object for a given index."""
        complex_graph = sample_inplace_to_torch(copy.deepcopy(self.complex_graphs[idx]))
        if self.require_ligand:
            complex_graph["metadata"]["mol"] = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]))
        if self.binding_affinity_values_dict is not None:
            try:
                complex_graph["features"]["affinity"] = torch.tensor(
                    [
                        self.binding_affinity_values_dict[
                            complex_graph["metadata"]["sample_ID"].lower()
                        ][default_ligand_ccd_id]
                    ],
                    dtype=torch.float32,
                )
            except Exception as e:
                log.info(
                    f"Binding affinity value not found for {complex_graph['metadata']['sample_ID']} due to: {e}"
                )
                complex_graph["features"]["affinity"] = torch.tensor(
                    [torch.nan], dtype=torch.float32
                )
        else:
            complex_graph["features"]["affinity"] = torch.tensor([torch.nan], dtype=torch.float32)
        return centralize_complex_graph(complex_graph)

    def preprocessing(self):
        """Preprocesses the complexes for training."""
        log.info(
            f"Processing complexes from [{self.split_path}] and saving them to [{self.full_cache_path}]"
        )

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        log.info(f"Loading {len(complex_names_all)} complexes.")

        if self.esm_embeddings_path is not None:
            log.info("Loading ESM embeddings")
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            for embedding_filepath in os.listdir(self.esm_embeddings_path):
                key = Path(embedding_filepath).stem
                key_name = key.split("_chain_")[0]
                if key_name in complex_names_all:
                    embedding = torch.load(
                        os.path.join(self.esm_embeddings_path, embedding_filepath)
                    )["representations"][33]
                    chain_embeddings_dictlist[key_name].append(embedding)
                    chain_indices_dictlist[key_name].append(int(key.split("_chain_")[1]))
            lm_embeddings_chains_all = []
            for name in complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = {
                    idx: complex_chains_embeddings[i] for idx, i in enumerate(chain_reorder_idx)
                }
                lm_embeddings_chains_all.append(reordered_chains)
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(complex_names_all) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            complex_names = complex_names_all[1000 * i : 1000 * (i + 1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000 * i : 1000 * (i + 1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(complex_names),
                desc=f"Loading complexes {i}/{len(complex_names_all)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.get_complex,
                    zip(
                        complex_names,
                        lm_embeddings_chains,
                        [None] * len(complex_names),
                        [None] * len(complex_names),
                    ),
                ):
                    if t is not None:
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "wb") as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "wb") as f:
                pickle.dump((rdkit_ligands), f)

    def inference_preprocessing(self):
        """Preprocesses the complexes for inference."""
        ligands_list = []
        log.info("Reading molecules and generating local structures with RDKit")
        for ligand_description in tqdm(self.ligand_descriptions):
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
                ligands_list.append(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = AddHs(mol)
                    generate_conformer(mol)
                ligands_list.append(mol)

        if self.esm_embeddings_path is not None:
            log.info("Reading language model embeddings.")
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path):
                raise Exception("ESM embeddings path does not exist: ", self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(
                    glob.glob(
                        os.path.join(self.esm_embeddings_path, os.path.basename(protein_path))
                        + "*"
                    )
                )
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)["representations"][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)

        log.info("Generating graphs for ligands and proteins")
        # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
        list_indices = list(range(len(self.protein_path_list) // 1000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                continue
            protein_paths_chunk = self.protein_path_list[1000 * i : 1000 * (i + 1)]
            ligand_description_chunk = self.ligand_descriptions[1000 * i : 1000 * (i + 1)]
            ligands_chunk = ligands_list[1000 * i : 1000 * (i + 1)]
            lm_embeddings_chains = lm_embeddings_chains_all[1000 * i : 1000 * (i + 1)]
            complex_graphs, rdkit_ligands = [], []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(protein_paths_chunk),
                desc=f"Loading complexes {i}/{len(protein_paths_chunk)//1000+1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.get_complex,
                    zip(
                        protein_paths_chunk,
                        lm_embeddings_chains,
                        ligands_chunk,
                        ligand_description_chunk,
                    ),
                ):
                    if t is not None:
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "wb") as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "wb") as f:
                pickle.dump((rdkit_ligands), f)

    def check_all_complexes(self):
        """Checks if all complexes are already in the cache."""
        if os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")):
            return True

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        for i in range(len(complex_names_all) // 1000 + 1):
            if not os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                return False
        return True

    def collect_all_complexes(self):
        """Collects all complexes from the cache."""
        log.info("Collecting all complexes from cache", self.full_cache_path)
        if os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl")):
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), "rb") as f:
                complex_graphs = pickle.load(f)  # nosec
            if self.require_ligand:
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), "rb") as f:
                    rdkit_ligands = pickle.load(f)  # nosec
            else:
                rdkit_ligands = None
            return complex_graphs, rdkit_ligands

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[: self.limit_complexes]
        complex_graphs_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), "rb") as f:
                log.info(f"Loading heterographs{i}.pkl")
                item = pickle.load(f)  # nosec
                complex_graphs_all.extend(item)

        rdkit_ligands_all = []
        for i in range(len(complex_names_all) // 1000 + 1):
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), "rb") as f:
                item = pickle.load(f)  # nosec
                rdkit_ligands_all.extend(item)

        return complex_graphs_all, rdkit_ligands_all

    def get_complex(self, par):
        """Returns a list of HeteroData objects and a list of RDKit molecules for a given
        complex."""
        name, lm_embedding_chains, ligand, _ = par
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            log.error(f"Data directory not found for {name}")
            return [], []

        try:
            lig = read_mol(self.pdbbind_dir, name, suffix=self.ligand_file, remove_hs=False)
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                log.error(
                    f"Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Skipping preprocessing for this example..."
                )
                return [], []

            if self.remove_hs:
                lig = RemoveHs(lig)

            lig_samples = [
                process_molecule(
                    lig,
                    ref_conf_xyz=np.array(lig.GetConformer().GetPositions()),
                    return_as_dict=True,
                )
            ]

        except Exception as e:
            log.error(f"Skipping {name} because of error: {e}")
            return [], []

        try:
            holo_protein_filepath = os.path.join(
                self.pdbbind_dir, name, f"{name}_{self.protein_file}.pdb"
            )
            holo_af_protein = pdb_filepath_to_protein(holo_protein_filepath)
            holo_protein_sample = process_protein(
                holo_af_protein,
                sample_name=f"{name}_",
            )
            complex_graph = merge_protein_and_ligands(
                lig_samples,
                holo_protein_sample,
                n_lig_patches=self.n_lig_patches,
            )

        except Exception as e:
            log.error(f"Skipping holo {name} because of error: {e}")
            return [], []

        if np.isnan(complex_graph["features"]["res_atom_positions"]).any():
            log.error(
                f"NaN in holo receptor pos for {name}. Skipping preprocessing for this example..."
            )
            return None

        try:
            if self.apo_protein_structure_dir is not None:
                apo_protein_filepath = os.path.join(
                    self.apo_protein_structure_dir, f"{name}_holo_aligned_esmfold_protein.pdb"
                )
                apo_af_protein = pdb_filepath_to_protein(apo_protein_filepath)
                apo_protein_sample = process_protein(
                    apo_af_protein,
                    sample_name=f"{name}_",
                    sequences_to_embeddings=lm_embedding_chains,
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
            log.error(f"Skipping apo {name} because of error: {e}")
            return [], []

        if (
            self.min_protein_length is not None
            and complex_graph["metadata"]["num_a"] < self.min_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return [], []
        if (
            self.max_protein_length is not None
            and complex_graph["metadata"]["num_a"] > self.max_protein_length
            and not self.is_test_dataset
        ):
            log.info(f"Skipping {name} because of its length {complex_graph['metadata']['num_a']}")
            return [], []

        complex_graph["metadata"]["sample_ID"] = name
        return [complex_graph], [lig]


def read_mol(pdbbind_dir, name, suffix="ligand", remove_hs=False):
    """Reads a ligand from the given directory and returns it as an RDKit molecule."""
    lig = read_molecule(
        os.path.join(pdbbind_dir, name, f"{name}_{suffix}.mol2"),
        remove_hs=remove_hs,
        sanitize=True,
    )
    if lig is None:  # read sdf file if mol2 file cannot be sanitized
        log.info(
            "Reading the .mol2 file failed. We found a .sdf file instead and are trying to use that. Be aware that the .sdf files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
        )
        lig = read_molecule(
            os.path.join(pdbbind_dir, name, f"{name}_{suffix}.sdf"),
            remove_hs=remove_hs,
            sanitize=True,
        )
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(lig)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    """Reads all ligands from the given directory and returns them as a list of RDKit molecules."""
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".mol2") and "rdkit" not in file:
            lig = read_molecule(
                os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True
            )
            if lig is None and os.path.exists(
                os.path.join(pdbbind_dir, name, file[:-4] + ".sdf")
            ):  # read sdf file if mol2 file cannot be sanitized
                log.info(
                    "Using the .mol2 file failed. We found a .sdf file instead and are trying to use that. Be aware that the .sdf files from PDBBind 2020 are missing chirality tags, although we will do our best to impute such information automatically using RDKit. Reference: https://www.blopig.com/blog/2021/09/watch-out-when-using-pdbbind."
                )
                lig = read_molecule(
                    os.path.join(pdbbind_dir, name, file[:-4] + ".sdf"),
                    remove_hs=remove_hs,
                    sanitize=True,
                )
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(lig)
            if lig is not None:
                ligs.append(lig)
    return ligs
