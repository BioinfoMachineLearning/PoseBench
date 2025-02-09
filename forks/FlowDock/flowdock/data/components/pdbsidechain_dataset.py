# Adapted from: https://github.com/gcorso/DiffDock
# Significant contribution from Ben Fry

import ast
import copy
import os.path
import pickle  # nosec
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
import torch
from beartype.typing import Optional
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowdock.data.components import residue_constants
from flowdock.data.components.constants import (
    aa_long2short,
    aa_to_cg_indices,
    amino_acid_smiles,
    cg_rdkit_indices,
)
from flowdock.data.components.mol_features import process_molecule
from flowdock.data.components.process_mols import generate_conformer
from flowdock.utils import RankedLogger
from flowdock.utils.data_utils import (
    atom37_to_atom14,
    centralize_complex_graph,
    combine_molecules,
    convert_protein_pts_to_pdb,
    get_mol_with_new_conformer_coords,
    merge_protein_and_ligands,
    min_max_normalize_array,
    pdb_filepath_to_protein,
    process_protein,
)
from flowdock.utils.model_utils import sample_inplace_to_torch
from flowdock.utils.utils import fasta_to_dict, read_strings_from_txt

log = RankedLogger(__name__, rank_zero_only=True)


def compute_num_ca_neighbors(
    coords, cg_coords, idx, is_valid_bb_node, max_dist=5, buffer_residue_num=7
):
    """Counts number of residues with heavy atoms within max_dist (Angstroms) of this sidechain
    that are not residues within +/- buffer_residue_num in primary sequence.

    From Ben's code
    Note: Gabriele removed the chain_index
    """

    # Extract coordinates of all residues in the protein.
    bb_coords = coords

    # Compute the indices that we should not consider interactions.
    excluded_neighbors = [
        idx - x for x in reversed(range(0, buffer_residue_num + 1)) if (idx - x) >= 0
    ]
    excluded_neighbors.extend([idx + x for x in range(1, buffer_residue_num + 1)])

    # Create indices of an N x M distance matrix where N is num BB nodes and M is num CG nodes.
    e_idx = torch.stack(
        [
            torch.arange(bb_coords.shape[0])
            .unsqueeze(-1)
            .expand((-1, cg_coords.shape[0]))
            .flatten(),
            torch.arange(cg_coords.shape[0])
            .unsqueeze(0)
            .expand((bb_coords.shape[0], -1))
            .flatten(),
        ]
    )

    # Expand bb_coords and cg_coords into the same dimensionality.
    bb_coords_exp = bb_coords[e_idx[0]]
    cg_coords_exp = cg_coords[e_idx[1]].unsqueeze(1)

    # Every row is distance of chemical group to each atom in backbone coordinate frame.
    bb_exp_idces, _ = (torch.cdist(bb_coords_exp, cg_coords_exp).squeeze(-1) < max_dist).nonzero(
        as_tuple=True
    )
    bb_idces_within_thresh = torch.unique(e_idx[0][bb_exp_idces])

    # Only count residues that are not adjacent or origin in primary sequence and are valid backbone residues (fully resolved coordinate frame).
    bb_idces_within_thresh = bb_idces_within_thresh[
        ~torch.isin(bb_idces_within_thresh, torch.tensor(excluded_neighbors))
        & is_valid_bb_node[bb_idces_within_thresh]
    ]

    return len(bb_idces_within_thresh)


def identify_valid_vandermers(args):
    """Constructs a tensor containing all the number of contacts for each residue that can be
    sampled from for chemical groups.

    By using every sidechain as a chemical group, we will load the actual chemical groups at
    training time. These can be used to sample as probabilities once divided by the sum.
    """
    complex_graph, max_dist, buffer_residue_num = args

    # Constructs a mask tracking whether index is a valid coordinate frame / residue label to train over.
    # is_in_residue_vocabulary = torch.tensor([x in aa_short2long for x in data['seq']]).bool()
    coords, seq = (
        complex_graph["features"]["coords"],
        complex_graph["features"]["seq"],
    )
    is_valid_bb_node = (
        coords[:, :4].isnan().sum(dim=(1, 2)) == 0
    ).bool()  # * is_in_residue_vocabulary

    valid_cg_idces = []
    for idx, aa in enumerate(seq):
        if aa not in aa_to_cg_indices:
            valid_cg_idces.append(0)
        else:
            indices = aa_to_cg_indices[aa]
            cg_coordinates = coords[idx][indices]

            # remove chemical group residues that aren't fully resolved (i.e., contain NaNs or all-zero rows).
            if (
                torch.any(cg_coordinates.isnan()).item()
                or torch.any(cg_coordinates.eq(0).all(dim=1)).item()
            ):
                valid_cg_idces.append(0)
                continue

            nbr_count = compute_num_ca_neighbors(
                coords,
                cg_coordinates,
                idx,
                is_valid_bb_node,
                max_dist=max_dist,
                buffer_residue_num=buffer_residue_num,
            )
            valid_cg_idces.append(nbr_count)

    return complex_graph["metadata"]["sample_ID"], torch.tensor(valid_cg_idces)


def fast_identify_valid_vandermers(coords, seq, max_dist=5, buffer_residue_num=7):
    """Fast version of identify_valid_vandermers that only computes the number of neighbors for
    each residue."""

    offset = 10000 + max_dist
    R = coords.shape[0]

    coords = coords.numpy().reshape(-1, 3)
    pdist_mat = squareform(pdist(coords))
    pdist_mat = pdist_mat.reshape((R, 14, R, 14))
    pdist_mat = np.nan_to_num(pdist_mat, nan=offset)
    pdist_mat = np.min(pdist_mat, axis=(1, 3))

    # compute pairwise distances
    pdist_mat = pdist_mat + np.diag(np.ones(len(seq)) * offset)
    for i in range(1, buffer_residue_num + 1):
        pdist_mat += np.diag(np.ones(len(seq) - i) * offset, k=i) + np.diag(
            np.ones(len(seq) - i) * offset, k=-i
        )

    # get number of residues that are within max_dist of each other
    nbr_count = np.sum(pdist_mat < max_dist, axis=1)
    return torch.tensor(nbr_count)


def subgraph_mol(mol, atoms_to_keep):
    """Subgraphs a molecule based on the atoms to keep."""
    new_mol = Chem.RWMol(mol)
    atoms_to_remove = [
        atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in atoms_to_keep
    ]
    for atom_idx in reversed(sorted(atoms_to_remove)):
        new_mol.RemoveAtom(atom_idx)
    return new_mol.GetMol()


def compute_cg_features(aa, aa_smile):
    """Given an amino acid and a smiles string returns the stacked tensor of chemical group atom
    encodings.

    The order of the output tensor rows corresponds to the index the atoms appear in
    aa_to_cg_indices from constants.
    """

    # Handle any residues that we don't have chemical groups for (ex: GLY if not using bb_cnh and bb_cco)
    aa_short = aa_long2short[aa]
    if aa_short not in aa_to_cg_indices:
        return None

    # Create rdkit molecule from smiles string.
    mol = Chem.MolFromSmiles(aa_smile)

    atoms_to_keep = [i for i, _ in cg_rdkit_indices[aa].items()]
    mol = subgraph_mol(mol, atoms_to_keep)

    return mol


class PDBSidechainDataset(Dataset):
    """A dataset for extracting sidechain "ligands" from PDB structures."""

    def __init__(
        self,
        root,
        transform=None,
        cache_path=os.path.join("data", "cache"),
        pdbsidechain_metadata_dir=os.path.join("data", "pdbsidechain"),
        esm_embeddings_path=None,
        esm_embeddings_sequences_path=None,
        apo_protein_structure_dir=None,
        split="train",
        limit_complexes=0,
        num_workers=0,
        remove_hs=True,
        multiplicity=1,
        vandermers_max_dist=5,
        vandermers_buffer_residue_num=7,
        vandermers_min_contacts=5,
        vandermers_max_surrogate_binding_affinity=15.0,
        vandermers_second_ligand_max_closeness=10.0,
        vandermers_extract_second_ligand=False,
        merge_clusters=1,
        vandermers_extraction=True,
        add_random_ligand=False,
        min_protein_length: Optional[int] = 10,
        max_protein_length: Optional[int] = 4000,
        is_test_dataset=False,
        vandermers_use_prob_as_surrogate_binding_affinity=False,
        a2h_assessment_csv_filepath=None,
        filter_using_a2h_assessment=False,
        a2h_min_tmscore=None,
        a2h_max_rmsd=None,
        a2h_min_protein_length=None,
        a2h_max_protein_length=None,
        postprocess_min_protein_length=None,
        postprocess_max_protein_length=None,
        n_lig_patches=32,
    ):
        """Initializes the dataset."""

        super().__init__(root, transform)
        assert remove_hs is True, "The argument `remove_hs` must be `True` for now"
        self.root = root
        self.split = split
        self.pdbsidechain_metadata_dir = pdbsidechain_metadata_dir
        self.esm_embeddings_path = esm_embeddings_path + (
            f"_limit{limit_complexes}" if limit_complexes == 4 else ""
        )
        self.esm_embeddings_sequences_path = esm_embeddings_sequences_path
        self.apo_protein_structure_dir = apo_protein_structure_dir + (
            f"_limit{limit_complexes}" if limit_complexes == 4 else ""
        )
        self.limit_complexes = limit_complexes
        self.multiplicity = multiplicity
        self.num_workers = num_workers
        self.vandermers_second_ligand_max_closeness = vandermers_second_ligand_max_closeness
        self.vandermers_extract_second_ligand = vandermers_extract_second_ligand
        self.merge_clusters = merge_clusters
        self.vandermers_extraction = vandermers_extraction
        self.add_random_ligand = add_random_ligand
        self.min_protein_length = min_protein_length
        self.max_protein_length = max_protein_length
        self.is_test_dataset = is_test_dataset
        self.postprocess_min_protein_length = postprocess_min_protein_length
        self.postprocess_max_protein_length = postprocess_max_protein_length
        self.vandermers_use_prob_as_surrogate_binding_affinity = (
            vandermers_use_prob_as_surrogate_binding_affinity
        )
        self.n_lig_patches = n_lig_patches

        if postprocess_min_protein_length is not None:
            assert postprocess_min_protein_length >= 10, "`min_protein_length` must be >= 10"
            log.info(
                f"Postprocess-filtering out proteins with length < {postprocess_min_protein_length}"
            )
        if postprocess_max_protein_length is not None:
            assert postprocess_max_protein_length <= 4000, "`max_protein_length` must be <= 4000"
            log.info(
                f"Postprocess-filtering out proteins with length > {postprocess_max_protein_length}"
            )

        if vandermers_extraction:
            self.cg_node_feature_lookup_dict = {
                aa_long2short[aa]: compute_cg_features(aa, aa_smile)
                for aa, aa_smile in amino_acid_smiles.items()
            }

        self.cache_path = os.path.join(
            cache_path,
            f"PDB_limit{self.limit_complexes}_INDEX{self.split}"
            + ("" if min_protein_length is None else f"_minProteinLength{min_protein_length}")
            + ("" if max_protein_length is None else f"_maxProteinLength{max_protein_length}"),
        )
        self.read_split()

        if not self.check_all_proteins():
            os.makedirs(self.cache_path, exist_ok=True)
            self.preprocess()

        self.vandermers_max_dist = vandermers_max_dist
        self.vandermers_buffer_residue_num = vandermers_buffer_residue_num
        self.vandermers_min_contacts = vandermers_min_contacts
        self.vandermers_max_surrogate_binding_affinity = vandermers_max_surrogate_binding_affinity
        self.collect_proteins()

        filtered_proteins = []
        if vandermers_extraction:
            for complex_graph in tqdm(self.protein_graphs):
                if complex_graph["metadata"]["sample_ID"] in self.vandermers and torch.any(
                    self.vandermers[complex_graph["metadata"]["sample_ID"]] >= 10
                ):
                    filtered_proteins.append(complex_graph)
            log.info(
                f"Computed vandermers and kept {len(filtered_proteins)} proteins out of {len(self.protein_graphs)}"
            )
        else:
            filtered_proteins = self.protein_graphs

        second_filter = []
        for complex_graph in tqdm(filtered_proteins):
            second_filter.append(complex_graph)
        log.info(
            f"Checked embeddings available and kept {len(second_filter)} proteins out of {len(filtered_proteins)}"
        )

        self.protein_graphs = second_filter

        # filter clusters that have no protein graphs
        self.split_clusters = list({g["features"]["cluster"] for g in self.protein_graphs})
        self.cluster_to_complexes = {c: [] for c in self.split_clusters}
        for p in self.protein_graphs:
            self.cluster_to_complexes[p["features"]["cluster"]].append(p)
        self.split_clusters = [
            c for c in self.split_clusters if len(self.cluster_to_complexes[c]) > 0
        ]
        log.info(
            f"Total elements in set: {len(self.split_clusters) * self.multiplicity // self.merge_clusters}",
        )

        # analyze and potentially filter the van der Mers dataset based on its apo-to-holo (a2h) structural assessment
        if a2h_assessment_csv_filepath is not None and os.path.exists(a2h_assessment_csv_filepath):
            a2h_assessment_df = pd.read_csv(a2h_assessment_csv_filepath)
            a2h_assessment_df["ID"] = [
                "_".join(Path(paths[0]).stem.split("_")[:2])
                for paths in a2h_assessment_df["Filepath"].apply(ast.literal_eval).tolist()
            ]
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
            # plt.xlim(0, 1000)
            # sns.histplot(a2h_assessment_df["Apo_Length"])
            # plt.title("Apo Protein Length")
            # plt.savefig(plot_dir / "apo_length_hist.png")
            if filter_using_a2h_assessment and not is_test_dataset:
                log.info(
                    f"Filtering the van der Mers {self.split} dataset based on its apo-to-holo (a2h) structural assessment"
                )
                a2h_assessment_df = a2h_assessment_df[
                    (a2h_assessment_df["TM-score"] >= a2h_min_tmscore)
                    & (a2h_assessment_df["RMSD"] <= a2h_max_rmsd)
                    & (a2h_assessment_df["Apo_Length"] >= a2h_min_protein_length)
                    & (a2h_assessment_df["Apo_Length"] <= a2h_max_protein_length)
                ]
                a2h_filtered_ids = {id: None for id in a2h_assessment_df["ID"].tolist()}
                third_filter = [
                    p
                    for p in self.protein_graphs
                    if p["metadata"]["sample_ID"] in a2h_filtered_ids
                ]
                self.protein_graphs = third_filter

                # filter clusters that have no protein graphs
                self.split_clusters = list({g["features"]["cluster"] for g in self.protein_graphs})
                self.cluster_to_complexes = {c: [] for c in self.split_clusters}
                for p in self.protein_graphs:
                    self.cluster_to_complexes[p["features"]["cluster"]].append(p)
                self.split_clusters = [
                    c for c in self.split_clusters if len(self.cluster_to_complexes[c]) > 0
                ]
                log.info(
                    f"Total elements in set after `a2h` filtering: {len(self.split_clusters) * self.multiplicity // self.merge_clusters}",
                )

        self.name_to_complex = {p["metadata"]["sample_ID"]: p for p in self.protein_graphs}
        log.info(
            f"{len(self.name_to_complex)} total protein chains available from {len(self.split_clusters) * self.multiplicity // self.merge_clusters} clusters after all {self.split} filtering"
        )
        with open(
            os.path.join(
                self.cache_path,
                f"pdbsidechain_{self.split}_names.txt",
            ),
            "w",
        ) as f:
            f.write("\n".join(list(self.name_to_complex.keys())))

        self.define_probabilities()

        if self.add_random_ligand:
            raise NotImplementedError("add_random_ligand is not implemented yet")
            assert os.path.exists(
                os.path.join(self.pdbsidechain_metadata_dir, "smiles_list.csv")
            ), "smiles_list.csv not found"
            # read csv with all smiles
            with open(os.path.join(self.pdbsidechain_metadata_dir, "smiles_list.csv")) as f:
                self.smiles_list = f.readlines()
            self.smiles_list = [s.split(",")[0] for s in self.smiles_list]

    def define_probabilities(self):
        """Defines the probabilities for each residue to be selected as a sidechain."""
        if not self.vandermers_extraction:
            return

        if self.vandermers_min_contacts is not None:
            self.probabilities = torch.arange(1000) - self.vandermers_min_contacts + 1
            self.probabilities[: self.vandermers_min_contacts] = 0
        else:
            assert os.path.exists(
                os.path.join(self.pdbsidechain_metadata_dir, "pdbbind_counts.pkl")
            ), "pdbbind_counts.pkl not found"
            with open(
                os.path.join(self.pdbsidechain_metadata_dir, "pdbbind_counts.pkl"), "rb"
            ) as f:
                pdbbind_counts = pickle.load(f)  # nosec

            pdb_counts = torch.ones(1000)
            for contacts in self.vandermers.values():
                pdb_counts.index_add_(0, contacts, torch.ones(contacts.shape))
            log.info(f"pdbbind_counts[:30]: {pdbbind_counts[:30]}")
            log.info(f"pdb_counts[:30]: {pdb_counts[:30]}")

            self.probabilities = pdbbind_counts / pdb_counts
            self.probabilities[:7] = 0

    def len(self):
        """Returns the number of proteins in the dataset."""
        return len(self.split_clusters) * self.multiplicity // self.merge_clusters

    def get(self, idx=None, protein=None, smiles=None):
        """Returns a protein graph from the dataset."""
        assert idx is not None or (
            protein is not None and smiles is not None
        ), "provide idx or protein or smile"

        if protein is None or smiles is None:
            idx = idx % len(self.split_clusters)
            if self.merge_clusters > 1:
                idx = idx * self.merge_clusters
                idx = idx + random.randint(0, self.merge_clusters - 1)  # nosec
                idx = min(idx, len(self.split_clusters) - 1)
            cluster = self.split_clusters[idx]
            protein_samples = copy.deepcopy(
                random.choice(self.cluster_to_complexes[cluster])  # nosec
            )
        else:
            protein_samples = copy.deepcopy(self.name_to_complex[protein])

        if self.vandermers_extraction:
            # select sidechain to remove
            vandermers_contacts = self.vandermers[protein_samples["metadata"]["sample_ID"]]
            vandermers_probs = self.probabilities[vandermers_contacts].numpy()
            original_vandermers_probs = vandermers_probs.copy()

            if not np.any(vandermers_contacts.numpy() >= 10):
                new_idx = random.randint(0, self.len())  # nosec
                log.warning(
                    f"No vandermers with >= 10 contacts found. Retrying with new example from cluster with index {new_idx}."
                )
                return self.get(new_idx)

            sidechain_idx = np.random.choice(
                np.arange(len(vandermers_probs)), p=vandermers_probs / np.sum(vandermers_probs)
            )

            # remove part of the sequence
            residues_to_keep = np.ones(len(protein_samples["features"]["seq"]), dtype=bool)
            residues_to_keep[
                max(0, sidechain_idx - self.vandermers_buffer_residue_num) : min(
                    sidechain_idx + self.vandermers_buffer_residue_num + 1,
                    len(protein_samples["features"]["seq"]),
                )
            ] = False

            if self.vandermers_extract_second_ligand:
                pos_idx = protein_samples["features"]["coords"][sidechain_idx, 1, :]
                far_enough = (
                    torch.sum(
                        (protein_samples["features"]["coords"][..., 1, :] - pos_idx[None, :]) ** 2,
                        dim=-1,
                    )
                    > self.vandermers_second_ligand_max_closeness**2
                )
                vandermers_probs = vandermers_probs * far_enough.float().numpy()
                vandermers_probs[
                    max(0, sidechain_idx - self.vandermers_buffer_residue_num) : min(
                        sidechain_idx + self.vandermers_buffer_residue_num + 1,
                        len(protein_samples["features"]["seq"]),
                    )
                ] = 0
                if np.all(vandermers_probs <= 0):
                    new_idx = random.randint(0, self.len())  # nosec
                    log.warning(
                        f"No second vandermer available. Retrying with new example from cluster with index {new_idx}."
                    )
                    return self.get(new_idx)
                sc2_idx = np.random.choice(
                    np.arange(len(vandermers_probs)),
                    p=vandermers_probs / np.sum(vandermers_probs),
                )

                residues_to_keep[
                    max(0, sc2_idx - self.vandermers_buffer_residue_num) : min(
                        sc2_idx + self.vandermers_buffer_residue_num + 1,
                        len(protein_samples["features"]["seq"]),
                    )
                ] = False

            # create the sidechain ligand
            sidechain_aa = protein_samples["features"]["seq"][sidechain_idx]
            lig_list = [self.cg_node_feature_lookup_dict[sidechain_aa]]
            ligand_pos_list = [
                protein_samples["features"]["coords"][sidechain_idx][
                    protein_samples["features"]["mask"][sidechain_idx]
                ]
            ]
            sidechain_indices = [sidechain_idx]
            if self.vandermers_extract_second_ligand:
                sidechain_aa2 = protein_samples["features"]["seq"][sc2_idx]
                lig_list.append(self.cg_node_feature_lookup_dict[sidechain_aa2])
                ligand_pos_list.append(
                    protein_samples["features"]["coords"][sc2_idx][
                        protein_samples["features"]["mask"][sc2_idx]
                    ]
                )
                sidechain_indices.append(sc2_idx)
            try:
                lig_samples = []
                for lig, ligand_pos in zip(lig_list, ligand_pos_list):
                    lig_mol = get_mol_with_new_conformer_coords(
                        lig,
                        ligand_pos.numpy() if torch.is_tensor(ligand_pos) else ligand_pos,
                    )
                    lig_samples.append(
                        process_molecule(
                            lig_mol,
                            ref_conf_xyz=np.array(lig_mol.GetConformer().GetPositions()),
                            return_as_dict=True,
                        )
                    )
            except Exception as e:
                new_idx = random.randint(0, self.len())  # nosec
                log.warning(
                    f"Vandermer could not be constructed for the cluster with index {idx} due to: {e}. Retrying  with new example from cluster with index {new_idx}."
                )
                return self.get(new_idx)

            # subset the protein features to keep only the residues that are not part of the selected sidechain
            residues_to_keep = torch.from_numpy(residues_to_keep)
            num_residues = residues_to_keep.shape[0]
            for key in protein_samples:
                for subkey in protein_samples[key].keys():
                    if (
                        isinstance(protein_samples[key][subkey], int)
                        and protein_samples[key][subkey] == num_residues
                    ):
                        protein_samples[key][subkey] = residues_to_keep.sum().item()
                    elif (
                        isinstance(protein_samples[key][subkey], np.ndarray)
                        and protein_samples[key][subkey].shape[0] == num_residues
                    ):
                        protein_samples[key][subkey] = protein_samples[key][subkey][
                            residues_to_keep
                        ]
                    elif (
                        isinstance(protein_samples[key][subkey], list)
                        and len(protein_samples[key][subkey]) > 0
                        and isinstance(protein_samples[key][subkey][0], tuple)
                        and len(protein_samples[key][subkey][0]) == 2
                        and isinstance(protein_samples[key][subkey][0][0], str)
                        and isinstance(protein_samples[key][subkey][0][1], str)
                    ):
                        new_tuples_list = []
                        for tuples_list in protein_samples[key][subkey]:
                            if len(tuples_list[1]) == num_residues:
                                new_tuple = (
                                    tuples_list[0],
                                    "".join(np.array(list(tuples_list[1]))[residues_to_keep]),
                                )
                            else:
                                new_tuple = tuples_list
                            new_tuples_list.append(new_tuple)
                        protein_samples[key][subkey] = new_tuples_list

            for key in [
                "coords",
                "seq",
                "mask",
                "cluster",
                "orig_seq",
                "to_keep",
            ]:
                # clean up features prior to batching
                if key in protein_samples["features"]:
                    del protein_samples["features"][key]
            complex_graph = merge_protein_and_ligands(
                lig_samples,
                protein_samples,
                n_lig_patches=self.n_lig_patches,
            )
        else:
            lig_list = [None]
            complex_graph = protein_samples

        if self.add_random_ligand:
            raise NotImplementedError("add_random_ligand is not implemented yet")
            if smiles is not None:
                mol = MolFromSmiles(smiles)
                try:
                    generate_conformer(mol)
                except Exception as e:
                    log.error(f"Failed to generate the given ligand, returning `None` due to: {e}")
                    return None
            else:
                success = False
                while not success:
                    smiles = random.choice(self.smiles_list)  # nosec
                    mol = MolFromSmiles(smiles)
                    try:
                        success = not generate_conformer(mol)
                    except Exception as e:
                        log.error(f"Changing ligand due to: {e}")

            lig = get_mol_with_new_conformer_coords(
                mol,
                mol.GetConformer().GetPositions()
                - np.mean(mol.GetConformer().GetPositions(), axis=0, keepdims=True),
            )
            lig_samples = [
                process_molecule(
                    lig,
                    ref_conf_xyz=np.array(lig.GetConformer().GetPositions()),
                    return_as_dict=True,
                )
            ]
            for key in [
                "coords",
                "seq",
                "mask",
                "cluster",
                "orig_seq",
                "to_keep",
            ]:
                # clean up features prior to batching
                if key in protein_samples["features"]:
                    del protein_samples["features"][key]
            complex_graph = merge_protein_and_ligands(
                lig_samples,
                protein_samples,
                n_lig_patches=self.n_lig_patches,
            )

        complex_graph["metadata"]["mol"] = combine_molecules(lig_list) if all(lig_list) else None
        complex_graph = sample_inplace_to_torch(complex_graph)

        affinity = [torch.nan for _ in range(len(lig_list))]
        if self.vandermers_extraction and self.vandermers_use_prob_as_surrogate_binding_affinity:
            normalized_vandermers_probs = min_max_normalize_array(
                original_vandermers_probs / np.sum(original_vandermers_probs)
            )
            affinity = [
                normalized_vandermers_probs[sidechain_index]
                * self.vandermers_max_surrogate_binding_affinity
                for sidechain_index in sidechain_indices
            ]
        complex_graph["features"]["affinity"] = torch.tensor(affinity, dtype=torch.float32)

        return centralize_complex_graph(complex_graph)

    def read_split(self):
        """Reads the split from the CSV file."""
        # read CSV file
        df = pd.read_csv(os.path.join(self.root, "list.csv"))
        log.info("Loaded list CSV file")

        # get clusters and filter by split
        if self.split == "train":
            val_clusters = set(
                read_strings_from_txt(os.path.join(self.root, "valid_clusters.txt"))
            )
            test_clusters = set(
                read_strings_from_txt(os.path.join(self.root, "test_clusters.txt"))
            )
            clusters = df["CLUSTER"].unique()
            clusters = [
                int(c) for c in clusters if c not in val_clusters and c not in test_clusters
            ]
        elif self.split == "val":
            clusters = [
                int(s)
                for s in read_strings_from_txt(os.path.join(self.root, "valid_clusters.txt"))
            ]
        elif self.split == "test":
            clusters = [
                int(s) for s in read_strings_from_txt(os.path.join(self.root, "test_clusters.txt"))
            ]
        else:
            raise ValueError("Split must be train, val or test")
        log.info(f"{self.split} clusters: {len(clusters)}")
        clusters = set(clusters)

        self.chains_in_cluster = []
        complexes_in_cluster = set()
        for chain, cluster in zip(df["CHAINID"], df["CLUSTER"]):
            if cluster not in clusters:
                continue
            # limit to one chain per complex
            if chain[:4] not in complexes_in_cluster:
                self.chains_in_cluster.append((chain, cluster))
                complexes_in_cluster.add(chain[:4])
        log.info(f"Filtered chains in cluster: {len(self.chains_in_cluster)}")

        if self.limit_complexes > 0:
            self.chains_in_cluster = self.chains_in_cluster[: self.limit_complexes]

    def check_all_proteins(self):
        """Checks if all proteins have been preprocessed."""
        for i in range(len(self.chains_in_cluster) // 10000 + 1):
            if not os.path.exists(os.path.join(self.cache_path, f"protein_graphs{i}.pkl")):
                return False
        return True

    def collect_proteins(self):
        """Collects all proteins from the cache."""
        self.protein_graphs = []
        self.vandermers = {}
        total_recovered = 0
        log.info(f"Loading {len(self.chains_in_cluster)} protein graphs.")
        list_indices = list(range(len(self.chains_in_cluster) // 10000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            with open(os.path.join(self.cache_path, f"protein_graphs{i}.pkl"), "rb") as f:
                log.info(f"Loading protein graphs {i}")
                item = pickle.load(f)  # nosec
                if (
                    self.postprocess_min_protein_length is not None
                    and self.postprocess_max_protein_length is not None
                ):
                    item = [
                        sample
                        for sample in item
                        if self.postprocess_min_protein_length
                        <= len(sample["features"]["res_type"])
                        <= self.postprocess_max_protein_length
                    ]
                elif self.postprocess_min_protein_length is not None:
                    item = [
                        sample
                        for sample in item
                        if len(sample["features"]["res_type"])
                        >= self.postprocess_min_protein_length
                    ]
                elif self.postprocess_max_protein_length is not None:
                    item = [
                        sample
                        for sample in item
                        if len(sample["features"]["res_type"])
                        <= self.postprocess_max_protein_length
                    ]
                total_recovered += len(item)
                self.protein_graphs.extend(item)

            if not self.vandermers_extraction:
                continue

            if os.path.exists(
                os.path.join(
                    self.cache_path,
                    f"vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl",
                )
            ):
                with open(
                    os.path.join(
                        self.cache_path,
                        f"vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl",
                    ),
                    "rb",
                ) as f:
                    vandermers = pickle.load(f)  # nosec
                    self.vandermers.update(vandermers)
                continue

            vandermers = {}
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(item), desc=f"Computing vandermers {i}") as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                arguments = zip(
                    item,
                    [self.vandermers_max_dist] * len(item),
                    [self.vandermers_buffer_residue_num] * len(item),
                )
                for t in map_fn(identify_valid_vandermers, arguments):
                    if t is not None:
                        vandermers[t[0]] = t[1]
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(
                os.path.join(
                    self.cache_path,
                    f"vandermers{i}_{self.vandermers_max_dist}_{self.vandermers_buffer_residue_num}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(vandermers, f)
            self.vandermers.update(vandermers)

        log.info(
            f"Kept {len(self.protein_graphs)} proteins out of {len(self.chains_in_cluster)} total"
        )

    def preprocess(self):
        """Preprocesses all proteins in the dataset."""
        if self.esm_embeddings_path is not None:
            log.info("Loading ESM embeddings")
            sequences_to_embeddings = {}
            receptor_names_all_dict = {chain[0]: None for chain in self.chains_in_cluster}
            sequences_dict = fasta_to_dict(self.esm_embeddings_sequences_path)
            for embedding_filepath in os.listdir(self.esm_embeddings_path):
                key = Path(embedding_filepath).stem
                chain_name = key.split("_chain_")[0]
                chain_id = key.split("_chain_")[1]
                full_chain_id = f"{chain_name}_{chain_id}"
                if full_chain_id in receptor_names_all_dict:
                    embedding = torch.load(
                        os.path.join(self.esm_embeddings_path, embedding_filepath)
                    )["representations"][33]
                    seq = sequences_dict[key]
                    sequences_to_embeddings[seq + f":{chain_id}"] = embedding
        else:
            sequences_to_embeddings = None

        # running preprocessing in parallel on multiple workers and saving the progress every 10000 proteins
        list_indices = list(range(len(self.chains_in_cluster) // 10000 + 1))
        random.shuffle(list_indices)
        for i in list_indices:
            if os.path.exists(os.path.join(self.cache_path, f"protein_graphs{i}.pkl")):
                continue
            chains_names = self.chains_in_cluster[10000 * i : 10000 * (i + 1)]
            protein_graphs = []
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(
                total=len(chains_names),
                desc=f"Loading protein batch {i}/{len(self.chains_in_cluster) // 10000 + 1}",
            ) as pbar:
                map_fn = p.imap_unordered if self.num_workers > 1 else map
                for t in map_fn(
                    self.load_chain,
                    zip(chains_names, [sequences_to_embeddings] * len(chains_names)),
                ):
                    if t is not None:
                        protein_graphs.append(t)
                    pbar.update()
            if self.num_workers > 1:
                p.__exit__(None, None, None)

            with open(os.path.join(self.cache_path, f"protein_graphs{i}.pkl"), "wb") as f:
                pickle.dump(protein_graphs, f)

        log.info("Finished preprocessing and saving protein graphs")

    def load_chain(self, c):
        """Loads a protein graph from a chain name."""
        chain, cluster = c[0]
        sequences_to_embeddings = c[1]
        chain_id = chain.split("_")[1]
        pt_filepath = os.path.join(self.root, "pdb", chain[1:3], f"{chain}.pt")
        holo_protein_filepath = os.path.join(self.root, "pdb", chain[1:3], f"{chain}.pdb")
        if not os.path.exists(pt_filepath):
            log.error(f"PyTorch file not found: {chain}")
            return None
        if not os.path.exists(holo_protein_filepath):
            try:
                convert_protein_pts_to_pdb([pt_filepath], holo_protein_filepath)
                if not os.path.exists(holo_protein_filepath):
                    log.error(
                        f"Failed to convert PyTorch {pt_filepath} to PDB {holo_protein_filepath}"
                    )
                    return None

            except Exception as e:
                log.error(
                    f"Error in converting {pt_filepath} to {holo_protein_filepath} due to: {e}"
                )
                return None

        try:
            holo_af_protein = pdb_filepath_to_protein(holo_protein_filepath)
            holo_protein_sample = process_protein(
                holo_af_protein,
                sample_name=f"{chain.split('_')[0]}_",
            )
            complex_graph = holo_protein_sample

        except Exception as e:
            log.error(f"Error in extracting holo receptor {chain}: {e}")
            return None

        if np.isnan(complex_graph["features"]["res_atom_positions"]).any():
            log.error(
                f"NaN in holo receptor pos for {chain}. Skipping preprocessing for this example..."
            )
            return None

        try:
            if self.apo_protein_structure_dir is not None:
                apo_protein_filepath = os.path.join(
                    self.apo_protein_structure_dir, f"{chain}_holo_aligned_esmfold_protein.pdb"
                )
                apo_af_protein = pdb_filepath_to_protein(apo_protein_filepath)
                apo_protein_sample = process_protein(
                    apo_af_protein,
                    sample_name=f"{chain.split('_')[0]}_",
                    sequences_to_embeddings=sequences_to_embeddings,
                    chain_id=chain_id,
                )
                for key in complex_graph.keys():
                    for subkey, value in apo_protein_sample[key].items():
                        complex_graph[key]["apo_" + subkey] = value
                if not np.array_equal(
                    complex_graph["features"]["res_type"],
                    complex_graph["features"]["apo_res_type"],
                ):
                    log.error(
                        f"Residue type mismatch between holo protein and apo protein for {chain}. Skipping preprocessing for this example..."
                    )
                    return None
                if np.isnan(complex_graph["features"]["apo_res_atom_positions"]).any():
                    log.error(
                        f"NaN in apo receptor pos for {chain}. Skipping preprocessing for this example..."
                    )
                    return None

        except Exception as e:
            log.error(f"Skipping apo {chain} because of the error: {e}")
            return None

        if (
            self.min_protein_length is not None
            and complex_graph["metadata"]["num_a"] < self.min_protein_length
            and not self.is_test_dataset
        ):
            log.info(
                f"Skipping {chain} because of its length {complex_graph['metadata']['num_a']}"
            )
            return None
        if (
            self.max_protein_length is not None
            and complex_graph["metadata"]["num_a"] > self.max_protein_length
            and not self.is_test_dataset
        ):
            log.info(
                f"Skipping {chain} because of its length {complex_graph['metadata']['num_a']}"
            )
            return None

        # NOTE: can manually confirm that OpenFold's `atom14` representation perfectly matches RoseTTAFold2's `atom14` representation
        # Also NOTE: RoseTTAFold2's `atom27` representation linked below is actually its `atom14` representation plus 13 hydrogen atoms per residue (starting at index 14)
        # OpenFold reference code: https://github.com/aqlaboratory/openfold/blob/80c85b54e1a81d9a66df3f1b6c257ff97f10acd3/openfold/np/residue_constants.py#L601
        # RoseTTAFold2 reference code: https://github.com/uw-ipd/RoseTTAFold2/blob/a26f29bb92d4946ef8bf225edba9240c9e647747/network/chemical.py#L17
        atom14_coords, atom14_mask = atom37_to_atom14(
            torch.from_numpy(complex_graph["features"]["res_type"]),
            torch.from_numpy(complex_graph["features"]["res_atom_positions"]),
            torch.from_numpy(complex_graph["features"]["res_atom_mask"]),
        )
        complex_seq = "".join(
            [
                residue_constants.restypes[res_idx]
                for res_idx in complex_graph["features"]["res_type"]
            ]
        )

        complex_graph["features"]["coords"] = atom14_coords
        complex_graph["features"]["mask"] = atom14_mask.bool()
        complex_graph["features"]["seq"] = complex_seq
        complex_graph["features"]["cluster"] = cluster
        complex_graph["metadata"]["sample_ID"] = chain

        return complex_graph


if __name__ == "__main__":
    dataset = PDBSidechainDataset(
        root=os.path.join("data", "pdbsidechain", "pdb_2021aug02"),
        split="train",
        multiplicity=1,
        limit_complexes=150,
    )
    log.info(f"Length of dataset: {len(dataset)}")
    log.info(f"First dataset example: {dataset[0]}")
    for p in dataset:
        log.info(f"Example: {p['metadata']['sample_ID']}")
        break
