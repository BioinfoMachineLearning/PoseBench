# Adapted from: https://github.com/zrqiao/NeuralPLexer

import msgpack
import msgpack_numpy as m
import networkx
import networkx as nx
import numpy as np
import rootutils
import torch
from beartype.typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Periodic table metadata
# Runtime accessing from mendeleev is surprisingly slow, tabulate here
from flowdock.data.components.physical import PGROUP_IDS, PPERIOD_IDS
from flowdock.utils import RankedLogger

BONDORDER = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.AROMATIC: 1.5,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
}

m.patch()
log = RankedLogger(__name__, rank_zero_only=True)


def one_of_k_encoding(x, allowable_set) -> List[bool]:
    """
    Maps inputs not in the allowable set to the last element.
    Modified from https://github.com/XuhanLiu/NGFP.

    :param x: input
    :param allowable_set: allowable set
    :return: one-hot encoding
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_encoding(atom: Chem.rdchem.Atom) -> np.ndarray:
    """Get atom encoding.

    :param atom: Chem.rdchem.Atom atom object
    :return: encoding_list
    """
    encoding_list = one_of_k_encoding(
        PGROUP_IDS[atom.GetSymbol()], list(range(1, 19))
    ) + one_of_k_encoding(PPERIOD_IDS[atom.GetSymbol()], list(range(1, 6)))
    return np.array(encoding_list)


def get_bond_encoding(bond: Chem.rdchem.Bond) -> np.ndarray:
    """Get bond encoding.

    :param bond: Chem.rdchem.Bond bond object
    :return: np.array
    """
    bt = bond.GetBondType()
    return np.array(
        [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
    )


def mol_to_graph(mol):
    """Convert RDKit Mol to NetworkX graph
    Adapted from https://github.com/deepchem/deepchem

    Convert mol into a graph representation atoms are nodes, and bonds
    are vertices stored as graph

    Parameters
    ----------
    mol: rdkit Mol
      The molecule to convert into a graph.

    Returns
    -------
    graph: networkx.Graph
      Contains atoms indices as nodes, edges as bonds.
    """
    G = nx.DiGraph()
    num_atoms = mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))
    if mol.GetNumBonds() == 0:
        # assert num_atoms == 1
        for i in range(mol.GetNumAtoms()):
            G.add_edge(i, i, bond_idx=i)
        return G
    for i in range(mol.GetNumBonds()):
        from_idx = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        to_idx = mol.GetBondWithIdx(i).GetEndAtomIdx()
        G.add_edge(from_idx, to_idx, bond_idx=2 * i)
        G.add_edge(to_idx, from_idx, bond_idx=2 * i + 1)
    return G


def compute_bond_pair_triangles(nx_graph):
    """Compute bond pair triangles."""
    # Line graph edge trackers
    bab_dict, aaa_dict = dict(), dict()
    in_atoms, mid_atoms, out_atoms = [], [], []
    for atom_id in list(nx_graph):
        neighbor_nodes_array = np.array(nx_graph[atom_id])
        # Allow for repeated nodes iff num_atoms < 3
        if len(nx_graph) == 1:
            # Single-atom species
            local_in = np.array([atom_id], dtype=int)
            local_out = np.array([atom_id], dtype=int)
        elif len(nx_graph) == 2:
            # Allow for i=k to handle diatomic molecules
            local_in = np.array(neighbor_nodes_array, dtype=int)
            local_out = np.array(neighbor_nodes_array, dtype=int)
        else:
            # Masking out i=k for others
            nna_rep = np.repeat(
                neighbor_nodes_array[:, np.newaxis], len(neighbor_nodes_array), axis=1
            )
            mask = ~np.eye(nna_rep.shape[0], dtype=bool)
            local_in = nna_rep[mask]
            local_out = nna_rep.transpose()[mask]
        local_mid = np.full_like(local_in, atom_id)
        in_atoms.append(local_in)
        mid_atoms.append(local_mid)
        out_atoms.append(local_out)
    in_atoms = np.concatenate(in_atoms)
    mid_atoms = np.concatenate(mid_atoms)
    out_atoms = np.concatenate(out_atoms)
    in_bonds = np.array(
        [nx_graph[in_atoms[i]][mid_atoms[i]]["bond_idx"] for i in range(len(in_atoms))]
    )
    out_bonds = np.array(
        [nx_graph[mid_atoms[i]][out_atoms[i]]["bond_idx"] for i in range(len(out_atoms))]
    )
    for triangle_idx in range(len(in_atoms)):
        bab_dict[
            (in_bonds[triangle_idx], mid_atoms[triangle_idx], out_bonds[triangle_idx])
        ] = triangle_idx
        aaa_dict[
            (in_atoms[triangle_idx], mid_atoms[triangle_idx], out_atoms[triangle_idx])
        ] = triangle_idx
    return (
        np.stack([in_atoms, mid_atoms, out_atoms, in_bonds, out_bonds], axis=1),
        bab_dict,
        aaa_dict,
    )


def get_conformers_as_tensor(mol, n_conf, return_new_mol=False):
    """Get conformers as tensor."""
    conf_mol = Chem.AddHs(Chem.Mol(mol))
    try:
        AllChem.EmbedMultipleConfs(conf_mol, clearConfs=True, numConfs=n_conf, numThreads=0)
        assert len(conf_mol.GetConformers()) == n_conf
    except AssertionError:
        AllChem.EmbedMultipleConfs(
            conf_mol,
            clearConfs=True,
            numConfs=n_conf,
            numThreads=0,
            useRandomCoords=True,
        )
        assert len(conf_mol.GetConformers()) == n_conf
    AllChem.MMFFOptimizeMoleculeConfs(conf_mol, mmffVariant="MMFF94", numThreads=0)
    conf_mol = Chem.RemoveHs(conf_mol)
    xyzs = np.array([c.GetPositions() for c in conf_mol.GetConformers()])
    assert xyzs.shape[0] == n_conf
    if return_new_mol:
        return xyzs, conf_mol
    return xyzs


def is_potential_stereo_bond(bond: Chem.rdchem.Bond):
    """Check if bond is a potential stereo bond."""
    bond_order = BONDORDER.get(bond.GetBondType(), 1)
    if bond_order > 1.4 and bond_order < 3.0:
        return True
    else:
        return False


def is_potential_stereo_center(atom: Chem.rdchem.Atom):
    """
    Note: Instead of checking whether a real chiral tag can be assigned to the query atom,
    this function regards all neighbor atoms as unique groups.

    Based on rules from https://www.rdkit.org/docs/RDKit_Book.html#brief-description-of-the-findpotentialstereo-algorithm
    """
    assert atom.GetSymbol() != "H"
    if atom.GetDegree() >= 4:
        return True
    elif atom.GetTotalDegree() >= 4:
        return True
    elif atom.GetSymbol() in ["P", "As", "S", "Se"] and atom.GetDegree() >= 3:
        return True
    else:
        return False


def compute_stereo_encodings(
    query_atom_id: int,
    bonded_atom_id: int,
    ref_bond_pos: int,
    mol: Chem.rdchem.Mol,
    nx_graph: networkx.DiGraph,
    atom_idx_on_triangle: tuple,
    ref_geom_xyz,
    cutoff=0.05,
):
    """
    Three types of stereochemistry annotations:
        1. In-clique index of the atom(s) the query atom is bonded to;
        2. On which side of the tangent plane (or unsure);
        3. If off-plane, above or below the plane (or unsure);
    All stereogenic bonds must be explicitly labelled within the input graph.

    :return: torch.Tensor with stereochemistry annotations
    """
    # Is a single atom or not
    stereo_enc = [atom_idx_on_triangle[0] == atom_idx_on_triangle[1]]
    # Is spinor type or not
    stereo_enc += [atom_idx_on_triangle[0] == atom_idx_on_triangle[2]]
    stereo_enc += [atom_idx_on_triangle[i] == query_atom_id for i in range(3)]
    # Whether bonded to other atoms in the triplet
    stereo_enc += [atom_idx_on_triangle[i] in nx_graph[query_atom_id].keys() for i in range(3)]
    # stereo_enc += [
    #     mol.GetAtomWithIdx(bonded_atom_id).GetHybridization()
    #     == Chem.rdchem.HybridizationType.SP
    # ]
    stereo_enc += one_of_k_encoding(ref_bond_pos, [0, 1])
    if ref_geom_xyz is None:
        stereo_enc += [False] * 4
        return np.array(stereo_enc)
    in_bond_vec = ref_geom_xyz[atom_idx_on_triangle[1]] - ref_geom_xyz[atom_idx_on_triangle[0]]
    in_vec = in_bond_vec / np.linalg.norm(in_bond_vec)
    out_bond_vec = ref_geom_xyz[atom_idx_on_triangle[2]] - ref_geom_xyz[atom_idx_on_triangle[1]]
    out_vec = out_bond_vec / np.linalg.norm(out_bond_vec)
    z_vec = np.cross(in_vec, out_vec)

    nx_graph[query_atom_id][bonded_atom_id]["bond_idx"] // 2
    ref_bond_id = (
        nx_graph[atom_idx_on_triangle[ref_bond_pos]][atom_idx_on_triangle[ref_bond_pos + 1]][
            "bond_idx"
        ]
        // 2
    )

    # Resolving bond-centered stereochemistry
    # if mol.GetBondWithIdx(ref_bond_id).GetStereo() in [
    #     Chem.rdchem.BondStereo.STEREOCIS,
    #     Chem.rdchem.BondStereo.STEREOTRANS,
    #     Chem.rdchem.BondStereo.STEREOE,
    #     Chem.rdchem.BondStereo.STEREOZ,
    # ]:
    if is_potential_stereo_bond(mol.GetBondWithIdx(ref_bond_id)):
        query_bond_vec = ref_geom_xyz[query_atom_id] - ref_geom_xyz[bonded_atom_id]
        query_bond_vec = query_bond_vec / np.linalg.norm(query_bond_vec)
        if ref_bond_pos == 0:
            ref_bond_vec = in_vec
            if bonded_atom_id == atom_idx_on_triangle[0]:
                query_bond_vec = -query_bond_vec
        elif ref_bond_pos == 1:
            if bonded_atom_id == atom_idx_on_triangle[1]:
                query_bond_vec = -query_bond_vec
            ref_bond_vec = out_vec
        else:
            raise ValueError
        p_z_bond = np.dot(
            np.cross(ref_bond_vec, query_bond_vec),
            z_vec,
        )
        # print("bond p_z:", p_z_bond)
        if np.abs(p_z_bond) > cutoff:
            bond_stereo_enc = one_of_k_encoding(
                p_z_bond > 0,
                [True, False],
            )
        else:
            # Cannot resolve E/Z geometry
            bond_stereo_enc = [False, False]
    else:
        bond_stereo_enc = [False, False]
    stereo_enc += bond_stereo_enc

    # Resolving atom-centered stereochemistry
    if bonded_atom_id == atom_idx_on_triangle[1] and is_potential_stereo_center(
        mol.GetAtomWithIdx(bonded_atom_id)
    ):
        query_bonded_vec = ref_geom_xyz[query_atom_id] - ref_geom_xyz[bonded_atom_id]
        query_bonded_vec = query_bonded_vec / np.linalg.norm(query_bonded_vec)
        p_z_atom = np.dot(query_bonded_vec, z_vec)
        # print("atom p_z:", p_z_atom)
        if np.abs(p_z_atom) > cutoff:
            atom_stereo_enc = one_of_k_encoding(
                p_z_atom > 0,
                [True, False],
            )
        else:
            atom_stereo_enc = [False, False]
    else:
        atom_stereo_enc = [False, False]
    stereo_enc += atom_stereo_enc

    return np.array(stereo_enc)


def compute_all_stereo_chemistry_encodings(
    mol, nx_graph, atom_idx_on_triangles, aaa_dict, only_2d=False, ref_conf_xyz=None
):
    """Compute all stereochemistry encodings."""
    if not only_2d:
        if ref_conf_xyz is None:
            ref_conf_xyz = get_conformers_as_tensor(mol, 1)[0]
    else:
        ref_conf_xyz = None
    triangle_pairs_list, stereo_encodings_list = [], []
    for triangle_atoms in atom_idx_on_triangles:
        triangle_atoms = tuple(int(aidx) for aidx in triangle_atoms)
        for query_atom in nx_graph[triangle_atoms[0]].keys():
            if query_atom == triangle_atoms[1]:
                continue
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(query_atom, triangle_atoms[0], triangle_atoms[1])],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[0],
                    0,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
        for query_atom in nx_graph[triangle_atoms[1]].keys():
            if query_atom == triangle_atoms[0]:
                continue
            if query_atom == triangle_atoms[2]:
                continue
            # Outgoing bonds
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(triangle_atoms[0], triangle_atoms[1], query_atom)],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[1],
                    0,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
            # Incoming bonds
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(query_atom, triangle_atoms[1], triangle_atoms[2])],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[1],
                    1,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )
        for query_atom in nx_graph[triangle_atoms[2]].keys():
            if query_atom == triangle_atoms[1]:
                continue
            triangle_pairs_list.append(
                [
                    aaa_dict[triangle_atoms],
                    aaa_dict[(triangle_atoms[1], triangle_atoms[2], query_atom)],
                ]
            )
            stereo_encodings_list.append(
                compute_stereo_encodings(
                    query_atom,
                    triangle_atoms[2],
                    1,
                    mol,
                    nx_graph,
                    triangle_atoms,
                    ref_conf_xyz,
                )
            )

    return np.array(triangle_pairs_list), stereo_encodings_list


def tensorize_indexers(
    bond_atom_ids,
    triangle_ids,
    triangle_pair_ids,
    prop_init_ids=None,
    prop_ids=None,
    prop_pair_ids=None,
    allow_dummy=False,
):
    """Tensorize indexers for bond-centered and atom-centered stereochemistry annotations."""
    # Incidence matrices flattened - querying nodes from hyperedges
    if len(bond_atom_ids) == 0:
        if not allow_dummy:
            raise ValueError("There must be at least one bond (ij), got 0")
        bond_atom_ids = np.zeros((0, 2), dtype=np.int_)
    if len(triangle_ids) == 0:
        if not allow_dummy:
            raise ValueError("There must be at least one triplet (ijk), got 0")
        triangle_ids = np.zeros((0, 5), dtype=np.int_)
        prop_init_ids = np.zeros((0, 3), dtype=np.int_)
        # prop_ids = np.zeros((0, 2), dtype=np.int_)
    if len(triangle_pair_ids) == 0:
        triangle_pair_ids = np.zeros((0, 2), dtype=np.int_)
        # prop_pair_ids = np.zeros((0, 3), dtype=np.int_)
    indexer = {
        "gather_idx_ij_i": np.array(bond_atom_ids[:, 0], dtype=np.int_),
        "gather_idx_ij_j": np.array(bond_atom_ids[:, 1], dtype=np.int_),
        "gather_idx_ijk_i": np.array(triangle_ids[:, 0], dtype=np.int_),
        "gather_idx_ijk_j": np.array(triangle_ids[:, 1], dtype=np.int_),
        "gather_idx_ijk_k": np.array(triangle_ids[:, 2], dtype=np.int_),
        "gather_idx_ijk_ij": np.array(triangle_ids[:, 3], dtype=np.int_),
        "gather_idx_ijk_jk": np.array(triangle_ids[:, 4], dtype=np.int_),
        "gather_idx_ijkl_ijk": np.array(triangle_pair_ids[:, 0], dtype=np.int_),
        "gather_idx_ijkl_jkl": np.array(triangle_pair_ids[:, 1], dtype=np.int_),
        # "gather_idx_u0ijk_ijk": np.array(prop_init_ids[:, 0], dtype=np.int_),
        # "gather_idx_u0ijk_u0": np.array(prop_init_ids[:, 1], dtype=np.int_),
    }
    return indexer


def process_molecule(
    mol: Chem.Mol,
    only_2d: bool = False,
    return_mol: bool = False,
    ref_conf_xyz: Optional[np.ndarray] = None,
    return_as_dict: bool = False,
) -> tuple:
    """Process (multi-fragment) molecule and return metadata, indexer and features.

    :param mol: Chem.Mol molecule object
    :param only_2d: whether to compute 2D stereochemistry only
    :param return_mol: whether to return the molecule object
    :param ref_conf_xyz: reference 3D coordinates
    :param return_as_dict: whether to return as dictionary
    :return: feature set dictionary
    """
    # NOTE: all bonds are directional
    atom_encodings_list = [get_atom_encoding(atom) for atom in mol.GetAtoms()]
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    nx_graph = mol_to_graph(mol)
    bond_atom_ids = np.array(nx_graph.edges())
    if mol.GetNumBonds() == 0:
        bond_encodings_list = [np.zeros((4,)) for _ in range(len(bond_atom_ids))]
    else:
        bond_encodings_list = [
            get_bond_encoding(mol.GetBondWithIdx(eid // 2)) for eid in range(len(bond_atom_ids))
        ]
    triangle_aaabb_ids, _, aaa_dict = compute_bond_pair_triangles(nx_graph)
    atom_idx_on_triangles = triangle_aaabb_ids[:, :3]
    triangle_pairs_list, stereo_encodings_list = compute_all_stereo_chemistry_encodings(
        mol,
        nx_graph,
        atom_idx_on_triangles,
        aaa_dict,
        only_2d=only_2d,
        ref_conf_xyz=ref_conf_xyz,
    )
    metadata = {
        "num_molid": 1,
        "num_ligand_atom": len(atom_encodings_list),
        "num_i": len(atom_encodings_list),
        "num_j": len(atom_encodings_list),
        "num_k": len(atom_encodings_list),
        "num_u": len(atom_encodings_list),
        "num_ij": len(bond_atom_ids),
        "num_jk": len(bond_atom_ids),
        "num_ijk": len(triangle_aaabb_ids),
        "num_jkl": len(triangle_aaabb_ids),
        "num_ligand_clique": len(triangle_aaabb_ids),
        "num_ijkl": len(triangle_pairs_list),
    }
    indexer = tensorize_indexers(bond_atom_ids, triangle_aaabb_ids, triangle_pairs_list)
    # NOTE: the target `molid` is always `0`
    indexer["gather_idx_i_molid"] = np.zeros((metadata["num_i"],), dtype=np.int_)
    indexer["gather_idx_ijk_molid"] = np.zeros((metadata["num_ijk"],), dtype=np.int_)
    if len(stereo_encodings_list) == 0:
        stereo_encodings = np.zeros((0, 14), dtype=np.int_)
    else:
        stereo_encodings = np.stack(stereo_encodings_list, axis=0)
    features = {
        "atomic_numbers": np.array(atomic_numbers, dtype=int),
        "atom_encodings": np.stack(atom_encodings_list, axis=0),
        "bond_encodings": np.stack(bond_encodings_list, axis=0),
        "stereo_chemistry_encodings": stereo_encodings,
        "sdf_coordinates": ref_conf_xyz,
    }
    if return_mol:
        if return_as_dict:
            return {
                "metadata": metadata,
                "indexer": indexer,
                "features": features,
                "misc": {},
                "mol": mol,
            }
        return mol, metadata, indexer, features
    if return_as_dict:
        return {
            "metadata": metadata,
            "indexer": indexer,
            "features": features,
            "misc": {},
        }
    return metadata, indexer, features


def smiles2inputs(tokenizer, smiles, pad_length=128):
    """Adapted from megamolbart."""

    assert isinstance(smiles, str)
    if pad_length:
        assert pad_length >= len(smiles) + 2

    tokens = tokenizer.tokenize([smiles], pad=True)

    # Append to tokens and mask if appropriate
    if pad_length:
        for i in range(len(tokens["original_tokens"])):
            num_i = len(tokens["original_tokens"][i])
            n_pad = pad_length - len(tokens["original_tokens"][i])
            tokens["original_tokens"][i] += [tokenizer.pad_token] * n_pad
            tokens["masked_pad_masks"][i] += [1] * n_pad

    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens["original_tokens"]))
    pad_mask = torch.tensor(tokens["masked_pad_masks"]).bool()
    encode_input = {
        "features": {"encoder_input": token_ids, "encoder_pad_mask": pad_mask},
        "indexer": {"gather_idx_i_molid": np.zeros((num_i,), dtype=np.int_)},
        "metadata": {"num_i": num_i, "num_molid": 1},
    }

    return encode_input


def process_mol_file(
    fname,
    featurize=True,
    tokenizer=None,
    return_mol=False,
    sanitize=True,
    coord_feats=True,
    pair_feats=False,
    discard_coords=False,
    **kwargs,
):
    """Process molecule file and return feature set."""
    if fname.endswith("msgpack"):
        with open(fname, "rb") as data_file:
            byte_data = data_file.read()
            data_loaded = msgpack.unpackb(byte_data)
        return data_loaded
    elif fname.endswith("sdf"):
        fsuppl = Chem.SDMolSupplier(fname, sanitize=sanitize)
        mol = next(fsuppl)
        if not sanitize:
            mol.UpdatePropertyCache(strict=False)
    elif fname.endswith("mol2"):
        mol = Chem.MolFromMol2File(fname, sanitize=sanitize)
    elif fname.endswith("pdb"):
        mol = Chem.MolFromPDBFile(fname, sanitize=sanitize)
        if not sanitize:
            mol.UpdatePropertyCache(strict=False)
    else:
        log.warning("No suffix found for ligand input, assuming SMILES input")
        mol = Chem.MolFromSmiles(fname, sanitize=sanitize)
    if sanitize:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
    else:
        mol = Chem.RemoveHs(mol, sanitize=False)
    if not featurize:
        return mol
    if tokenizer is not None:
        return smiles2inputs(tokenizer, Chem.rdmolfiles.MolToSmiles(mol))
    if discard_coords:
        conf_xyz = get_conformers_as_tensor(mol, 1)[0]
    else:
        conf_xyz = np.array(mol.GetConformer().GetPositions())
    metadata, indexer, features = process_molecule(
        mol, return_mol=False, ref_conf_xyz=conf_xyz, **kwargs
    )
    if coord_feats:
        features["sdf_coordinates"] = conf_xyz
    feature_set = {
        "metadata": metadata,
        "indexer": indexer,
        "features": features,
        "misc": {},
    }
    if pair_feats:
        attach_pair_idx_and_encodings(feature_set, **kwargs)
    if return_mol:
        return feature_set, mol
    return feature_set


def attach_pair_idx_and_encodings(feature_set, max_n_frames=None, lazy_eval=False):
    """Attach pair index and encodings to the feature set."""
    if lazy_eval:
        if "scatter_idx_u0ijk_Uijk" in feature_set["indexer"].keys():
            return feature_set
    num_triplets = feature_set["metadata"]["num_ijk"]
    num_atoms = feature_set["metadata"]["num_i"]
    num_frame_pairs = feature_set["metadata"]["num_ijkl"]
    if max_n_frames is None:
        max_n_frames = num_triplets
    else:
        max_n_frames = max(min(num_triplets, max_n_frames), 1)
    key_frame_idx = np.random.choice(num_triplets, size=max_n_frames, replace=False)
    key_atom_idx = feature_set["indexer"]["gather_idx_ijk_j"][key_frame_idx]
    num_key_frames = key_frame_idx.shape[0]
    # scatter_idx_u0ijk_Uijk = (
    #     feature_set["indexer"]["gather_idx_u0ijk_ijk"]
    #     + feature_set["indexer"]["gather_idx_u0ijk_u0"] * num_triplets
    # )
    gather_idx_UI_Uijk = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None],
            (num_key_frames, num_key_frames),
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            key_frame_idx[None, :],
            (num_key_frames, num_key_frames),
        ).flatten()
    )
    gather_idx_Uijkl_Uijk = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None], (num_key_frames, num_frame_pairs)
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            feature_set["indexer"]["gather_idx_ijkl_ijk"][None, :],
            (num_key_frames, num_frame_pairs),
        ).flatten()
    )
    gather_idx_Uijkl_ujkl = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None], (num_key_frames, num_frame_pairs)
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            feature_set["indexer"]["gather_idx_ijkl_jkl"][None, :],
            (num_key_frames, num_frame_pairs),
        ).flatten()
    )
    gather_idx_Uijkl_ijkl = np.broadcast_to(
        np.arange(num_frame_pairs)[None, :], (num_key_frames, num_frame_pairs)
    ).flatten()

    adjacency_mat = np.zeros((num_atoms, num_atoms), dtype=np.int_)
    adjacency_mat[
        feature_set["indexer"]["gather_idx_ij_i"],
        feature_set["indexer"]["gather_idx_ij_j"],
    ] = 1
    sum_pair_path_dist = [np.eye(num_atoms, dtype=np.int_)]
    for path_length in range(3):
        sum_pair_path_dist.append(np.matmul(sum_pair_path_dist[-1], adjacency_mat))
    sum_pair_path_dist = np.stack(sum_pair_path_dist, axis=2)
    atom_pair_feature_mat = np.zeros((num_atoms, num_atoms, 4), dtype=np.float_)
    atom_pair_feature_mat[
        feature_set["indexer"]["gather_idx_ij_i"],
        feature_set["indexer"]["gather_idx_ij_j"],
    ] = feature_set["features"]["bond_encodings"]
    atom_pair_feature_mat = np.concatenate(
        [atom_pair_feature_mat, (sum_pair_path_dist > 0).astype(np.float_)], axis=2
    )
    uv_adj_mat = np.sum(sum_pair_path_dist, axis=2) > 0
    gather_idx_uv_u = np.broadcast_to(np.arange(num_atoms)[:, None], (num_atoms, num_atoms))[
        uv_adj_mat
    ]
    gather_idx_uv_v = np.broadcast_to(np.arange(num_atoms)[None, :], (num_atoms, num_atoms))[
        uv_adj_mat
    ]

    atom_frame_pair_feat_initial_ = np.concatenate(
        [
            atom_pair_feature_mat[key_atom_idx, :][:, feature_set["indexer"]["gather_idx_ijk_i"]],
            atom_pair_feature_mat[key_atom_idx, :][:, feature_set["indexer"]["gather_idx_ijk_j"]],
            atom_pair_feature_mat[key_atom_idx, :][:, feature_set["indexer"]["gather_idx_ijk_k"]],
        ],
        axis=2,
    ).reshape((num_key_frames * num_triplets, atom_pair_feature_mat.shape[2] * 3))

    # Generate on-the-fly to reduce disk usage
    feature_set["indexer"].update(
        {
            "gather_idx_U_u": key_atom_idx,
            "gather_idx_I_ijk": key_frame_idx,
            "gather_idx_I_molid": np.zeros((num_key_frames,), dtype=np.int_),
            "gather_idx_UI_Uijk": gather_idx_UI_Uijk,
            "gather_idx_UI_u": np.broadcast_to(
                key_atom_idx[:, None], (num_key_frames, num_key_frames)
            ).flatten(),
            "gather_idx_UI_U": np.broadcast_to(
                np.arange(num_key_frames)[:, None], (num_key_frames, num_key_frames)
            ).flatten(),
            "gather_idx_UI_I": np.broadcast_to(
                np.arange(num_key_frames)[None, :], (num_key_frames, num_key_frames)
            ).flatten(),
            # "scatter_idx_u0ijk_Uijk": scatter_idx_u0ijk_Uijk,
            "gather_idx_Uijk_u": np.broadcast_to(
                key_atom_idx[:, None], (num_key_frames, num_triplets)
            ).flatten(),
            "gather_idx_Uijk_ijk": np.broadcast_to(
                np.arange(num_triplets)[None, :], (num_key_frames, num_triplets)
            ).flatten(),
            "gather_idx_Uijkl_Uijk": gather_idx_Uijkl_Uijk,
            "gather_idx_Uijkl_ujkl": gather_idx_Uijkl_ujkl,
            "gather_idx_Uijkl_ijkl": gather_idx_Uijkl_ijkl,
            "gather_idx_uv_u": gather_idx_uv_u,
            "gather_idx_uv_v": gather_idx_uv_v,
        }
    )
    feature_set["features"].update(
        {
            "atom_pair_encodings": atom_pair_feature_mat[uv_adj_mat],
            "atom_frame_pair_encodings": atom_frame_pair_feat_initial_,
        }
    )
    feature_set["metadata"]["num_v"] = num_atoms
    feature_set["metadata"]["num_I"] = num_key_frames
    feature_set["metadata"]["num_U"] = num_key_frames
    feature_set["metadata"]["num_Uijk"] = num_triplets * num_key_frames
    feature_set["metadata"]["num_ujkl"] = num_triplets * num_key_frames
    feature_set["metadata"]["num_Uijkl"] = num_frame_pairs * num_key_frames
    feature_set["metadata"]["num_uv"] = gather_idx_uv_u.shape[0]
    feature_set["metadata"]["num_UI"] = num_key_frames * num_key_frames
    return feature_set


def iterable_query(k1, k2, dc):
    """Query a key from a list of dictionaries."""
    return [d[k1][k2] for d in dc]


def collate_idx_numpy(idx_ten_list, dst_sample_sizes):
    """Collate NumPy index tensors."""
    elewise_dst_offsets = np.repeat(
        np.cumsum(np.array([0] + dst_sample_sizes[:-1], dtype=np.int_), axis=0),
        np.array([idx_ten.shape[0] for idx_ten in idx_ten_list], dtype=np.int_),
    )
    col_idx_ten = np.concatenate(idx_ten_list, axis=0) + elewise_dst_offsets
    return col_idx_ten


def collate_idx_tensors(idx_ten_list, dst_sample_sizes):
    """Collate PyTorch index tensors."""
    device = idx_ten_list[0].device
    elewise_dst_offsets = torch.repeat_interleave(
        torch.cumsum(
            torch.tensor([0] + dst_sample_sizes[:-1], dtype=torch.long, device=device),
            dim=0,
        ),
        torch.tensor(
            [idx_ten.size(0) for idx_ten in idx_ten_list],
            dtype=torch.long,
            device=device,
        ),
    )
    col_idx_ten = torch.cat(idx_ten_list, dim=0).add_(elewise_dst_offsets)
    return col_idx_ten


def collate_samples(list_of_samples: list, exclude=[]):
    """Collate samples into a batch."""
    list_of_samples = [sample for sample in list_of_samples if sample is not None]
    batch_metadata = dict()
    for key in list_of_samples[0]["metadata"].keys():
        batch_metadata[f"{key}_per_sample"] = iterable_query("metadata", key, list_of_samples)
        if key.startswith("num_"):
            if key.endswith("_per_sample"):
                # Track the batch-level summary only
                continue
            batch_metadata[key] = sum(batch_metadata[f"{key}_per_sample"])

    batch_indexer = dict()
    for indexer_name in list_of_samples[0]["indexer"].keys():
        indexer_name_parts = indexer_name.split("_")
        batch_indexer[indexer_name] = collate_idx_tensors(
            iterable_query("indexer", indexer_name, list_of_samples),
            batch_metadata[f"num_{indexer_name_parts[-1]}_per_sample"],
        )

    batch_features = dict()
    for feature_name in list_of_samples[0]["features"].keys():
        if feature_name in exclude:
            continue
        batch_features[feature_name] = torch.cat(
            iterable_query("features", feature_name, list_of_samples), dim=0
        ).float()
    ret_batch = {
        "metadata": batch_metadata,
        "indexer": batch_indexer,
        "features": batch_features,
        "batch_size": len(list_of_samples),
    }
    if "misc" in list_of_samples[0].keys():
        ret_batch["misc"] = {
            key: [x for sample in list_of_samples for x in sample["misc"][key]]
            for key in list_of_samples[0]["misc"].keys()
        }
    if "labels" in list_of_samples[0].keys():
        ret_batch["labels"] = dict()
        for label_name in list_of_samples[0]["labels"].keys():
            ret_batch["labels"][label_name] = torch.stack(
                iterable_query("labels", label_name, list_of_samples), dim=0
            ).float()
    return ret_batch


def collate_numpy_samples(list_of_samples: list):
    """Collate NumPy samples into a batch."""
    list_of_samples = [sample for sample in list_of_samples if sample is not None]
    if len(list_of_samples) == 0:
        return {
            "metadata": {},
            "indexer": {},
            "features": {},
            "labels": {},
            "misc": {},
            "batch_size": 0,
        }
    batch_metadata = dict()
    for key in list_of_samples[0]["metadata"].keys():
        batch_metadata[f"{key}_per_sample"] = iterable_query("metadata", key, list_of_samples)
        if key.startswith("num_"):
            if key.endswith("_per_sample"):
                continue
            batch_metadata[key] = sum(batch_metadata[f"{key}_per_sample"])

    batch_indexer = dict()
    for indexer_name in list_of_samples[0]["indexer"].keys():
        i2 = indexer_name.split("_")[-1]
        batch_indexer[indexer_name] = collate_idx_numpy(
            iterable_query("indexer", indexer_name, list_of_samples),
            batch_metadata[f"num_{i2}_per_sample"],
        )

    batch_features = dict()
    for feature_name in list_of_samples[0]["features"].keys():
        batch_features[feature_name] = np.concatenate(
            iterable_query("features", feature_name, list_of_samples), axis=0
        ).astype(np.float32)
    batch_misc = {
        key: [x for sample in list_of_samples for x in sample["misc"][key]]
        for key in list_of_samples[0]["misc"].keys()
    }
    if "labels" not in list_of_samples[0].keys():
        return {
            "metadata": batch_metadata,
            "indexer": batch_indexer,
            "features": batch_features,
            "misc": batch_misc,
            "batch_size": len(list_of_samples),
        }

    batch_labels = dict()
    for label_name in list_of_samples[0]["labels"].keys():
        batch_labels[label_name] = np.stack(
            iterable_query("labels", label_name, list_of_samples), axis=0
        ).astype(np.float32)
    return {
        "metadata": batch_metadata,
        "indexer": batch_indexer,
        "features": batch_features,
        "labels": batch_labels,
        "misc": batch_misc,
        "batch_size": len(list_of_samples),
    }
