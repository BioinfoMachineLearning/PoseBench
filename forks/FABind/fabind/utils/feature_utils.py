from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
import os
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
import requests
from rdkit.Geometry import Point3D

from torchdrug import data as td     # conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg if fail to import
from torch_geometric.utils import dense_to_sparse

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    # Chem.WrapLogs()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem


def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()

def get_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveAllHs(mol)
    # mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    # rid = AllChem.EmbedMolecule(mol, ps)
    for repeat in range(n_repeat):
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == 0:
            break
    if rid == -1:
        print("rid", pdb, rid)
        ps.useRandomCoords = True
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == -1:
            mol.Compute2DCoords()
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # mol = Chem.RemoveAllHs(mol)
    return mol


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

# mol_mask[i][j] = 1 means that atom i and atom j are  
# connected by a bond(origin adjacent matrix), or 2-hop away, or in the same ring structure
def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution


def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False):
    # N x 3
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        # N x N
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    else:
        LAS_distance_constraint_mask = None
    # N x N x 16. Convert continuous distance to 16 classes(distance ranges from 0 to 15) one-hot vector. 
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask)
    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')
    # molstd = td.Molecule.from_molecule(mol ,node_feature=['property_prediction'])
    compound_node_features = molstd.node_feature # nodes_chemical_features
    edge_list = molstd.edge_list # [num_edge, 3] (node_in, node_out, relation)
    edge_weight = molstd.edge_weight # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature # [num_edge, edge_feature_dim]
    # coords: N x 3
    # compound_node_features: N x 56
    # edge_list: E x 3
    # edge_features: E x 19
    # pair_dis_distribution: N x N x 16
    x = (coords, compound_node_features, edge_list, edge_feature, pair_dis_distribution)
    return x

def extract_torchdrug_feature_from_mol_E3Bind(mol, has_LAS_mask=False):
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
        LAS_edge_index, _ = dense_to_sparse(LAS_distance_constraint_mask)
    else:
        LAS_distance_constraint_mask = None
        LAS_edge_index = None

    # assert LAS_distance_constraint_mask.transpose(0, 1) == LAS_distance_constraint_mask
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask)   # CONFUSE
    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')  #  it's the right way
    # molstd = td.Molecule.from_molecule(mol ,node_feature=['property_prediction'])
    compound_node_features = molstd.node_feature # nodes_chemical_features
    edge_list = molstd.edge_list # [num_edge, 3]
    edge_weight = molstd.edge_weight # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature # [num_edge, edge_feature_dim]
    x = (coords, compound_node_features, edge_list, edge_feature, pair_dis_distribution, LAS_edge_index)
    return x

import gvp
import gvp.data

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = gvp.data.ProteinGraphDataset([structure])
    protein = dataset[0]
    # x: alpha carbon coordinates, shape N * 3
    # seq: sequence converted to int tensor according to attribute self.letter_to_num, shape N
    # node_s, node_v: node features as described in the paper with dims (6, 3). s: N * 6, v: N * 3 * 3
    # edge_index: edge index tensor, shape 2 * E
    # edge_s, edge_v: edge features as described in the paper with dims (32, 1). s: E * 32, v: E * 1 * 3
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x

# Seed_everything(seed=42)

# used for testing.
def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile):
    clean_res_list, ligand_list = save_cleaned_protein(c, proteinFile)
    chain = c.id
    # should take a look of this ligand_list to ensure we choose the right ligand.
    seq_id = ligand_seq_id
    # download the ligand in sdf format from rcsb.org. because we pdb format doesn't contain bond information.
    # you could also use openbabel to do this.
    url = f"https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_seq_id={seq_id}&encoding=sdf&filename=ligand.sdf"
    r = requests.get(url)
    open(ligandFile , 'wb').write(r.content)
    return clean_res_list, ligand_list

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()

def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)

def generate_sdf_from_smiles_using_rdkit_E3Bind(smiles, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    # write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)
    return new_coords

def generate_and_write_sdf_from_smiles_using_rdkit_E3Bind(smiles, rdkitMolFile=None, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    if rdkitMolFile is not None:
        write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)
    return new_coords

def select_chain_within_cutoff_to_ligand_v2(x):
    # pdbFile = f"/pdbbind2020/pdbbind_files/{pdb}/{pdb}_protein.pdb"
    # ligandFile = f"/pdbbind2020/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    # toFile = f"{toFolder}/{pdb}_protein.pdb"
    # cutoff = 10
    pdbFile, ligandFile, cutoff, toFile = x
    
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    all_atoms = [atom for res in all_res for atom in res.get_atoms()]
    protein_coords = np.array([atom.coord for atom in all_atoms])
    chains = np.array([atom.full_id[2] for atom in all_atoms])

    mol = Chem.MolFromMolFile(ligandFile)
    lig_coords = mol.GetConformer().GetPositions()

    protein_atom_to_lig_atom_dis = scipy.spatial.distance.cdist(protein_coords, lig_coords)

    is_in_contact = (protein_atom_to_lig_atom_dis < cutoff).max(axis=1)
    chains_in_contact = set(chains[is_in_contact])
    
    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())