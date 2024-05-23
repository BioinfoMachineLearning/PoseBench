
import torch
from utils.metrics import *
import numpy as np
import pandas as pd
import scipy.spatial
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchmetrics
from torch_scatter import scatter_mean

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# from feature_utils import read_mol
import sys
from io import StringIO
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

def uniform_random_rotation(x):
    """Apply a random rotation in 3D, with a distribution uniform over the
    sphere.
    Arguments:
        x: vector or set of vectors with dimension (n, 3), where n is the
            number of vectors
    Returns:
        Array of shape (n, 3) containing the randomly rotated vectors of x,
        about the mean coordinate of x.
    Algorithm taken from "Fast Random Rotation Matrices" (James Avro, 1992):
    https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    """

    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = np.eye(3)
        x1 = np.random.rand()
        R[0, 0] = R[1, 1] = np.cos(2 * np.pi * x1)
        R[0, 1] = -np.sin(2 * np.pi * x1)
        R[1, 0] = np.sin(2 * np.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * np.pi * np.random.rand()
    x3 = np.random.rand()
    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = np.array([
        np.cos(x2) * np.sqrt(x3),
        np.sin(x2) * np.sqrt(x3),
        np.sqrt(1 - x3)
    ])
    H = np.eye(3) - (2 * np.outer(v, v))
    M = -(H @ R)
    x = x.reshape((-1, 3))
    mean_coord = np.mean(x, axis=0)
    return ((x - mean_coord) @ M) + mean_coord @ M

def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info

def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5

def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode):
    # protein
    input_edge_list = []
    input_protein_edge_feature_idx = []
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v

# During training, this function will add 5A noise to the ligand/pocket center.
def get_keepNode(com, protein_node_xyz, n_node, pocket_radius, use_whole_protein, 
                     use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com):
    if use_whole_protein:
        keepNode = np.ones(n_node, dtype=bool)
    else:
        keepNode = np.zeros(n_node, dtype=bool)
        # extract node based on compound COM.
        if use_compound_com_as_pocket:
            if add_noise_to_com: # com is the mean coordinate of the compound
                com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, com)
                keepNode[i] = dis < pocket_radius

    if chosen_pocket_com is not None:
        another_keepNode = np.zeros(n_node, dtype=bool)
        for a_com in chosen_pocket_com:
            if add_noise_to_com:
                a_com = a_com + add_noise_to_com * (2 * np.random.rand(*a_com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, a_com)
                another_keepNode[i] |= dis < pocket_radius
        keepNode |= another_keepNode
    return keepNode


def compute_dis_between_two_vector_tensor(a, b):
    return torch.sqrt(torch.sum((a - b)**2, dim=-1))

def get_keepNode_tensor(protein_node_xyz, pocket_radius, add_noise_to_com, chosen_pocket_com):
    if add_noise_to_com:
        chosen_pocket_com = chosen_pocket_com + add_noise_to_com * (2 * torch.rand_like(chosen_pocket_com) - 1)
    # Compute the distances between all nodes and the chosen_pocket_com in a vectorized manner
    dis = compute_dis_between_two_vector_tensor(protein_node_xyz, chosen_pocket_com.unsqueeze(0))
    # Create the keepNode tensor using a boolean mask
    keepNode = dis < pocket_radius

    return keepNode

def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )

def construct_data_from_graph_gvp_mean(args, protein_node_xyz, protein_seq,
                                 coords, compound_node_features, input_atom_edge_list, 
                                 input_atom_edge_attr_list, LAS_edge_index, rdkit_coords, compound_coords_init_mode='pocket_center_rdkit', includeDisMap=True, pdb_id=None, group='train', seed=42, data_path=None, contactCutoff=8.0, pocket_radius=20, interactionThresholdDistance=10, compoundMode=1, 
                                 add_noise_to_com=None, use_whole_protein=False, use_compound_com_as_pocket=True, chosen_pocket_com=None, random_rotation=False, pocket_idx_no_noise=True, protein_esm2_feat=None):
    n_node = protein_node_xyz.shape[0]
    # n_compound_node = coords.shape[0]
    # normalize the protein and ligand coords
    coords_bias = protein_node_xyz.mean(dim=0)
    coords = coords - coords_bias.numpy()
    protein_node_xyz = protein_node_xyz - coords_bias
    # centroid instead of com. 
    com = coords.mean(axis=0)
    if args.train_pred_pocket_noise and group == 'train':
        keepNode = get_keepNode(com, protein_node_xyz.numpy(), n_node, pocket_radius, use_whole_protein, 
                         use_compound_com_as_pocket, args.train_pred_pocket_noise, chosen_pocket_com)
    else:
        keepNode = get_keepNode(com, protein_node_xyz.numpy(), n_node, pocket_radius, use_whole_protein, 
                                use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com)

    keepNode_no_noise = get_keepNode(com, protein_node_xyz.numpy(), n_node, pocket_radius, use_whole_protein, 
                            use_compound_com_as_pocket, None, chosen_pocket_com)

    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    input_node_xyz = protein_node_xyz[keepNode]
    # input_edge_idx, input_protein_edge_s, input_protein_edge_v = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode)

    # construct heterogeneous graph data.
    data = HeteroData()

    # only if your ligand is real this y_contact is meaningful. Distance map between ligand atoms and protein amino acids.
    dis_map = scipy.spatial.distance.cdist(input_node_xyz.cpu().numpy(), coords)
    # y_contact = dis_map < contactCutoff # contactCutoff is 8A
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map>interactionThresholdDistance] = interactionThresholdDistance
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()
    # TODO The difference between contactCutoff and interactionThresholdDistance:
    # contactCutoff is for classification evaluation, interactionThresholdDistance is for distance regression.
    # additional information. keep records.
    data.node_xyz = input_node_xyz
    data.coords = torch.tensor(coords, dtype=torch.float)
    # data.y = torch.tensor(y_contact, dtype=torch.float).flatten() # whether the distance between ligand and protein is less than 8A.

    # pocket information
    if torch.is_tensor(protein_esm2_feat):
        data['pocket'].node_feats = protein_esm2_feat[keepNode]
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    data['pocket'].keepNode = torch.tensor(keepNode, dtype=torch.bool)
    
    data['compound'].node_feats = compound_node_features.float()
    data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
    
    # complex information
    n_protein = input_node_xyz.shape[0]
    n_protein_whole = protein_node_xyz.shape[0]
    n_compound = compound_node_features.shape[0]
    # use zero coord to init compound
    # data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
    #     (torch.zeros(n_compound + 2, 3), input_node_xyz), dim=0
    #     ).float()

    if args.local_eval:
        if group == 'test':
            from accelerate.utils import set_seed
            set_seed(seed)
            pre = args.data_path
            mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb_id}.sdf", None)
            rotable_bonds = get_torsions(mol)
            values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
            for idx in range(len(rotable_bonds)):
                SetDihedral(mol.GetConformer(), rotable_bonds[idx], values[idx])
            Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
            rdkit_coords = uniform_random_rotation(mol.GetConformer().GetPositions())
        
    if args.train_ligand_torsion_noise and group == 'train':
        pre = data_path
        try:
            mol = Chem.MolFromMolFile(f"{pre}/renumber_atom_index_same_as_smiles/{pdb_id}.sdf", sanitize=False)
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            mol = Chem.RemoveHs(mol)
                
            # mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb_id}.sdf", None)
        except:
            raise ValueError(f"cannot find {pdb_id}.sdf in {pre}/renumber_atom_index_same_as_smiles/")
        rotable_bonds = get_torsions(mol)
        # np.random.seed(np_seed)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformer(), rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer())
        rdkit_coords = uniform_random_rotation(mol.GetConformer().GetPositions())


    if compound_coords_init_mode == 'random':
        coords_init = 4 * (2 * torch.rand(coords.shape) - 1)
    elif compound_coords_init_mode == 'perturb_3A':
        coords_init = torch.tensor(coords) + 3 * (2 * torch.rand(coords.shape) - 1) 
    elif compound_coords_init_mode == 'perturb_4A':
        coords_init = torch.tensor(coords) + 4 * (2 * torch.rand(coords.shape) - 1)
    elif compound_coords_init_mode == 'perturb_5A':
        coords_init = torch.tensor(coords) + 5 * (2 * torch.rand(coords.shape) - 1)
    elif compound_coords_init_mode == 'compound_center':
        coords_init = torch.tensor(com).reshape(1, 3) + 10 * (2 * torch.rand(coords.shape) - 1)
    elif compound_coords_init_mode == 'pocket_center':
        coords_init = input_node_xyz.mean(dim=0).reshape(1, 3) + 5 * (2 * torch.rand(coords.shape) - 1)
    elif compound_coords_init_mode == 'pocket_center_rdkit':
        if random_rotation:
            rdkit_coords = torch.tensor(uniform_random_rotation(rdkit_coords))
        else:
            rdkit_coords = torch.tensor(rdkit_coords)
        coords_init = rdkit_coords - rdkit_coords.mean(dim=0).reshape(1, 3) + input_node_xyz.mean(dim=0).reshape(1, 3)
    elif compound_coords_init_mode == 'redocking':
        coords_rot = torch.tensor(uniform_random_rotation(coords))
        coords_init = coords_rot - coords_rot.mean(dim=0).reshape(1, 3) + input_node_xyz.mean(dim=0).reshape(1, 3)
    elif compound_coords_init_mode == 'redocking_no_rotate':
        coords_rot = torch.tensor(coords)
        coords_init = coords_rot - coords_rot.mean(dim=0).reshape(1, 3) + input_node_xyz.mean(dim=0).reshape(1, 3)
    
    # ground truth ligand and pocket
    data['complex'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            coords_init,
            torch.zeros(1, 3), 
            input_node_xyz
        ), dim=0
    ).float()

    if compound_coords_init_mode == 'redocking' or compound_coords_init_mode == 'redocking_no_rotate':
        data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                torch.tensor(coords),
                torch.zeros(1, 3), 
                torch.zeros_like(input_node_xyz)
            ), dim=0
        ).float()
    else:
        data['complex'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3), 
                torch.zeros_like(input_node_xyz)
            ), dim=0
        ).float()
    
    segment = torch.zeros(n_protein + n_compound + 2)
    segment[n_compound+1:] = 1 # compound: 0, protein: 1
    data['complex'].segment = segment # protein or ligand
    mask = torch.zeros(n_protein + n_compound + 2)
    mask[:n_compound+2] = 1 # glb_p can be updated
    data['complex'].mask = mask.bool()
    is_global = torch.zeros(n_protein + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound+1] = 1
    data['complex'].is_global = is_global.bool()

    data['complex', 'c2c', 'complex'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
    if compound_coords_init_mode == 'redocking' or compound_coords_init_mode == 'redocking_no_rotate':
        data['complex', 'LAS', 'complex'].edge_index = torch.nonzero(torch.ones(n_compound, n_compound)).t() + 1
    else:
        data['complex', 'LAS', 'complex'].edge_index = LAS_edge_index + 1

    # ground truth ligand and whole protein
    data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
        (
            torch.zeros(1, 3),
            coords_init - coords_init.mean(dim=0).reshape(1, 3), # for pocket prediction module, the ligand is centered at the protein center/origin
            torch.zeros(1, 3), 
            protein_node_xyz
        ), dim=0
    ).float()

    if compound_coords_init_mode == 'redocking' or compound_coords_init_mode == 'redocking_no_rotate':
        data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                torch.tensor(coords),
                torch.zeros(1, 3), 
                torch.zeros_like(protein_node_xyz)
            ), dim=0
        ).float()
    else:
        data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3), 
                torch.zeros_like(protein_node_xyz)
            ), dim=0
        ).float()

    segment = torch.zeros(n_protein_whole + n_compound + 2)
    segment[n_compound+1:] = 1 # compound: 0, protein: 1
    data['complex_whole_protein'].segment = segment # protein or ligand
    mask = torch.zeros(n_protein_whole + n_compound + 2)
    mask[:n_compound+2] = 1 # glb_p can be updated
    data['complex_whole_protein'].mask = mask.bool()
    is_global = torch.zeros(n_protein_whole + n_compound + 2)
    is_global[0] = 1
    is_global[n_compound+1] = 1
    data['complex_whole_protein'].is_global = is_global.bool()

    data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
    if compound_coords_init_mode == 'redocking' or compound_coords_init_mode == 'redocking_no_rotate':
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = torch.nonzero(torch.ones(n_compound, n_compound)).t() + 1
    else:
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1
    

    # for stage 3
    data['compound'].node_coords = coords_init
    data['compound'].rdkit_coords = rdkit_coords
    data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()
    data['LAS_edge_list'].x = data['complex', 'LAS', 'complex'].edge_index.clone().t()
    # add whole protein information for pocket prediction

    data.node_xyz_whole = protein_node_xyz
    data.coords_center = torch.tensor(com, dtype=torch.float).unsqueeze(0)
    data.seq_whole = protein_seq
    data.coord_offset = coords_bias.unsqueeze(0)
    # save the pocket index for binary classification
    if pocket_idx_no_noise:
        data.pocket_idx = torch.tensor(keepNode_no_noise, dtype=torch.int)
    else:
        data.pocket_idx = torch.tensor(keepNode, dtype=torch.int)

    if torch.is_tensor(protein_esm2_feat):
        data['protein_whole'].node_feats = protein_esm2_feat
    else:
        raise ValueError("protein_esm2_feat should be a tensor")

    return data, input_node_xyz, keepNode


@torch.no_grad()
def evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, data_loader, model, com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, relative_k, device, pred_dis=False, info=None, saveFileName=None, use_y_mask=False, skip_y_metrics_evaluation=False, stage=1):
    y_list = []
    y_pred_list = []
    com_coord_list = []
    com_coord_pred_list = []
    # contain the ground truth for classiifcation(may not all)
    pocket_coord_list = []
    pocket_coord_pred_list = []
    # contain the ground truth for regression(all)
    pocket_coord_direct_list = []
    pocket_coord_pred_direct_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []
    # protein_len_list = []
    # real_y_mask_list = []

    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pdb_list = []

    skip_count = 0
    count = 0
    batch_loss = 0.0
    batch_by_pred_loss = 0.0
    batch_distill_loss = 0.0
    com_coord_batch_loss = 0.0
    pocket_cls_batch_loss = 0.0
    pocket_coord_direct_batch_loss = 0.0
    keepNode_less_5_count = 0
    if args.disable_tqdm:
        data_iter = data_loader
    else:
        data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
    for data in data_iter:
        data = data.to(device)
        com_coord_pred, compound_batch, y_pred, y_pred_by_coord, pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pocket_coord_pred_direct, dis_map, keepNode_less_5 = model(data, stage=stage)       
        # y = data.y
        com_coord = data.coords
        
        sd = ((com_coord_pred - com_coord) ** 2).sum(dim=-1)
        rmsd = scatter_mean(src=sd, index=compound_batch, dim=0).sqrt()
        
        centroid_pred = scatter_mean(src=com_coord_pred, index=compound_batch, dim=0)
        centroid_true = scatter_mean(src=com_coord, index=compound_batch, dim=0)
        centroid_dis = (centroid_pred - centroid_true).norm(dim=-1)

        if pred_dis:
            contact_loss = args.pair_distance_loss_weight * criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
            contact_by_pred_loss = args.pair_distance_loss_weight * criterion(y_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
            contact_distill_loss = args.pair_distance_distill_loss_weight * criterion(y_pred_by_coord, y_pred) if len(y_pred) > 0 else torch.tensor([0])
        else:
            contact_loss = criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
            y_pred = y_pred.sigmoid()

        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float())
        pocket_coord_direct_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pocket_coord_pred_direct, data.coords_center)

        com_coord_loss = args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord)

        batch_loss += len(y_pred)*contact_loss.item()
        batch_by_pred_loss += len(y_pred_by_coord)*contact_by_pred_loss.item()
        batch_distill_loss += len(y_pred_by_coord)*contact_distill_loss.item()
        com_coord_batch_loss += len(com_coord_pred)*com_coord_loss.item()
        pocket_cls_batch_loss += len(pocket_cls_pred)*pocket_cls_loss.item()
        pocket_coord_direct_batch_loss += len(pocket_coord_pred_direct)*pocket_coord_direct_loss.item()
        keepNode_less_5_count += keepNode_less_5

        y_list.append(dis_map)
        y_pred_list.append(y_pred.detach())
        com_coord_list.append(com_coord)
        com_coord_pred_list.append(com_coord_pred.detach())

        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        # protein_len_list.append(batch_len)
        pocket_coord_pred_direct_list.append(pocket_coord_pred_direct.detach())
        pocket_coord_direct_list.append(data.coords_center)
        for i, j in enumerate(batch_len):
            count += 1
            pdb_list.append(data.pdb[i])
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() != 0:
                pred_pocket_center = p_coords_batched_whole.detach()[i][:j][pred_index_bool].mean(dim=0).unsqueeze(0)
                pocket_coord_pred_list.append(pred_pocket_center)
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))
            else: # all the prediction is False, skip
                skip_count += 1
                pred_index_true = pocket_cls_pred[i][:j].sigmoid().unsqueeze(-1)
                pred_index_false = 1. - pred_index_true
                pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
                pred_index_log_prob = torch.log(pred_index_prob)
                pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=args.gs_tau, hard=False)
                pred_index_one_hot_true = pred_index_one_hot[:, 1].unsqueeze(-1)
                pred_pocket_center_gumbel = pred_index_one_hot_true * p_coords_batched_whole[i][:j]
                pred_pocket_center_gumbel_mean = pred_pocket_center_gumbel.sum(dim=0) / pred_index_one_hot_true.sum(dim=0) 
                pocket_coord_pred_list.append(pred_pocket_center_gumbel_mean.unsqueeze(0).detach())
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))


        # real_y_mask_list.append(data.real_y_mask)
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    
    com_coord = torch.cat(com_coord_list)
    com_coord_pred = torch.cat(com_coord_pred_list)

    rmsd = torch.cat(rmsd_list)
    rmsd_2A = torch.cat(rmsd_2A_list)
    rmsd_5A = torch.cat(rmsd_5A_list)
    rmsd_25 = torch.quantile(rmsd, 0.25)
    rmsd_50 = torch.quantile(rmsd, 0.50)
    rmsd_75 = torch.quantile(rmsd, 0.75)
    centroid_dis = torch.cat(centroid_dis_list)
    centroid_dis_2A = torch.cat(centroid_dis_2A_list)
    centroid_dis_5A = torch.cat(centroid_dis_5A_list)
    centroid_dis_25 = torch.quantile(centroid_dis, 0.25)
    centroid_dis_50 = torch.quantile(centroid_dis, 0.50)
    centroid_dis_75 = torch.quantile(centroid_dis, 0.75)

    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)

    if len(pocket_coord_pred_list) > 0:
        pocket_coord_pred = torch.cat(pocket_coord_pred_list)
        pocket_coord = torch.cat(pocket_coord_list)
    pocket_coord_pred_direct = torch.cat(pocket_coord_pred_direct_list)
    pocket_coord_direct = torch.cat(pocket_coord_direct_list)

    pocket_cls_accuracy = (pocket_cls_pred == pocket_cls).sum().item() / len(pocket_cls_pred)
    
    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    metrics.update({"contact_loss":batch_loss/len(y_pred), "contact_by_pred_loss":batch_by_pred_loss/len(y_pred)})
    metrics.update({"com_coord_huber_loss": com_coord_batch_loss/len(com_coord_pred)})
    # Final evaluation metrics
    metrics.update({"rmsd": rmsd.mean().item(), "rmsd < 2A": rmsd_2A.mean().item(), "rmsd < 5A": rmsd_5A.mean().item()})
    metrics.update({"rmsd 25%": rmsd_25.item(), "rmsd 50%": rmsd_50.item(), "rmsd 75%": rmsd_75.item()})
    metrics.update({"centroid_dis": centroid_dis.mean().item(), "centroid_dis < 2A": centroid_dis_2A.mean().item(), "centroid_dis < 5A": centroid_dis_5A.mean().item()})
    metrics.update({"centroid_dis 25%": centroid_dis_25.item(), "centroid_dis 50%": centroid_dis_50.item(), "centroid_dis 75%": centroid_dis_75.item()})
    
    metrics.update({"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_cls_pred_list)})
    metrics.update({"pocket_coord_mse_loss": pocket_coord_direct_batch_loss / len(pocket_coord_pred_direct)})
    metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})

    if len(pocket_coord_pred_list) > 0:
        metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))

    return metrics

@torch.no_grad()
def evaluate_mean_pocket_cls_coord_pocket_pred(args, data_loader, model, com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, relative_k, device, pred_dis=False, info=None, saveFileName=None, use_y_mask=False, skip_y_metrics_evaluation=False, stage=1):
    # contain the ground truth for classiifcation(may not all)
    pocket_coord_list = []
    pocket_coord_pred_list = []
    # contain the ground truth for regression(all)
    pocket_coord_direct_list = []
    pocket_coord_pred_direct_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []

    pdb_list = []

    skip_count = 0
    count = 0
    
    pocket_cls_batch_loss = 0.0
    pocket_coord_direct_batch_loss = 0.0
    keepNode_less_5_count = 0
    for data in tqdm(data_loader, mininterval=args.tqdm_interval):
        data = data.to(device)
        
        pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pocket_coord_pred_direct, keepNode_less_5 = model(data, stage=stage)        
        
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float())
        pocket_coord_direct_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pocket_coord_pred_direct, data.coords_center)

        pocket_cls_batch_loss += len(pocket_cls_pred)*pocket_cls_loss.item()
        pocket_coord_direct_batch_loss += len(pocket_coord_pred_direct)*pocket_coord_direct_loss.item()
        keepNode_less_5_count += keepNode_less_5


        batch_len = protein_out_mask_whole.sum(dim=1).detach()
        # protein_len_list.append(batch_len)
        pocket_coord_pred_direct_list.append(pocket_coord_pred_direct.detach())
        pocket_coord_direct_list.append(data.coords_center)
        for i, j in enumerate(batch_len):
            count += 1
            pdb_list.append(data.pdb[i])
            pocket_cls_list.append(pocket_cls.detach()[i][:j])
            pocket_cls_pred_list.append(pocket_cls_pred.detach()[i][:j].sigmoid().round().int())
            pred_index_bool = (pocket_cls_pred.detach()[i][:j].sigmoid().round().int() == 1)
            if pred_index_bool.sum() != 0:
                pred_pocket_center = p_coords_batched_whole.detach()[i][:j][pred_index_bool].mean(dim=0).unsqueeze(0)
                pocket_coord_pred_list.append(pred_pocket_center)
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))
            else: # all the prediction is False, skip
                skip_count += 1
                pred_index_true = pocket_cls_pred[i][:j].sigmoid().unsqueeze(-1)
                pred_index_false = 1. - pred_index_true
                pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
                pred_index_log_prob = torch.log(pred_index_prob)
                pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=args.gs_tau, hard=False)
                pred_index_one_hot_true = pred_index_one_hot[:, 1].unsqueeze(-1)
                pred_pocket_center_gumbel = pred_index_one_hot_true * p_coords_batched_whole[i][:j]
                pred_pocket_center_gumbel_mean = pred_pocket_center_gumbel.sum(dim=0) / pred_index_one_hot_true.sum(dim=0) 
                pocket_coord_pred_list.append(pred_pocket_center_gumbel_mean.detach().cpu().numpy())
                pocket_coord_list.append(data.coords_center[i].unsqueeze(0))
    # real_y_mask = torch.cat(real_y_mask_list)
    pocket_cls = torch.cat(pocket_cls_list)
    pocket_cls_pred = torch.cat(pocket_cls_pred_list)

    if len(pocket_coord_pred_list) > 0:
        pocket_coord_pred = torch.cat(pocket_coord_pred_list)
        pocket_coord = torch.cat(pocket_coord_list)
    pocket_coord_pred_direct = torch.cat(pocket_coord_pred_direct_list)
    pocket_coord_direct = torch.cat(pocket_coord_direct_list)

    pocket_cls_accuracy = (pocket_cls_pred == pocket_cls).sum().item() / len(pocket_cls_pred)
    
    metrics = {"samples": count, "skip_samples": skip_count, "keepNode < 5": keepNode_less_5_count}
    
    metrics.update({"pocket_cls_bce_loss": pocket_cls_batch_loss / len(pocket_cls_pred_list)})
    metrics.update({"pocket_coord_mse_loss": pocket_coord_direct_batch_loss / len(pocket_coord_pred_direct)})
    metrics.update({"pocket_cls_accuracy": pocket_cls_accuracy})

    if len(pocket_coord_pred_list) > 0:
        metrics.update(pocket_metrics(pocket_coord_pred, pocket_coord))

    return metrics

def gumbel_softmax_no_random(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    gumbels = logits / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
