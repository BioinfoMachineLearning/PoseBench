# conda activate py39
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
from torch import nn

from datetime import datetime
import logging
from io import StringIO
import sys
from scipy.spatial.transform import Rotation


def compute_RMSD(a, b):
    return torch.sqrt((((a-b)**2).sum(axis=-1)).mean())

from rdkit.Geometry import Point3D
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

# new_coords = movable_coords.detach().numpy().astype(np.double)
# write_with_new_coords(mol, new_coords, toFile)

def distance_loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=None, mode=0):
    # compute the current distance matrix
    dis = torch.cdist(protein_nodes_xyz, x)
    # do not consider the distance upper than 10 A0
    dis_clamp = torch.clamp(dis, max=10)
    # may be the author find that L1 loss is the best among these 3 modes
    if mode == 0:
        interaction_loss = ((dis_clamp - y_pred).abs()).sum()
    elif mode == 1:
        interaction_loss = ((dis_clamp - y_pred)**2).sum()
    elif mode == 2:
        # probably not a good choice. x^0.5 has infinite gradient at x=0. added 1e-5 for numerical stability.
        interaction_loss = (((dis_clamp - y_pred).abs() + 1e-5)**0.5).sum()
    # current inter-compound distance matrix
    config_dis = torch.cdist(x, x)
    if LAS_distance_constraint_mask is not None:
        configuration_loss = 1 * (((config_dis-compound_pair_dis_constraint).abs())[LAS_distance_constraint_mask]).sum()
        # basic exlcuded-volume. the distance between compound atoms should be at least 1.22Å
        configuration_loss += 2 * ((1.22 - config_dis).relu()).sum()
    else:
        configuration_loss = 1 * ((config_dis-compound_pair_dis_constraint).abs()).sum()
    # In first 500 epochs, only consider the interaction loss.
    # In the following epochs, consider both interaction loss and configuration loss(increased weight)
    if epoch < 500:
        loss = interaction_loss
    else:
        loss = 1 * (interaction_loss + 5e-3 * (epoch - 500) * configuration_loss)
    return loss, (interaction_loss.item(), configuration_loss.item())


def distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz, 
                        compound_pair_dis_constraint, total_epoch=5000, loss_function=distance_loss_function, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    # pocket center
    c_pred = protein_nodes_xyz.mean(axis=0)
    # random initialization. center at the protein center.
    # the center of block has a random shift of -5~5Å, drawn from the uniform distribution, in all three axes.
    x = (5 * (2 * torch.rand(coords.shape) - 1) + c_pred.reshape(1, 3).detach())
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=0.1)
    # optimizer = torch.optim.LBFGS([x], lr=0.01)
    loss_list = []
    rmsd_list = []
    if show_progress:
        it = tqdm(range(total_epoch))
    else:
        it = range(total_epoch)
    for epoch in it:
        optimizer.zero_grad()
        loss, (interaction_loss, configuration_loss) = loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=LAS_distance_constraint_mask, mode=mode)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        rmsd = compute_RMSD(coords, x.detach())
        rmsd_list.append(rmsd.item())
        # break
    return x, loss_list, rmsd_list

def get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, n_repeat=1, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    info = []
    if show_progress:
        it = tqdm(range(n_repeat))
    else:
        it = range(n_repeat)
    for repeat in it:
        # random initialization.
        # x = torch.rand(coords.shape, requires_grad=True)
        # mode = 0
        x, loss_list, rmsd_list = distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz, 
                            compound_pair_dis_constraint, LAS_distance_constraint_mask=LAS_distance_constraint_mask, mode=mode, show_progress=False)
        # rmsd = compute_rmsd(coords.detach().cpu().numpy(), movable_coords.detach().cpu().numpy())
        # print(coords, movable_coords)
        # rmsd = compute_rmsd(coords, x.detach())
        rmsd = rmsd_list[-1]
        try:
            info.append([repeat, rmsd, float(loss_list[-1]), x.detach().cpu().numpy()])
        except:
            info.append([repeat, rmsd, 0, x.detach().cpu().numpy()])
    info = pd.DataFrame(info, columns=['repeat', 'rmsd', 'loss', 'coords'])
    return info


def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    Chem.WrapLogs()
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