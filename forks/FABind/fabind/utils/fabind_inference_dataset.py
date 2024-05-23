from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature
from utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
from torch_geometric.data import HeteroData
import torch


class InferenceDataset(Dataset):
    def __init__(self, index_csv, pdb_file_dir, preprocess_dir):
        super().__init__(None, None, None, None)
        # extract pair index from csv file
        with open(index_csv, 'r') as f:
            content = f.readlines()
        info = []
        for line in content[1:]:
            smiles, pdb = line.strip().split(',')
            info.append([smiles, pdb])
        info = pd.DataFrame(info, columns=['smiles', 'pdb'])

        # read preprocessed data
        self.protein_feature, self.protein_structure = torch.load(os.path.join(preprocess_dir, 'processed_protein.pt'))
        
        self.data = []
        for i in tqdm(range(len(info))):
            input_dict = {}

            # obtain compound
            try:
                smiles = info.iloc[i].smiles
                mol, molecule_info = torch.load(os.path.join(preprocess_dir, 'mol', f'mol_{i}.pt'))
            except:
                print('\nFailed to read molecule id ', i, ' We are skipping it.')
                continue

            # obtain proteins
            try:
                # obtain protein
                protein_structure = self.protein_structure[info.iloc[i].pdb]                
                # obtain protein esm feature
                protein_esm_feature = self.protein_feature[info.iloc[i].pdb]
            except:
                print('\nFailed to read protein pdb ', info.iloc[i].pdb, ' We are skipping it.')
                continue

            # add to input dict
            input_dict['protein_esm_feature'] = protein_esm_feature
            input_dict['protein_structure'] = protein_structure
            input_dict['molecule'] = mol
            input_dict['molecule_smiles'] = smiles
            input_dict['molecule_info'] = molecule_info
            self.data.append(input_dict)


    def len(self):
        return len(self.data)

    def get(self, idx):
        input_dict = self.data[idx]
        protein_node_xyz = torch.tensor(input_dict['protein_structure']['coords'])[:, 1]
        protein_seq = input_dict['protein_structure']['seq']
        protein_esm_feature = input_dict['protein_esm_feature']
        smiles = input_dict['molecule_smiles']
        rdkit_coords, compound_node_features, input_atom_edge_list, LAS_edge_index = input_dict['molecule_info']
        
        n_protein_whole = protein_node_xyz.shape[0]
        n_compound = compound_node_features.shape[0]

        data = HeteroData()

        data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)
        protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)
        coords_init = rdkit_coords - rdkit_coords.mean(axis=0)


        
        # compound graph
        data['compound'].node_feats = compound_node_features.float()
        data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
        data['compound'].node_coords = coords_init
        data['compound'].rdkit_coords = coords_init
        data['compound'].smiles = smiles
        data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()
        data['LAS_edge_list'].x = (LAS_edge_index + 1).clone().t()

        data.node_xyz_whole = protein_node_xyz
        data.seq_whole = protein_seq
        data.idx = idx
        data.uid = input_dict['protein_structure']['name']
        data.mol = input_dict['molecule']

        # complex whole graph
        data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0), # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3), 
                protein_node_xyz
            ), dim=0
        ).float()
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
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

        data['protein_whole'].node_feats = protein_esm_feature
        
        return data


