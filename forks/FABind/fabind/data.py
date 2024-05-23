import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset
from utils.utils import construct_data_from_graph_gvp_mean
import lmdb
import pickle

class FABindDataSet(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, args=None,
                use_whole_protein=False, compound_coords_init_mode=None, seed=42, pre=None,
                transform=None, pre_transform=None, pre_filter=None, noise_for_predicted_pocket=5.0, test_random_rotation=False, pocket_idx_no_noise=True, use_esm2_feat=False):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        # this will call the process function to save the data, protein_dict and compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.compound_rdkit_coords = torch.load(self.processed_paths[3])
        self.protein_dict = lmdb.open(self.processed_paths[1], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.compound_dict = lmdb.open(self.processed_paths[2], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        if use_esm2_feat:
            self.protein_esm2_feat = lmdb.open(self.processed_paths[4], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.compound_coords_init_mode = compound_coords_init_mode
        self.add_noise_to_com = add_noise_to_com
        self.noise_for_predicted_pocket = noise_for_predicted_pocket
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.use_whole_protein = use_whole_protein
        self.test_random_rotation = test_random_rotation
        self.pocket_idx_no_noise = pocket_idx_no_noise
        self.use_esm2_feat = use_esm2_feat
        self.seed = seed
        self.args = args
        self.pre = pre
    
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein_1d_3d.lmdb', 'compound_LAS_edge_index.lmdb', 'compound_rdkit_coords.pt', 'esm2_t33_650M_UR50D.lmdb']

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        pocket_com = line['pocket_com']
        use_compound_com = line['use_compound_com']
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else self.use_whole_protein
        group = line['group'] if "group" in line.index else 'train'
        if group == 'train' and use_compound_com:
            add_noise_to_com = self.add_noise_to_com
        elif group == 'train' and not use_compound_com:
            add_noise_to_com = self.noise_for_predicted_pocket
        else:
            add_noise_to_com = None

        if group == 'train':
            random_rotation = True
        elif group == 'test' and self.test_random_rotation:
            random_rotation = True
        else:
            random_rotation = False

        protein_name = line['protein_name'] # pdb id
        if self.proteinMode == 0:
            with self.protein_dict.begin() as txn:
                protein_node_xyz, protein_seq= pickle.loads(txn.get(protein_name.encode()))
            if self.use_esm2_feat:
                with self.protein_esm2_feat.begin() as txn:
                    protein_esm2_feat = pickle.loads(txn.get(protein_name.encode()))
            else:
                protein_esm2_feat = None

        name = line['compound_name']
        rdkit_coords = self.compound_rdkit_coords[name]
        # compound embedding from torchdrug
        with self.compound_dict.begin() as txn:
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution, LAS_edge_index = pickle.loads(txn.get(name.encode()))

        if self.proteinMode == 0:
            data, input_node_list, keepNode = construct_data_from_graph_gvp_mean(self.args, protein_node_xyz, protein_seq, 
                                coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, LAS_edge_index, rdkit_coords, compound_coords_init_mode=self.compound_coords_init_mode, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                                pocket_radius=self.pocket_radius, add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, pdb_id=name, group=group, seed=self.seed, data_path=self.pre, 
                                use_compound_com_as_pocket=use_compound_com, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode, random_rotation=random_rotation, pocket_idx_no_noise=self.pocket_idx_no_noise,
                                protein_esm2_feat=protein_esm2_feat)


        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}'
        data.group = group

        return data

def get_data(args, logger, addNoise=None, use_whole_protein=False, compound_coords_init_mode='pocket_center_rdkit', pre="/PDBbind_data/pdbbind2020"):
    if args.data == "0":
        logger.log_message(f"Loading dataset")
        logger.log_message(f"compound feature based on torchdrug")
        logger.log_message(f"protein feature based on esm2")
        add_noise_to_com = float(addNoise) if addNoise else None

        new_dataset = FABindDataSet(f"{pre}/dataset", add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, compound_coords_init_mode=compound_coords_init_mode, pocket_radius=args.pocket_radius, noise_for_predicted_pocket=args.noise_for_predicted_pocket, 
                                            test_random_rotation=args.test_random_rotation, pocket_idx_no_noise=args.pocket_idx_no_noise, use_esm2_feat=args.use_esm2_feat, seed=args.seed, pre=pre, args=args)
        # load compound features extracted using torchdrug.
        # c_length: number of atoms in the compound
        # This filter may cause some samples to be filtered out. So the actual number of samples is less than that in the original papers.
        train_tmp = new_dataset.data.query("c_length < 100 and native_num_contact > 5 and group =='train' and use_compound_com").reset_index(drop=True)
        valid_test_tmp = new_dataset.data.query("(group == 'valid' or group == 'test') and use_compound_com").reset_index(drop=True)
        new_dataset.data = pd.concat([train_tmp, valid_test_tmp], axis=0).reset_index(drop=True)
        d = new_dataset.data
        only_native_train_index = d.query("group =='train'").index.values
        train = new_dataset[only_native_train_index]
        valid_index = d.query("group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("group =='test'").index.values
        test = new_dataset[test_index]

    return train, valid, test