import torch
import argparse
import os
from utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
import pandas as pd
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Preprocess molecules.')
parser.add_argument("--index_csv", type=str, default="../inference_examples/test.csv",
                    help="Specify the index path for molecules.")
parser.add_argument("--save_mols_dir", type=str, default="../inference_examples/mol",
                    help="Specify where to save the processed pt.")
parser.add_argument("--num_threads", type=int, default=10,
                    help="Multiprocessing threads number")
args = parser.parse_args()
os.system(f'mkdir -p {args.save_mols_dir}')

with open(args.index_csv, 'r') as f:
    content = f.readlines()
info = []
for line in content[1:]:
    smiles, pdb = line.strip().split(',')
    info.append([smiles, pdb])
info = pd.DataFrame(info, columns=['smiles', 'pdb'])

def get_mol_info(idx):
    try:
        smiles = info.iloc[idx].smiles
        mol = read_smiles(smiles)
        mol = generate_conformation(mol)
        molecule_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        
        torch.save([mol, molecule_info], os.path.join(args.save_mols_dir, f'mol_{idx}.pt'))
    except Exception as e:
        print('Failed to read molecule id ', idx, ' We are skipping it. The reason is the exception: ', e)
        
idx = [i for i in range(len(info))]

with Pool(processes=args.num_threads) as p:
    _ = p.map(get_mol_info, idx)

