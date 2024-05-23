import esm
import torch
from tqdm import tqdm
import os
import argparse
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature


parser = argparse.ArgumentParser(description='Preprocess protein.')
parser.add_argument("--pdb_file_dir", type=str, default="../inference_examples/pdb_files",
                    help="Specify the pdb data path.")
parser.add_argument("--save_pt_dir", type=str, default="../inference_examples",
                    help="Specify where to save the processed pt.")
parser.add_argument("--cuda_device_index", type=int, default=0,
                    help="Specify the cuda device index.")
args = parser.parse_args()

esm2_dict = {}
protein_dict = {}

device = (f"cuda:{args.cuda_device_index}" if args.cuda_device_index is not None else "cuda") if torch.cuda.is_available() else "cpu"

# Load ESM-2 model with different sizes
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

for pdb_file in tqdm(os.listdir(args.pdb_file_dir)):
    pdb = pdb_file.split(".")[0].split("_holo_aligned_")[0]

    pdb_filepath = os.path.join(args.pdb_file_dir, pdb_file)
    if os.path.isdir(pdb_filepath):
        pdb_filepath = os.path.join(pdb_filepath, f"{pdb}_protein.pdb")
    protein_structure = extract_protein_structure(pdb_filepath)
    protein_structure['name'] = pdb
    esm2_dict[pdb] = extract_esm_feature(protein_structure, model, batch_converter, device)
    protein_dict[pdb] = protein_structure

torch.save([esm2_dict, protein_dict], os.path.join(args.save_pt_dir, 'processed_protein.pt'))
